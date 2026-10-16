"""LLM abstraction: Ollama (default; Phi-3 Mini), Hugging Face Transformers, OpenAI-compatible API,
or 'extractive' (no generation, used for offline tests and as a graceful fallback).

Compatibility notes
* Phi-3 Mini (3.8B) is TEXT-ONLY. Image questions need a separate vision model (VISION_MODEL,
  e.g. llava:7b / llama3.2-vision on Ollama, or gpt-4o-mini via OpenAI).
* Phi-3-mini-4k has a 4096-token window: keep MAX_CONTEXT_CHARS <= ~7000.
"""
from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Optional

import requests

from config import Settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


def _b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


class BaseLLM:
    name = "base"
    supports_vision = False
    generates = True

    def chat(self, messages: list[dict], images: Optional[list[str]] = None) -> str:
        raise NotImplementedError


class ExtractiveLLM(BaseLLM):
    name = "extractive (no LLM)"
    generates = False

    def chat(self, messages, images=None):
        raise LLMError("The extractive provider does not generate text.")


class OllamaLLM(BaseLLM):
    def __init__(self, settings: Settings, model: str, vision: bool = False):
        self.s, self.model, self.supports_vision = settings, model, vision
        self.name = f"ollama:{model}"

    def chat(self, messages, images=None):
        msgs = [dict(m) for m in messages]
        if images:
            last = next(m for m in reversed(msgs) if m["role"] == "user")
            last["images"] = [_b64(p) for p in images]
        try:
            r = requests.post(
                f"{self.s.ollama_url.rstrip('/')}/api/chat",
                json={"model": self.model, "messages": msgs, "stream": False,
                      "options": {"temperature": self.s.temperature, "num_ctx": self.s.num_ctx,
                                  "num_predict": self.s.max_new_tokens}},
                timeout=self.s.request_timeout)
        except requests.ConnectionError as exc:
            raise LLMError(f"Cannot reach Ollama at {self.s.ollama_url}. Start it with `ollama serve` "
                           f"and pull the model with `ollama pull {self.model}`.") from exc
        except requests.Timeout as exc:
            raise LLMError("The model timed out. Try a smaller model or raise REQUEST_TIMEOUT.") from exc
        if r.status_code == 404:
            raise LLMError(f"Model '{self.model}' is not installed in Ollama. Run `ollama pull {self.model}`.")
        if not r.ok:
            raise LLMError(f"Ollama error {r.status_code}: {r.text[:200]}")
        return r.json()["message"]["content"].strip()


class OpenAILLM(BaseLLM):
    def __init__(self, settings: Settings, model: str, vision: bool = True):
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set (put it in your environment or .env file).")
        self.s, self.model, self.supports_vision = settings, model, vision
        self.name = f"openai:{model}"

    def chat(self, messages, images=None):
        msgs = [dict(m) for m in messages]
        if images:
            last = next(m for m in reversed(msgs) if m["role"] == "user")
            parts = [{"type": "text", "text": last["content"]}]
            for p in images:
                mime = mimetypes.guess_type(p)[0] or "image/png"
                parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{_b64(p)}"}})
            last["content"] = parts
        try:
            r = requests.post(f"{self.s.openai_base_url.rstrip('/')}/chat/completions",
                              headers={"Authorization": f"Bearer {self.s.openai_api_key}"},
                              json={"model": self.model, "messages": msgs, "temperature": self.s.temperature,
                                    "max_tokens": self.s.max_new_tokens},
                              timeout=self.s.request_timeout)
        except requests.RequestException as exc:
            raise LLMError(f"OpenAI request failed: {type(exc).__name__}") from exc
        if r.status_code == 401:
            raise LLMError("OpenAI rejected the API key (401).")
        if not r.ok:
            raise LLMError(f"OpenAI error {r.status_code}: {r.text[:200]}")
        return r.json()["choices"][0]["message"]["content"].strip()


class HuggingFaceLLM(BaseLLM):
    """Local transformers inference (needs `pip install transformers torch accelerate`)."""

    def __init__(self, settings: Settings):
        self.s = settings
        self.name = f"hf:{settings.hf_model}"
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            try:
                import torch
                from transformers import pipeline
            except ImportError as exc:
                raise LLMError("Install transformers, torch and accelerate to use LLM_PROVIDER=huggingface.") from exc
            self._pipe = pipeline("text-generation", model=self.s.hf_model, device_map="auto",
                                  torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        return self._pipe

    def chat(self, messages, images=None):
        msgs = [dict(m) for m in messages]
        if msgs and msgs[0]["role"] == "system":  # fold system into first user turn (template-safe)
            sys_txt = msgs.pop(0)["content"]
            first_user = next(m for m in msgs if m["role"] == "user")
            first_user["content"] = f"{sys_txt}\n\n{first_user['content']}"
        out = self._load()(msgs, max_new_tokens=self.s.max_new_tokens, do_sample=self.s.temperature > 0,
                           temperature=max(self.s.temperature, 1e-3), return_full_text=False)
        return out[0]["generated_text"].strip()


def build_llm(settings: Settings) -> BaseLLM:
    p = settings.llm_provider.lower()
    if p == "extractive":
        return ExtractiveLLM()
    if p == "ollama":
        return OllamaLLM(settings, settings.llm_model)
    if p == "openai":
        return OpenAILLM(settings, settings.llm_model)
    if p in ("huggingface", "hf"):
        return HuggingFaceLLM(settings)
    raise ValueError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'")


def build_vision_llm(settings: Settings, main: BaseLLM) -> Optional[BaseLLM]:
    p = settings.vision_provider.lower()
    if p == "auto":
        return main if isinstance(main, OpenAILLM) else None
    if p == "none":
        return None
    if p == "ollama":
        return OllamaLLM(settings, settings.vision_model, vision=True)
    if p == "openai":
        return OpenAILLM(settings, settings.vision_model, vision=True)
    raise ValueError(f"Unknown VISION_PROVIDER '{settings.vision_provider}'")
