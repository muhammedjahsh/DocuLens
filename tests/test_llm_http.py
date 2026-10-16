"""Exercises the Ollama and OpenAI-compatible clients against a local mock server that
speaks their wire formats (no real model needed)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from rag.llm import LLMError, OllamaLLM, OpenAILLM, build_llm, build_vision_llm
from rag.pipeline import RAGService
from rag.embeddings import HashingEmbedder
from tests.conftest import make_settings

SEEN: list[dict] = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
        if body.get("model") == "missing":
            self.send_response(404); self.end_headers(); self.wfile.write(b"{}"); return
        user = body["messages"][-1]["content"]
        text = "The total revenue is 5,250,000 [S1]." if "QUESTION" in str(user) else "ok"
        payload = ({"message": {"content": text}} if self.path == "/api/chat"
                   else {"choices": [{"message": {"content": text}}]})
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps(payload).encode())


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_ollama_request_format_and_images(server, settings, tmp_path):
    settings.ollama_url = server
    img = tmp_path / "a.png"; img.write_bytes(b"\x89PNG")
    SEEN.clear()
    out = OllamaLLM(settings, "phi3:mini").chat([{"role": "user", "content": "hi"}], images=[str(img)])
    req = SEEN[-1]["body"]
    assert out == "ok" and req["model"] == "phi3:mini" and req["stream"] is False
    assert req["options"]["num_ctx"] == settings.num_ctx and req["messages"][-1]["images"]


def test_ollama_missing_model_message(server, settings):
    settings.ollama_url = server
    with pytest.raises(LLMError, match="ollama pull missing"):
        OllamaLLM(settings, "missing").chat([{"role": "user", "content": "x"}])


def test_ollama_down_message(settings):
    settings.ollama_url = "http://127.0.0.1:9"
    with pytest.raises(LLMError, match="ollama serve"):
        OllamaLLM(settings, "phi3:mini").chat([{"role": "user", "content": "x"}])


def test_openai_format_key_and_vision(server, settings, tmp_path):
    settings.openai_base_url, settings.openai_api_key = server, "sk-test-123456789012"
    img = tmp_path / "a.png"; img.write_bytes(b"\x89PNG")
    SEEN.clear()
    OpenAILLM(settings, "gpt-4o-mini").chat([{"role": "user", "content": "describe"}], images=[str(img)])
    r = SEEN[-1]
    assert r["path"] == "/chat/completions" and r["auth"] == "Bearer sk-test-123456789012"
    parts = r["body"]["messages"][-1]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_openai_requires_key(settings):
    settings.openai_api_key = ""
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        OpenAILLM(settings, "gpt-4o-mini")


def test_vision_provider_selection(settings):
    settings.llm_provider, settings.vision_provider = "ollama", "auto"
    main = build_llm(settings)
    assert build_vision_llm(settings, main) is None          # Phi-3 is text-only
    settings.vision_provider, settings.vision_model = "ollama", "llava:7b"
    assert build_vision_llm(settings, main).supports_vision


def test_full_rag_answer_through_ollama_client(server, tmp_path_factory, samples):
    s = make_settings(tmp_path_factory.mktemp("llm"))
    s.llm_provider, s.ollama_url = "ollama", server
    svc = RAGService(s, embedder=HashingEmbedder())
    svc.ingest_bytes(samples["acme"].name, samples["acme"].read_bytes())
    SEEN.clear()
    a = svc.ask("What is the total revenue shown in the table?")
    assert a.text.startswith("The total revenue is 5,250,000") and a.cited_sources and not a.warnings
    prompt = SEEN[-1]["body"]["messages"][-1]["content"]
    assert "VERIFIED CALCULATIONS" in prompt and "5,250,000" in prompt and "[S1]" in prompt
