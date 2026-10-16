"""Embedding backends behind one interface.

* SentenceTransformerEmbedder - production default (BAAI/bge-small-en-v1.5).
* HashingEmbedder - dependency-free lexical embedding used for unit tests / offline smoke
  tests ONLY. It has no semantic understanding (no synonyms), so do not use it for real work.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Protocol

import numpy as np

from config import Settings

log = logging.getLogger(__name__)
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run `pip install sentence-transformers` "
                "or set EMBEDDING_BACKEND=hashing for a lexical-only test mode."
            ) from exc
        self.name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._bge = "bge" in model_name.lower()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        q = (_BGE_QUERY_PREFIX + text) if self._bge else text
        return self.model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0].tolist()


class HashingEmbedder:
    name = "hashing-384"
    dim = 384

    @staticmethod
    def _tokens(text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in self._tokens(text):
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = np.linalg.norm(v)
        return (v / n).tolist() if n else v.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hashing":
        log.warning("Using HashingEmbedder: lexical test mode, no semantic search.")
        return HashingEmbedder()
    return SentenceTransformerEmbedder(settings.embedding_model)
