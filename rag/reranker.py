"""Rerankers: lexical (default, no model download), cross-encoder (optional), or none."""
from __future__ import annotations

import logging
import math
from typing import Optional

from config import Settings
from .vector_store import Hit

log = logging.getLogger(__name__)


class Reranker:
    def rerank(self, query: str, hits: list[Hit], prior: dict[str, float]) -> list[Hit]:
        raise NotImplementedError


class NoReranker(Reranker):
    def rerank(self, query, hits, prior):
        for h in hits:
            h.score = prior.get(h.id, 0.0)
        return sorted(hits, key=lambda h: h.score, reverse=True)


class LexicalReranker(Reranker):
    """final = 0.45 * fused-rank prior + 0.35 * query-term overlap + 0.20 * vector similarity."""

    def rerank(self, query, hits, prior):
        top = max(prior.values(), default=1.0) or 1.0
        for h in hits:
            h.score = 0.45 * (prior.get(h.id, 0.0) / top) + 0.35 * h.lexical + 0.20 * h.vector_sim
        return sorted(hits, key=lambda h: h.score, reverse=True)


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query, hits, prior):
        if not hits:
            return hits
        scores = self.model.predict([(query, h.text[:1500]) for h in hits])
        top = max(prior.values(), default=1.0) or 1.0
        for h, s in zip(hits, scores):
            ce = 1 / (1 + math.exp(-float(s)))
            h.score = 0.7 * ce + 0.3 * (prior.get(h.id, 0.0) / top)
        return sorted(hits, key=lambda h: h.score, reverse=True)


def build_reranker(settings: Settings) -> Reranker:
    kind = settings.reranker
    if kind == "none":
        return NoReranker()
    if kind == "cross-encoder":
        try:
            return CrossEncoderReranker(settings.cross_encoder_model)
        except Exception as exc:
            log.warning("Cross-encoder unavailable (%s); falling back to lexical reranking.", exc)
    return LexicalReranker()
