"""Vector store abstraction + ChromaDB implementation, and a JSON document registry.

Swap in FAISS/Qdrant by implementing `VectorStore` (add / query / get / delete_doc / count).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from config import Settings
from ingestion.models import Chunk
from .embeddings import Embedder

log = logging.getLogger(__name__)


@dataclass
class Hit:
    id: str
    text: str
    metadata: dict[str, Any]
    vector_sim: float = 0.0      # cosine similarity (0..1) from the vector index, 0 if not retrieved by vectors
    lexical: float = 0.0         # fraction of query terms present
    score: float = 0.0           # final ranking score


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None: ...
    @abstractmethod
    def query(self, query_embedding: list[float], n: int, doc_ids: Optional[list[str]] = None) -> list[Hit]: ...
    @abstractmethod
    def get_all(self, doc_ids: Optional[list[str]] = None) -> list[Hit]: ...
    @abstractmethod
    def delete_doc(self, doc_id: str) -> None: ...
    @abstractmethod
    def count(self) -> int: ...
    @abstractmethod
    def reset(self) -> None: ...


def _where(doc_ids: Optional[list[str]]) -> Optional[dict]:
    if not doc_ids:
        return None
    return {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": list(doc_ids)}}


class ChromaStore(VectorStore):
    def __init__(self, settings: Settings, embedder: Embedder, persist: bool = True):
        import chromadb

        self.embedder = embedder
        if persist:
            self.client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        else:
            self.client = chromadb.EphemeralClient()
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", embedder.name).strip("_")[:30]
        suffix = hashlib.md5(embedder.name.encode()).hexdigest()[:6]
        self.name = f"docs_{slug}_{suffix}"  # new embedding model -> new collection (dimensions differ)
        self.col = self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})
        self.version = 0  # bumped on every mutation; used to invalidate BM25 caches

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        for i in range(0, len(chunks), 64):
            batch = chunks[i:i + 64]
            embs = self.embedder.embed_documents([c.embed_text for c in batch])
            self.col.upsert(ids=[c.id for c in batch], embeddings=embs,
                            documents=[c.text for c in batch], metadatas=[c.metadata for c in batch])
        self.version += 1

    def query(self, query_embedding, n, doc_ids=None):
        total = self.count()
        if total == 0:
            return []
        res = self.col.query(query_embeddings=[query_embedding], n_results=min(n, total), where=_where(doc_ids),
                             include=["documents", "metadatas", "distances"])
        hits = []
        for i, cid in enumerate(res["ids"][0]):
            sim = max(0.0, 1.0 - float(res["distances"][0][i]))  # cosine distance -> similarity
            hits.append(Hit(cid, res["documents"][0][i], res["metadatas"][0][i], vector_sim=sim))
        return hits

    def get_all(self, doc_ids=None):
        res = self.col.get(where=_where(doc_ids), include=["documents", "metadatas"])
        return [Hit(i, d, m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]

    def delete_doc(self, doc_id: str) -> None:
        self.col.delete(where={"doc_id": doc_id})
        self.version += 1

    def count(self) -> int:
        return self.col.count()

    def reset(self) -> None:
        self.client.delete_collection(self.name)
        self.col = self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})
        self.version += 1


def build_vector_store(settings: Settings, embedder: Embedder, persist: bool = True) -> VectorStore:
    if settings.vector_store == "chroma":
        return ChromaStore(settings, embedder, persist=persist)
    raise ValueError(f"Unsupported VECTOR_STORE '{settings.vector_store}'. Implement VectorStore for FAISS/Qdrant.")


class DocumentRegistry:
    """Per-document JSON: status, warnings, structured tables, image records, page renders."""

    def __init__(self, settings: Settings):
        self.dir = settings.registry_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, doc_id: str) -> Path:
        return self.dir / f"{doc_id}.json"

    def save(self, info: dict[str, Any]) -> None:
        tmp = self._path(info["doc_id"]).with_suffix(".tmp")
        tmp.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path(info["doc_id"]))

    def get(self, doc_id: str) -> Optional[dict[str, Any]]:
        p = self._path(doc_id)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def list(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.dir.glob("*.json")):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                log.warning("Corrupted registry entry %s ignored", p.name)
        return out

    def delete(self, doc_id: str) -> None:
        self._path(doc_id).unlink(missing_ok=True)
