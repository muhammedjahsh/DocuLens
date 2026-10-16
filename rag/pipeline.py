"""RAGService: the single entry point used by the UI, tests and the evaluation script.
Ingestion (validate -> extract -> chunk -> embed -> store) and querying
(analyse -> retrieve -> calculate -> generate -> cite) are independent, injectable parts."""
from __future__ import annotations

import logging
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from config import Settings, get_settings
from ingestion.models import ExtractionResult
from ingestion.ocr_processor import OCREngine
from ingestion.pipeline import compute_doc_id, extract_document
from ingestion.models import TableRecord
from utils.validation import ValidationError, validate_upload
from .answer_generator import Answer, AnswerGenerator, INSUFFICIENT
from .chunking import build_chunks
from .embeddings import Embedder, build_embedder
from .llm import BaseLLM, build_llm, build_vision_llm
from .reranker import build_reranker
from .retriever import Retriever, analyze_query
from .table_calc import answer_from_tables, detect_ops
from .vector_store import DocumentRegistry, Hit, VectorStore, build_vector_store

log = logging.getLogger(__name__)
_GENERIC_NAME_TOKENS = {"report", "annual", "final", "draft", "copy", "document", "doc", "docx", "pdf", "file", "scan", "scanned",
                        "image", "img", "new", "old", "sample", "test", "page", "data", "the", "and", "txt", "png", "jpg", "jpeg"}
_SUMMARY_WORDS = {"summari", "summary", "overview", "main", "finding", "point", "key", "takeaway", "brief", "gist", "idea", "tldr", "tl", "dr"}

Progress = Callable[[str], None]


class RAGService:
    def __init__(self, settings: Optional[Settings] = None, *, embedder: Optional[Embedder] = None,
                 store: Optional[VectorStore] = None, llm: Optional[BaseLLM] = None,
                 vision_llm: Optional[BaseLLM] = None, persist: bool = True):
        self.s = settings or get_settings()
        self.s.ensure_dirs()
        self.embedder = embedder or build_embedder(self.s)
        self.store = store or build_vector_store(self.s, self.embedder, persist=persist)
        self.registry = DocumentRegistry(self.s)
        self.ocr = OCREngine(self.s)
        self.llm = llm or build_llm(self.s)
        self.vision = vision_llm if vision_llm is not None else build_vision_llm(self.s, self.llm)
        self.retriever = Retriever(self.s, self.store, self.embedder, build_reranker(self.s))
        self.generator = AnswerGenerator(self.s, self.llm, self.vision)

    # ------------------------------------------------------------------ ingestion
    def ingest_bytes(self, name: str, data: bytes, progress: Optional[Progress] = None) -> dict:
        say = progress or (lambda m: None)
        v = validate_upload(name, data, self.s)  # raises ValidationError
        doc_id = compute_doc_id(data)
        existing = self.registry.get(doc_id)
        if existing and existing.get("status") == "indexed":
            existing["duplicate"] = True
            return existing
        folder = self.s.upload_dir / doc_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / v.safe_name
        path.write_bytes(data)
        os.chmod(path, 0o600)  # stored as inert data; never executed
        try:
            say("Extracting content…")
            captioner = self._captioner() if self.s.caption_images_at_ingest else None
            res = extract_document(path, doc_id=doc_id, filename=v.safe_name, settings=self.s, ocr=self.ocr, captioner=captioner)
            say("Chunking…")
            chunks = build_chunks(res, self.s)
            if not chunks:
                raise ValueError(f"'{v.safe_name}' produced no searchable content.")
            say(f"Embedding {len(chunks)} chunks…")
            self.store.add(chunks)
        except Exception:
            self._purge_files(doc_id)
            self.registry.delete(doc_id)
            raise
        info = self._info(res, len(chunks), v.size)
        self.registry.save(info)
        return info

    def _captioner(self) -> Optional[Callable[[str], str]]:
        if not self.vision:
            return None
        return lambda path: self.vision.chat(
            [{"role": "user", "content": "Describe this image factually and concisely. If it is a chart or diagram, "
                                         "state its type, axes, labels and the main values or relationships you can read. "
                                         "Do not guess unreadable details."}], images=[path])

    @staticmethod
    def _info(res: ExtractionResult, n_chunks: int, size: int) -> dict:
        return {
            "doc_id": res.doc_id, "filename": res.filename, "file_type": res.file_type, "size": size,
            "n_pages": res.n_pages, "n_chunks": n_chunks, "n_tables": len(res.tables), "n_images": len(res.images),
            "ocr_pages": res.ocr_pages, "warnings": res.warnings, "status": "indexed",
            "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tables": [t.to_dict() for t in res.tables], "images": [i.to_dict() for i in res.images],
            "page_images": {str(k): v for k, v in res.page_images.items()},
        }

    def _purge_files(self, doc_id: str) -> None:
        for base in (self.s.upload_dir, self.s.pages_dir, self.s.images_dir):
            shutil.rmtree(base / doc_id, ignore_errors=True)

    def list_documents(self) -> list[dict]:
        return self.registry.list()

    def remove_document(self, doc_id: str) -> None:
        self.store.delete_doc(doc_id)
        self.registry.delete(doc_id)
        self._purge_files(doc_id)

    def remove_all(self) -> None:
        for d in self.registry.list():
            self.remove_document(d["doc_id"])
        self.store.reset()

    def page_preview(self, doc_id: str, page: int) -> Optional[str]:
        info = self.registry.get(doc_id) or {}
        p = (info.get("page_images") or {}).get(str(page))
        return p if p and Path(p).exists() else None

    def config_summary(self) -> dict:
        return {"LLM": self.llm.name, "Vision model": self.vision.name if self.vision else "not configured",
                "Embeddings": self.embedder.name, "Vector DB": f"{self.s.vector_store} ({self.store.count()} chunks)",
                "Reranker": self.s.reranker, "OCR": "Tesseract" if self.ocr.available else "unavailable"}

    # ------------------------------------------------------------------ querying
    def ask(self, question: str, history: Optional[list[dict]] = None, doc_ids: Optional[list[str]] = None) -> Answer:
        history = history or []
        t0 = time.perf_counter()
        docs = {d["doc_id"]: d for d in self.registry.list() if d.get("status") == "indexed"}
        if doc_ids:
            docs = {k: v for k, v in docs.items() if k in doc_ids}
        if not docs:
            return Answer(text="Please upload and select at least one document first.", insufficient=True)
        ids = list(docs)
        names = {k: v["filename"] for k, v in docs.items()}

        standalone = self.generator.condense_question(question, history)
        notes: list[str] = []
        hinted = self._doc_hint(standalone, docs)
        if hinted and len(hinted) < len(docs):
            docs = {k: docs[k] for k in hinted}
            ids, names = list(docs), {k: v["filename"] for k, v in docs.items()}
            notes.append("Searched only the document(s) named in your question: " + ", ".join(names.values()) + ".")
        plan = analyze_query(standalone)
        t_cond = time.perf_counter()

        # ---- whole-document / page-range summary --------------------------------
        if plan.is_summary and not (set(plan.terms) - _SUMMARY_WORDS - {"pdf", "docx", "page"} - {str(p) for p in plan.pages}):
            hits = self.store.get_all(ids)
            if plan.pages:
                hits = [h for h in hits if h.metadata.get("page") in plan.pages]
            if hits:
                per_doc: dict[str, list[Hit]] = {}
                for h in hits:
                    per_doc.setdefault(h.metadata["filename"], []).append(h)
                ans = self.generator.summarize(per_doc)
                ans.standalone_question, ans.notes = standalone, notes
                ans.timings.update(condense_s=t_cond - t0, total_s=time.perf_counter() - t0)
                return ans

        t1 = time.perf_counter()
        rr = self.retriever.retrieve(standalone, ids, doc_names=names)
        t2 = time.perf_counter()
        hits = list(rr.hits)

        # ---- deterministic table maths ----------------------------------------
        calc = []
        if rr.plan.is_table or detect_ops(standalone):
            tables = self._candidate_tables(hits, docs, rr.plan.is_table)
            calc = answer_from_tables(standalone, tables)
            have = {h.metadata.get("table_id") for h in hits}
            for c in calc:  # make sure the table behind each calculation is citable evidence
                if c.table_id not in have and not c.needs_clarification:
                    rec = next((t for t in tables if t.table_id == c.table_id), None)
                    if rec:
                        hits.append(Hit(id=f"calc:{c.table_id}", text=f"Table {c.table_id}\n" + rec.to_markdown(rec.rows[:40]),
                                        metadata={"doc_id": next((k for k, v in docs.items() if v["filename"] == rec.filename), ""),
                                                  "filename": rec.filename, "page": rec.page if rec.page else -1,
                                                  "content_type": "table", "heading": rec.heading, "table_id": rec.table_id,
                                                  "image_path": "", "page_image": ""}))
        t3 = time.perf_counter()

        images = self._images_for_vision(hits, rr.plan) if (self.vision or rr.plan.is_visual) else []
        ans = self.generator.answer(standalone, hits, rr.sufficient or bool(calc), history, calc, images, sorted(rr.plan.intents))
        ans.standalone_question, ans.notes = standalone, notes
        ans.timings.update(condense_s=t1 - t0, retrieval_s=t2 - t1, calc_s=t3 - t2, total_s=time.perf_counter() - t0)
        return ans

    @staticmethod
    def _doc_hint(question: str, docs: dict) -> list[str]:
        """Documents the user names explicitly, via a distinctive token of the file name
        (e.g. 'Beta Ltd' -> beta_ltd_report.pdf). Returns [] when nothing/everything matches."""
        qtok = {t.rstrip("s") for t in re.findall(r"[a-z]+", question.lower())}
        out = []
        for doc_id, d in docs.items():
            toks = {t for t in re.findall(r"[a-z]+", Path(d["filename"]).stem.lower()) if len(t) >= 3 and t not in _GENERIC_NAME_TOKENS}
            if toks and any(t.rstrip("s") in qtok for t in toks):
                out.append(doc_id)
        return out

    def _candidate_tables(self, hits: list[Hit], docs: dict, table_intent: bool) -> list[TableRecord]:
        wanted = [h.metadata.get("table_id") for h in hits if h.metadata.get("table_id")]
        all_tables = {t["table_id"]: t for d in docs.values() for t in d.get("tables", [])}
        ordered = [all_tables[w] for w in dict.fromkeys(wanted) if w in all_tables]
        if table_intent and not ordered:
            ordered = list(all_tables.values())
        return [TableRecord.from_dict(t) for t in ordered[:6]]

    def _images_for_vision(self, hits: list[Hit], plan) -> list[str]:
        out: list[str] = []
        for h in hits:
            m = h.metadata
            cand = m.get("image_path") if m.get("content_type") == "image" else (m.get("page_image") if (plan.is_visual or plan.pages) else "")
            if cand and cand not in out and Path(cand).exists():
                out.append(cand)
        return out[: self.s.max_vision_images]
