"""Turns extracted Elements into Chunks with citation-ready metadata.

* Prose is split on paragraph -> sentence boundaries with overlap; chunks never cross a page.
* Tables become Markdown chunks (header repeated every TABLE_ROWS_PER_CHUNK rows) plus the
  full structured table lives in the registry for deterministic calculation.
* Images are indexed by caption + OCR text; the original image path stays in metadata.
"""
from __future__ import annotations

import re
from typing import Iterable

from config import Settings
from ingestion.models import Chunk, Element, ExtractionResult

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_text(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    units: list[str] = []
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            units.append(para)
            continue
        for sent in _SENT.split(para):
            sent = sent.strip()
            while len(sent) > size:  # pathological: no punctuation
                units.append(sent[:size])
                sent = sent[size:]
            if sent:
                units.append(sent)
    chunks, cur = [], ""
    for u in units:
        if cur and len(cur) + len(u) + 2 > size:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail  # start at a word boundary
            cur = (tail + "\n\n" + u) if tail else u
        else:
            cur = f"{cur}\n\n{u}" if cur else u
    if cur:
        chunks.append(cur)
    return chunks


def _meta(el: Element, doc_id: str, ctype: str, extra: dict | None = None) -> dict:
    m = {
        "doc_id": doc_id,
        "filename": el.filename,
        "page": el.page if el.page is not None else -1,  # chroma metadata cannot hold None
        "content_type": ctype,
        "heading": el.heading or "",
        "table_id": "",
        "image_path": "",
    }
    m.update(extra or {})
    return m


def build_chunks(result: ExtractionResult, settings: Settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    counters: dict[tuple, int] = {}

    def nid(kind: str, page) -> str:
        k = (kind, page)
        counters[k] = counters.get(k, 0) + 1
        return f"{result.doc_id}:{kind}:{page if page is not None else 'na'}:{counters[k]}"

    for el in result.elements:
        prefix = f"[{el.filename}" + (f" | page {el.page}" if el.page is not None else "") + \
                 (f" | section: {el.heading}" if el.heading else "") + "] "
        if el.kind in ("text", "ocr_text"):
            ctype = "ocr_text" if el.kind == "ocr_text" else "text"
            for piece in split_text(el.text, settings.chunk_chars, settings.chunk_overlap):
                chunks.append(Chunk(nid(ctype, el.page), piece, prefix + piece, _meta(el, result.doc_id, ctype)))
        elif el.kind == "table" and el.table:
            t = el.table
            step = settings.table_rows_per_chunk
            for start in range(0, max(len(t.rows), 1), step):
                part = t.rows[start:start + step]
                md = t.to_markdown(part)
                label = f"Table {t.table_id} (rows {start + 1}-{start + len(part)} of {len(t.rows)}; columns: {', '.join(t.headers)})\n"
                chunks.append(Chunk(nid("table", el.page), label + md, prefix + label + md,
                                    _meta(el, result.doc_id, "table", {"table_id": t.table_id})))
        elif el.kind == "image" and el.image:
            im = el.image
            desc = []
            if im.caption:
                desc.append(f"Visual description (model-generated): {im.caption}")
            if im.ocr_text:
                desc.append(f"Text found in image (OCR): {im.ocr_text}")
            if not desc:
                desc.append("Image/figure with no extractable text; inspect the image itself.")
            body = "\n".join(desc)
            chunks.append(Chunk(nid("image", el.page), body, prefix + "Image/figure. " + body,
                                _meta(el, result.doc_id, "image", {"image_path": im.path})))
    # attach rendered page path so vision queries can pull the page
    for c in chunks:
        p = c.metadata["page"]
        c.metadata["page_image"] = result.page_images.get(p, "") if p != -1 else ""
    return chunks
