"""Table extraction and normalisation.

* PDF: PyMuPDF `find_tables()` first; pdfplumber as fallback (TABLE_BACKEND=auto).
* DOCX: handled in docx_processor, using `build_record` from here.
Tables are kept as structured TableRecord objects (headers + rows) so numeric
questions can be answered with pandas rather than by the LLM.
"""
from __future__ import annotations

import logging
import re
from typing import Optional, Sequence

from .models import TableRecord

log = logging.getLogger(__name__)


def clean_cell(v: object) -> str:
    return re.sub(r"\s+", " ", str(v if v is not None else "")).strip()


def _dedupe_headers(headers: Sequence[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i, h in enumerate(headers):
        h = h or f"col_{i + 1}"
        if h in seen:
            seen[h] += 1
            h = f"{h}_{seen[h]}"
        else:
            seen[h] = 1
        out.append(h)
    return out


def build_record(grid: Sequence[Sequence[object]], *, table_id: str, filename: str,
                 page: Optional[int], heading: str = "", extractor: str = "") -> Optional[TableRecord]:
    """Normalise a raw cell grid (first row = header) into a TableRecord."""
    rows = [[clean_cell(c) for c in r] for r in grid]
    rows = [r for r in rows if any(r)]  # drop empty rows
    if len(rows) < 2:
        return None
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep = [j for j in range(width) if any(r[j] for r in rows)]  # drop empty columns
    rows = [[r[j] for j in keep] for r in rows]
    if len(rows[0]) < 2:  # a "table" with a single column is almost always layout noise
        return None
    headers = _dedupe_headers(rows[0])
    return TableRecord(table_id=table_id, filename=filename, page=page, headers=headers,
                       rows=rows[1:], heading=heading, extractor=extractor)


def extract_pdf_tables(page, plumber_page, *, doc_id: str, filename: str, page_no: int,
                       backend: str, heading: str = "") -> tuple[list[TableRecord], list[tuple]]:
    """Returns (tables, bboxes). bboxes are used to avoid duplicating table text as prose."""
    tables: list[TableRecord] = []
    bboxes: list[tuple] = []
    if backend in ("auto", "pymupdf"):
        try:
            found = page.find_tables()
            for t_idx, t in enumerate(found.tables):
                rec = build_record(t.extract(), table_id=f"{doc_id}-p{page_no}-t{t_idx + 1}",
                                   filename=filename, page=page_no, heading=heading, extractor="pymupdf")
                if rec:
                    tables.append(rec)
                    bboxes.append(tuple(t.bbox))
        except Exception as exc:
            log.warning("PyMuPDF table extraction failed on p%s: %s", page_no, exc)
    if not tables and backend in ("auto", "pdfplumber") and plumber_page is not None:
        try:
            for t_idx, t in enumerate(plumber_page.find_tables()):
                rec = build_record(t.extract(), table_id=f"{doc_id}-p{page_no}-t{t_idx + 1}",
                                   filename=filename, page=page_no, heading=heading, extractor="pdfplumber")
                if rec:
                    tables.append(rec)
                    bboxes.append(tuple(t.bbox))
        except Exception as exc:
            log.warning("pdfplumber table extraction failed on p%s: %s", page_no, exc)
    return tables, bboxes


def bbox_overlap_ratio(a: Sequence[float], b: Sequence[float]) -> float:
    """Share of box `a` that lies inside box `b`."""
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    return ((x1 - x0) * (y1 - y0)) / area_a
