"""Dispatches an uploaded file to the right extractor."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Optional

from config import Settings
from .docx_processor import process_docx
from .image_processor import process_image
from .models import Element, ExtractionResult
from .ocr_processor import OCREngine
from .pdf_processor import process_pdf


def compute_doc_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def process_txt(path: Path, *, doc_id: str, filename: str) -> ExtractionResult:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeError:
            continue
    text = text.replace("\r\n", "\n").strip()
    if not text:
        raise ValueError(f"'{filename}' contains no text.")
    res = ExtractionResult(doc_id=doc_id, filename=filename, file_type="txt")
    res.elements.append(Element("text", filename, None, text=text))
    return res


def extract_document(path: Path, *, doc_id: str, filename: str, settings: Settings, ocr: OCREngine,
                     captioner: Optional[Callable[[str], str]] = None) -> ExtractionResult:
    ext = Path(filename).suffix.lower()
    kw = dict(doc_id=doc_id, filename=filename)
    if ext == ".pdf":
        res = process_pdf(path, settings=settings, ocr=ocr, captioner=captioner, **kw)
    elif ext == ".docx":
        res = process_docx(path, settings=settings, ocr=ocr, captioner=captioner, **kw)
    elif ext in (".png", ".jpg", ".jpeg"):
        res = process_image(path, settings=settings, ocr=ocr, captioner=captioner, **kw)
    elif ext == ".txt":
        res = process_txt(path, **kw)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    if not any(e.text.strip() or e.table for e in res.elements) and not res.images:
        raise ValueError(f"No readable content could be extracted from '{filename}'.")
    return res
