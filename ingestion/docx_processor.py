"""DOCX extraction with python-docx. Body order (paragraph/table/paragraph...) is preserved.
Word has no fixed pages, so `page` is None and citations use the section heading instead."""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Callable, Optional

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from PIL import Image

from config import Settings
from .models import Element, ExtractionResult, ImageRecord
from .ocr_processor import OCREngine
from .table_extractor import build_record

log = logging.getLogger(__name__)


def _table_grid(tbl: Table) -> list[list[str]]:
    grid = []
    for row in tbl.rows:
        seen, cells = set(), []
        for cell in row.cells:
            if id(cell._tc) in seen:  # merged cell repeated by python-docx
                cells.append("")
                continue
            seen.add(id(cell._tc))
            cells.append(cell.text)
        grid.append(cells)
    return grid


def process_docx(path: Path, *, doc_id: str, filename: str, settings: Settings, ocr: OCREngine,
                 captioner: Optional[Callable[[str], str]] = None) -> ExtractionResult:
    res = ExtractionResult(doc_id=doc_id, filename=filename, file_type="docx")
    try:
        doc = Document(str(path))
    except Exception as exc:
        raise ValueError(f"Could not open Word document '{filename}': {exc}") from exc

    heading, buf, t_idx = "", [], 0

    def flush() -> None:
        if buf:
            res.elements.append(Element("text", filename, None, text="\n\n".join(buf), heading=heading))
            buf.clear()

    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, doc)
            text = p.text.strip()
            if not text:
                continue
            style = (p.style.name if p.style is not None else "") or ""
            if style.lower().startswith(("heading", "title")):
                flush()
                heading = text
                buf.append(f"## {text}")
            elif "list" in style.lower():
                buf.append(f"- {text}")
            else:
                buf.append(text)
        elif tag == "tbl":
            flush()
            t_idx += 1
            rec = build_record(_table_grid(Table(child, doc)), table_id=f"{doc_id}-t{t_idx}",
                               filename=filename, page=None, heading=heading, extractor="python-docx")
            if rec:
                res.tables.append(rec)
                res.elements.append(Element("table", filename, None, heading=heading, table=rec))
    flush()

    # ---- embedded images ---------------------------------------------------
    img_dir = settings.images_dir / doc_id
    n = 0
    for rel in doc.part.rels.values():
        if "image" not in rel.reltype:
            continue
        try:
            blob = rel.target_part.blob
            img = Image.open(io.BytesIO(blob))
            if img.width < settings.min_embedded_image_px or img.height < settings.min_embedded_image_px:
                continue
            n += 1
            img_dir.mkdir(parents=True, exist_ok=True)
            out = img_dir / f"img{n}.png"
            img.convert("RGB").save(out)
        except Exception as exc:
            log.warning("Skipping unreadable DOCX image: %s", exc)
            continue
        rec = ImageRecord(image_id=f"{doc_id}-i{n}", filename=filename, page=None, path=str(out),
                          width=img.width, height=img.height)
        if ocr.available:
            rec.ocr_text = ocr.run(Image.open(out)).text
        if captioner and settings.caption_images_at_ingest:
            try:
                rec.caption = captioner(str(out))
            except Exception as exc:
                log.warning("Captioning failed: %s", exc)
        res.images.append(rec)
        res.elements.append(Element("image", filename, None, image=rec, text=(rec.caption + "\n" + rec.ocr_text).strip()))
    if not ocr.available and n:
        res.warnings.append(f"{n} embedded image(s) were stored but not OCR'd: {ocr.reason}")
    return res
