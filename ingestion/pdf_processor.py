"""PDF extraction with PyMuPDF (+ pdfplumber fallback for tables).

Per page:
  1. text blocks (excluding regions that are tables) with heading detection by font size
  2. tables -> TableRecord
  3. embedded images (size-filtered) saved to disk, OCR'd when possible
  4. if the page has too little text -> render page and OCR it (scanned page)
  5. optionally render every page to PNG for preview and for vision-model queries
"""
from __future__ import annotations

import logging
import statistics
from pathlib import Path
from typing import Callable, Optional

import pymupdf
from PIL import Image

from config import Settings
from .models import Element, ExtractionResult, ImageRecord
from .ocr_processor import OCREngine
from .table_extractor import bbox_overlap_ratio, extract_pdf_tables

log = logging.getLogger(__name__)
Captioner = Callable[[str], str]


def _body_font_size(doc: "pymupdf.Document") -> float:
    sizes: list[float] = []
    for pno in range(min(len(doc), 15)):
        for b in doc[pno].get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for sp in line["spans"]:
                    if sp["text"].strip():
                        sizes.extend([sp["size"]] * len(sp["text"]))
    return statistics.median(sizes) if sizes else 11.0


def process_pdf(path: Path, *, doc_id: str, filename: str, settings: Settings, ocr: OCREngine,
                captioner: Optional[Captioner] = None) -> ExtractionResult:
    res = ExtractionResult(doc_id=doc_id, filename=filename, file_type="pdf")
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        raise ValueError(f"Could not open PDF '{filename}': {exc}") from exc
    if doc.needs_pass:
        raise ValueError(f"'{filename}' is password-protected.")
    if len(doc) == 0:
        raise ValueError(f"'{filename}' has no pages.")
    n_pages = len(doc)
    if n_pages > settings.max_pdf_pages:
        res.warnings.append(f"Only the first {settings.max_pdf_pages} of {n_pages} pages were processed.")
        n_pages = settings.max_pdf_pages
    res.n_pages = n_pages

    plumber = None
    if settings.table_backend in ("auto", "pdfplumber"):
        try:
            import pdfplumber

            plumber = pdfplumber.open(path)
        except Exception as exc:
            log.warning("pdfplumber unavailable: %s", exc)

    body = _body_font_size(doc)
    page_dir = settings.pages_dir / doc_id
    img_dir = settings.images_dir / doc_id
    current_heading = ""
    ocr_warned = False

    try:
        for pno in range(n_pages):
            page = doc[pno]
            page_no = pno + 1

            # ---- tables ------------------------------------------------
            tables, tbboxes = extract_pdf_tables(
                page, plumber.pages[pno] if plumber else None, doc_id=doc_id, filename=filename,
                page_no=page_no, backend=settings.table_backend, heading=current_heading)
            # ---- prose with heading detection -------------------------------
            paragraphs: list[str] = []
            for b in page.get_text("dict")["blocks"]:
                if b.get("type") != 0:
                    continue
                if any(bbox_overlap_ratio(b["bbox"], tb) > 0.6 for tb in tbboxes):
                    continue  # this text belongs to a table
                lines, max_size, bold = [], 0.0, False
                for line in b["lines"]:
                    txt = "".join(sp["text"] for sp in line["spans"]).strip()
                    if txt:
                        lines.append(txt)
                        for sp in line["spans"]:
                            if sp["text"].strip():
                                max_size = max(max_size, sp["size"])
                                bold = bold or bool(sp["flags"] & 16)
                text = " ".join(lines).strip()
                if not text:
                    continue
                is_heading = len(text) < 120 and (max_size >= body * 1.25 or (bold and len(lines) == 1 and max_size >= body))
                if is_heading and not text.endswith((".", ",")):
                    current_heading = text
                    paragraphs.append(f"## {text}")
                else:
                    paragraphs.append(text)
            page_text = "\n\n".join(paragraphs).strip()
            plain_len = len(page_text.replace("## ", ""))

            for t in tables:  # a table takes the heading that precedes/sits on its page
                t.heading = current_heading
                res.tables.append(t)
                res.elements.append(Element("table", filename, page_no, heading=current_heading, table=t))

            # ---- scanned page -> OCR the whole rendered page -------------
            if plain_len < settings.min_text_chars_per_page and not tables:
                if ocr.available:
                    pix = page.get_pixmap(dpi=settings.ocr_dpi)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    r = ocr.run(img)
                    if r.text:
                        res.elements.append(Element("ocr_text", filename, page_no, text=r.text, heading=current_heading))
                        res.ocr_pages.append(page_no)
                        if 0 <= r.confidence < 60:
                            res.warnings.append(f"Page {page_no}: low OCR confidence ({r.confidence:.0f}/100); verify figures.")
                    elif not page.get_images():
                        res.warnings.append(f"Page {page_no}: no text found (blank page?).")
                elif not ocr_warned:
                    res.warnings.append(f"Some pages have no text layer and OCR is unavailable: {ocr.reason}")
                    ocr_warned = True
            elif page_text:
                res.elements.append(Element("text", filename, page_no, text=page_text, heading=current_heading))

            # ---- embedded images -----------------------------------------
            for i_idx, info in enumerate(page.get_images(full=True)):
                xref = info[0]
                try:
                    pix = pymupdf.Pixmap(doc, xref)
                    if pix.width < settings.min_embedded_image_px or pix.height < settings.min_embedded_image_px:
                        continue
                    if pix.n - pix.alpha >= 4:  # CMYK -> RGB
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    if pix.alpha:
                        pix = pymupdf.Pixmap(pix, 0)
                    # a full-page scan image was already OCR'd above; skip duplicating it as a "figure"
                    if res.ocr_pages and res.ocr_pages[-1] == page_no and pix.width * pix.height > 0.6 * page.rect.width * page.rect.height:
                        continue
                    img_dir.mkdir(parents=True, exist_ok=True)
                    out = img_dir / f"p{page_no}_img{i_idx + 1}.png"
                    pix.save(out)
                except Exception as exc:
                    log.warning("Skipping image xref %s on page %s: %s", xref, page_no, exc)
                    continue
                rec = ImageRecord(image_id=f"{doc_id}-p{page_no}-i{i_idx + 1}", filename=filename, page=page_no,
                                  path=str(out), width=pix.width, height=pix.height)
                if ocr.available:
                    rec.ocr_text = ocr.run(Image.open(out)).text
                if captioner and settings.caption_images_at_ingest:
                    try:
                        rec.caption = captioner(str(out))
                    except Exception as exc:
                        log.warning("Captioning failed for %s: %s", rec.image_id, exc)
                res.images.append(rec)
                res.elements.append(Element("image", filename, page_no, heading=current_heading, image=rec,
                                            text=(rec.caption + "\n" + rec.ocr_text).strip()))

            # ---- page render for preview / vision ---------------------------
            if settings.render_pages:
                page_dir.mkdir(parents=True, exist_ok=True)
                out = page_dir / f"page_{page_no}.png"
                page.get_pixmap(dpi=settings.preview_dpi).save(out)
                res.page_images[page_no] = str(out)
    finally:
        doc.close()
        if plumber:
            plumber.close()
    return res
