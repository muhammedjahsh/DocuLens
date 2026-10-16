"""Standalone image files (PNG/JPG/JPEG): normalise, OCR, optional vision caption."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from PIL import Image, ImageOps

from config import Settings
from .models import Element, ExtractionResult, ImageRecord
from .ocr_processor import OCREngine

log = logging.getLogger(__name__)


def process_image(path: Path, *, doc_id: str, filename: str, settings: Settings, ocr: OCREngine,
                  captioner: Optional[Callable[[str], str]] = None) -> ExtractionResult:
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels  # decompression-bomb guard
    res = ExtractionResult(doc_id=doc_id, filename=filename, file_type="image", n_pages=1)
    try:
        img = Image.open(path)
        img.load()
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not read image '{filename}': {exc}") from exc

    out_dir = settings.images_dir / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "image.png"
    img.save(out)
    rec = ImageRecord(image_id=f"{doc_id}-i1", filename=filename, page=1, path=str(out),
                      width=img.width, height=img.height)
    if ocr.available:
        r = ocr.run(img)
        rec.ocr_text = r.text
        if 0 <= r.confidence < 60:
            res.warnings.append(f"Low OCR confidence ({r.confidence:.0f}/100); verify extracted text.")
        if r.text:
            res.ocr_pages.append(1)
    else:
        res.warnings.append(ocr.reason)
    if captioner and settings.caption_images_at_ingest:
        try:
            rec.caption = captioner(str(out))
        except Exception as exc:
            log.warning("Captioning failed: %s", exc)
    res.images.append(rec)
    res.page_images[1] = str(out)
    res.elements.append(Element("image", filename, 1, image=rec, text=(rec.caption + "\n" + rec.ocr_text).strip()))
    return res
