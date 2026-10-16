"""OCR with Tesseract (pytesseract) and OpenCV pre-processing.

OCR is optional at runtime: if Tesseract is not installed, `OCREngine.available`
is False and callers record a warning instead of silently returning nothing.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass

import numpy as np
from PIL import Image

from config import Settings

log = logging.getLogger(__name__)


@dataclass
class OCRResult:
    text: str
    confidence: float  # mean word confidence 0-100, -1 if unknown


class OCREngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.available = False
        self.reason = ""
        if not settings.ocr_enabled:
            self.reason = "OCR disabled by configuration (OCR_ENABLED=false)."
            return
        try:
            import pytesseract

            if settings.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
            elif not shutil.which("tesseract"):
                raise RuntimeError("tesseract executable not found on PATH")
            pytesseract.get_tesseract_version()
            self._pt = pytesseract
            self.available = True
        except Exception as exc:  # ImportError, TesseractNotFoundError, RuntimeError
            self.reason = (
                f"OCR unavailable ({exc}). Install Tesseract and set TESSERACT_CMD if it is not on PATH."
            )
            log.warning(self.reason)

    # ------------------------------------------------------------------
    @staticmethod
    def preprocess(img: Image.Image) -> np.ndarray:
        """Grayscale -> upscale small images -> denoise -> adaptive threshold -> light deskew."""
        import cv2

        arr = np.array(img.convert("L"))
        h, w = arr.shape
        if max(h, w) < 1200:  # small screenshots OCR much better when upscaled
            scale = 1800 / max(h, w)
            arr = cv2.resize(arr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        arr = cv2.fastNlMeansDenoising(arr, None, h=10, templateWindowSize=7, searchWindowSize=21)
        arr = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
        return OCREngine._deskew(arr)

    @staticmethod
    def _deskew(binary: np.ndarray) -> np.ndarray:
        import cv2

        coords = np.column_stack(np.where(binary < 128))
        if len(coords) < 500:
            return binary
        angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) < 0.5 or abs(angle) > 15:  # ignore noise and implausible skews
            return binary
        h, w = binary.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(binary, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)

    def _ocr(self, arr_or_img, psm: int) -> OCRResult:
        pt = self._pt
        img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
        data = pt.image_to_data(img, lang=self.settings.ocr_lang, config=f"--psm {psm}", output_type=pt.Output.DICT)
        lines: dict[tuple, list[str]] = {}
        confs: list[float] = []
        for i, word in enumerate(data["text"]):
            word = (word or "").strip()
            if not word:
                continue
            conf = float(data["conf"][i])
            if conf >= 0:
                confs.append(conf)
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(word)
        text = "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))
        return OCRResult(text.strip(), sum(confs) / len(confs) if confs else -1.0)

    def run(self, img: Image.Image) -> OCRResult:
        """OCR the image both as-is (grayscale) and pre-processed, and keep the more trustworthy result.
        (Aggressive thresholding helps noisy scans but can hurt clean screenshots/diagrams.)"""
        if not self.available:
            return OCRResult("", -1.0)
        candidates: list[OCRResult] = []
        for make, psm in ((lambda: img.convert("L"), 6), (lambda: self.preprocess(img), 6), (lambda: img.convert("L"), 11)):
            try:
                candidates.append(self._ocr(make(), psm))
            except Exception as exc:
                log.warning("OCR attempt failed: %s", exc)
        candidates = [c for c in candidates if c.text]
        if not candidates:
            return OCRResult("", -1.0)
        # score = mean confidence, lightly rewarding more recognised words
        return max(candidates, key=lambda c: c.confidence * (1 + 0.02 * min(len(c.text.split()), 50)))
