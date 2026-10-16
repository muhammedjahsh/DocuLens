"""Upload validation: extension allow-list, size limit, magic bytes, zip-bomb check,
and filename sanitisation. Files are never executed or opened by anything except
the parsers in `ingestion/`."""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from config import Settings

_MAGIC = {
    ".pdf": (b"%PDF",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".docx": (b"PK\x03\x04",),
}


class ValidationError(ValueError):
    """Raised with a user-presentable message."""


@dataclass
class ValidatedFile:
    safe_name: str
    extension: str
    size: int


def sanitize_filename(name: str) -> str:
    base = Path(name.replace("\\", "/")).name  # drop any directory components
    base = re.sub(r"[^A-Za-z0-9._\- ()]", "_", base).strip(" .")
    if not base:
        base = "upload"
    return base[:150]


def validate_upload(name: str, data: bytes, settings: Settings) -> ValidatedFile:
    safe = sanitize_filename(name)
    ext = Path(safe).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise ValidationError(
            f"Unsupported file type '{ext or 'none'}' for '{safe}'. "
            f"Allowed: {', '.join(settings.allowed_extensions)}."
        )
    if len(data) == 0:
        raise ValidationError(f"'{safe}' is empty.")
    if len(data) > settings.max_file_bytes:
        raise ValidationError(
            f"'{safe}' is {len(data) / 1_048_576:.1f} MB; the limit is {settings.max_file_mb} MB."
        )
    magics = _MAGIC.get(ext)
    if magics and not any(data.startswith(m) for m in magics):
        raise ValidationError(f"'{safe}' does not look like a valid {ext[1:].upper()} file (content/extension mismatch).")
    if ext == ".docx":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                total = sum(i.file_size for i in zf.infolist())
                if total > settings.max_docx_uncompressed_mb * 1_048_576:
                    raise ValidationError(f"'{safe}' expands to an unreasonable size and was rejected.")
                if "word/document.xml" not in zf.namelist():
                    raise ValidationError(f"'{safe}' is not a valid Word document.")
        except zipfile.BadZipFile as exc:
            raise ValidationError(f"'{safe}' is corrupted (invalid DOCX archive).") from exc
    if ext == ".txt":
        if b"\x00" in data[:4096]:
            raise ValidationError(f"'{safe}' looks binary, not plain text.")
    return ValidatedFile(safe_name=safe, extension=ext, size=len(data))
