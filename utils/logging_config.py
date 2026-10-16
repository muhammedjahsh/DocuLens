"""Logging setup with a filter that masks API keys / bearer tokens."""
from __future__ import annotations

import logging
import os
import re

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{12,}"),
]


class SecretMaskingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # pragma: no cover
            return True
        for pat in _SECRET_PATTERNS:
            msg = pat.sub("***REDACTED***", msg)
        key = os.getenv("OPENAI_API_KEY", "")
        if key:
            msg = msg.replace(key, "***REDACTED***")
        record.msg, record.args = msg, ()
        return True


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if getattr(root, "_rag_configured", False):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    handler.addFilter(SecretMaskingFilter())
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root._rag_configured = True  # type: ignore[attr-defined]
