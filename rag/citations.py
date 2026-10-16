"""Source labels, citation validation and unsupported-number detection.

The LLM is told to cite as [S1], [S2]... Anything it cites that was NOT in the supplied
context is stripped (so invented citations never reach the user)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .vector_store import Hit

_GROUP = re.compile(r"\[([^\[\]]{1,40})\]")
_LABELS = re.compile(r"^\s*S\d+(\s*[,;]\s*S?\d+)*\s*$", re.I)


@dataclass
class SourceRef:
    label: str
    filename: str
    page: Optional[int]
    content_type: str
    heading: str
    excerpt: str
    table_id: str = ""
    image_path: str = ""
    page_image: str = ""
    cited: bool = False

    @property
    def location(self) -> str:
        return format_location(self.filename, self.page, self.heading)


def format_location(filename: str, page: Optional[int], heading: str = "") -> str:
    if page is not None and page > 0:
        return f"{filename}, p. {page}"
    return f"{filename}, section “{heading}”" if heading else filename


def build_sources(hits: list[Hit], excerpt_chars: int = 450) -> list[SourceRef]:
    out = []
    for i, h in enumerate(hits, 1):
        m = h.metadata
        page = m.get("page", -1)
        out.append(SourceRef(
            label=f"S{i}", filename=m.get("filename", "?"), page=page if page and page > 0 else None,
            content_type=m.get("content_type", "text"), heading=m.get("heading", ""),
            excerpt=(h.text[:excerpt_chars] + ("…" if len(h.text) > excerpt_chars else "")),
            table_id=m.get("table_id", ""), image_path=m.get("image_path", ""), page_image=m.get("page_image", "")))
    return out


def _group_labels(inner: str) -> list[str]:
    return ["S" + n for n in re.findall(r"(\d+)", inner)]


def validate_citations(text: str, valid_labels: set[str]) -> tuple[str, list[str], list[str]]:
    """Returns (clean_text, cited_valid_in_order, removed_invalid)."""
    cited: list[str] = []
    invalid: list[str] = []

    def repl(m: re.Match) -> str:
        inner = m.group(1)
        if not _LABELS.match(inner):
            return m.group(0)
        keep = []
        for lab in _group_labels(inner):
            if lab in valid_labels:
                keep.append(lab)
                if lab not in cited:
                    cited.append(lab)
            else:
                invalid.append(lab)
        return f"[{', '.join(keep)}]" if keep else ""

    clean = _GROUP.sub(repl, text)
    return re.sub(r"[ \t]+([.,;:])", r"\1", re.sub(r"  +", " ", clean)).strip(), cited, invalid


_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def _norm_nums(text: str) -> set[str]:
    out = set()
    for m in _NUMBER.findall(text):
        n = m.replace(",", "").lstrip("-").rstrip(".")
        out.add(n)
        if "." in n:
            out.add(n.rstrip("0").rstrip("."))
    return out


def find_unsupported_numbers(answer: str, evidence: str) -> list[str]:
    """Numbers in the answer that appear nowhere in the evidence (candidate hallucinations).
    Citation labels, list numbering and small integers (<= 10) are ignored."""
    body = re.sub(r"\[S\d+(?:\s*[,;]\s*S?\d+)*\]", "", answer)
    body = re.sub(r"(?m)^\s*\d+[.)]\s", "", body)
    have = _norm_nums(evidence)
    bad = []
    for m in _NUMBER.findall(body):
        n = m.replace(",", "").lstrip("-").rstrip(".")
        try:
            small = float(n) <= 10 and "." not in n
        except ValueError:
            continue
        if small:
            continue
        if n not in have and n.rstrip("0").rstrip(".") not in have and m not in bad:
            bad.append(m)
    return bad
