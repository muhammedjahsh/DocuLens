"""Data structures shared by ingestion, indexing, retrieval and generation."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import pandas as pd


@dataclass
class TableRecord:
    table_id: str
    filename: str
    page: Optional[int]
    headers: list[str]
    rows: list[list[str]]
    heading: str = ""
    extractor: str = ""

    def to_dataframe(self) -> pd.DataFrame:
        width = len(self.headers)
        rows = [(r + [""] * width)[:width] for r in self.rows]
        return pd.DataFrame(rows, columns=self.headers)

    def to_markdown(self, rows: Optional[list[list[str]]] = None) -> str:
        esc = lambda v: str(v).replace("|", "\\|").replace("\n", " ")
        rows = self.rows if rows is None else rows
        out = ["| " + " | ".join(esc(h) for h in self.headers) + " |",
               "| " + " | ".join("---" for _ in self.headers) + " |"]
        for r in rows:
            r = (list(r) + [""] * len(self.headers))[: len(self.headers)]
            out.append("| " + " | ".join(esc(c) for c in r) + " |")
        return "\n".join(out)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TableRecord":
        return cls(**d)


@dataclass
class ImageRecord:
    image_id: str
    filename: str
    page: Optional[int]
    path: str
    ocr_text: str = ""
    caption: str = ""
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ImageRecord":
        return cls(**d)


@dataclass
class Element:
    """One extracted unit of content, before chunking.
    kind: text | ocr_text | table | image
    """
    kind: str
    filename: str
    page: Optional[int]
    text: str = ""
    heading: str = ""
    table: Optional[TableRecord] = None
    image: Optional[ImageRecord] = None


@dataclass
class ExtractionResult:
    doc_id: str
    filename: str
    file_type: str
    elements: list[Element] = field(default_factory=list)
    tables: list[TableRecord] = field(default_factory=list)
    images: list[ImageRecord] = field(default_factory=list)
    page_images: dict[int, str] = field(default_factory=dict)  # page -> rendered PNG path
    n_pages: Optional[int] = None
    ocr_pages: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    id: str
    text: str            # what is shown to the LLM / user
    embed_text: str      # what is embedded (text + section prefix)
    metadata: dict[str, Any]
