"""Central configuration. All values can be overridden with environment variables
(or a local .env file). Secrets are only ever read from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # ---- storage -------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "./data")))

    # ---- upload limits / validation -----------------------------------
    max_file_mb: int = field(default_factory=lambda: _env_int("MAX_FILE_MB", 50))
    max_pdf_pages: int = field(default_factory=lambda: _env_int("MAX_PDF_PAGES", 300))
    max_image_pixels: int = field(default_factory=lambda: _env_int("MAX_IMAGE_PIXELS", 60_000_000))
    max_docx_uncompressed_mb: int = field(default_factory=lambda: _env_int("MAX_DOCX_UNCOMPRESSED_MB", 300))
    allowed_extensions: tuple = (".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg")

    # ---- extraction ---------------------------------------------------
    ocr_enabled: bool = field(default_factory=lambda: _env_bool("OCR_ENABLED", True))
    ocr_lang: str = field(default_factory=lambda: _env("OCR_LANG", "eng"))
    tesseract_cmd: str = field(default_factory=lambda: _env("TESSERACT_CMD", ""))  # e.g. C:\Program Files\Tesseract-OCR\tesseract.exe
    ocr_dpi: int = field(default_factory=lambda: _env_int("OCR_DPI", 300))
    min_text_chars_per_page: int = field(default_factory=lambda: _env_int("MIN_TEXT_CHARS_PER_PAGE", 40))
    table_backend: str = field(default_factory=lambda: _env("TABLE_BACKEND", "auto"))  # auto|pymupdf|pdfplumber
    render_pages: bool = field(default_factory=lambda: _env_bool("RENDER_PAGES", True))
    preview_dpi: int = field(default_factory=lambda: _env_int("PREVIEW_DPI", 110))
    min_embedded_image_px: int = field(default_factory=lambda: _env_int("MIN_EMBEDDED_IMAGE_PX", 120))
    caption_images_at_ingest: bool = field(default_factory=lambda: _env_bool("CAPTION_IMAGES_AT_INGEST", False))

    # ---- chunking -----------------------------------------------------
    chunk_chars: int = field(default_factory=lambda: _env_int("CHUNK_CHARS", 900))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 150))
    table_rows_per_chunk: int = field(default_factory=lambda: _env_int("TABLE_ROWS_PER_CHUNK", 25))

    # ---- embeddings / vector store ------------------------------------
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "sentence-transformers"))  # sentence-transformers|hashing
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    vector_store: str = field(default_factory=lambda: _env("VECTOR_STORE", "chroma"))

    # ---- retrieval ----------------------------------------------------
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 6))
    candidate_k: int = field(default_factory=lambda: _env_int("CANDIDATE_K", 24))
    reranker: str = field(default_factory=lambda: _env("RERANKER", "lexical"))  # none|lexical|cross-encoder
    cross_encoder_model: str = field(default_factory=lambda: _env("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    min_vector_similarity: float = field(default_factory=lambda: _env_float("MIN_VECTOR_SIMILARITY", 0.50))
    min_lexical_overlap: float = field(default_factory=lambda: _env_float("MIN_LEXICAL_OVERLAP", 0.34))

    # ---- LLM ----------------------------------------------------------
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "ollama"))  # ollama|huggingface|openai|extractive
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "phi3:mini"))
    vision_provider: str = field(default_factory=lambda: _env("VISION_PROVIDER", "auto"))  # auto|none|ollama|openai
    vision_model: str = field(default_factory=lambda: _env("VISION_MODEL", "llava:7b"))
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    hf_model: str = field(default_factory=lambda: _env("HF_MODEL", "microsoft/Phi-3-mini-4k-instruct"))
    num_ctx: int = field(default_factory=lambda: _env_int("NUM_CTX", 4096))
    max_new_tokens: int = field(default_factory=lambda: _env_int("MAX_NEW_TOKENS", 600))
    temperature: float = field(default_factory=lambda: _env_float("TEMPERATURE", 0.1))
    max_context_chars: int = field(default_factory=lambda: _env_int("MAX_CONTEXT_CHARS", 7000))
    max_history_turns: int = field(default_factory=lambda: _env_int("MAX_HISTORY_TURNS", 4))
    max_vision_images: int = field(default_factory=lambda: _env_int("MAX_VISION_IMAGES", 3))
    request_timeout: int = field(default_factory=lambda: _env_int("REQUEST_TIMEOUT", 180))

    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # ---- derived paths --------------------------------------------------
    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def pages_dir(self) -> Path:
        return self.data_dir / "pages"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def registry_dir(self) -> Path:
        return self.data_dir / "registry"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        for d in (self.upload_dir, self.pages_dir, self.images_dir, self.registry_dir, self.chroma_dir):
            d.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
