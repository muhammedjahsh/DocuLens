import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Settings  # noqa: E402
from ingestion.ocr_processor import OCREngine  # noqa: E402
from rag.embeddings import HashingEmbedder  # noqa: E402
from rag.llm import BaseLLM, ExtractiveLLM, LLMError  # noqa: E402
from rag.pipeline import RAGService  # noqa: E402
from scripts.generate_samples import generate_all  # noqa: E402


class FakeLLM(BaseLLM):
    """Scripted LLM used to test prompting, citation validation and fallbacks without a model."""
    name = "fake"

    def __init__(self, reply="", fail=False):
        self.reply, self.fail, self.calls = reply, fail, []

    def chat(self, messages, images=None):
        self.calls.append((messages, images))
        if self.fail:
            raise LLMError("fake outage")
        return self.reply


def make_settings(tmp: Path) -> Settings:
    s = Settings()
    s.data_dir = tmp
    s.embedding_backend = "hashing"
    s.llm_provider = "extractive"
    s.min_vector_similarity = 0.35
    s.ensure_dirs()
    return s


@pytest.fixture(scope="session")
def samples(tmp_path_factory) -> dict:
    return generate_all(tmp_path_factory.mktemp("samples"))


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return make_settings(tmp_path / "data")


@pytest.fixture(scope="session")
def ocr() -> OCREngine:
    return OCREngine(make_settings(Path("/tmp/_ocr_probe")))


@pytest.fixture(scope="session")
def service(tmp_path_factory, samples) -> RAGService:
    svc = RAGService(make_settings(tmp_path_factory.mktemp("svc")), embedder=HashingEmbedder(), llm=ExtractiveLLM())
    for key in ("acme", "beta", "docx", "scanned", "invoice", "diagram", "txt"):
        p = samples[key]
        svc.ingest_bytes(p.name, p.read_bytes())
    return svc


@pytest.fixture(scope="session")
def doc_ids(service) -> dict:
    return {d["filename"]: d["doc_id"] for d in service.list_documents()}
