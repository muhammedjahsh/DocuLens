"""Smoke-test the Streamlit UI with Streamlit's AppTest (no browser, no model)."""
import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from rag.embeddings import HashingEmbedder  # noqa: E402
from rag.llm import ExtractiveLLM  # noqa: E402
from rag.pipeline import RAGService  # noqa: E402
from tests.conftest import ROOT, make_settings  # noqa: E402
import streamlit as st  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_service_cache():
    st.cache_resource.clear()  # app.get_service() is cached per process; isolate tests
    yield
    st.cache_resource.clear()


def test_app_renders_and_answers(tmp_path, samples, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("LLM_PROVIDER", "extractive")
    svc = RAGService(make_settings(tmp_path / "data"), embedder=HashingEmbedder(), llm=ExtractiveLLM())
    for k in ("acme", "txt"):
        svc.ingest_bytes(samples[k].name, samples[k].read_bytes())

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("acme_report_2025.pdf" in c.label for c in at.checkbox)

    at.chat_input[0].set_value("What is the total revenue shown in the table?").run()
    assert not at.exception
    answers = [m for m in at.chat_message if m.name == "assistant"]
    assert answers and "5,250,000" in answers[-1].markdown[0].value


def test_app_without_documents_shows_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("LLM_PROVIDER", "extractive")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception and any("Upload" in i.value for i in at.info)
