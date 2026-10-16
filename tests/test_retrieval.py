import pytest

from ingestion.models import TableRecord
from rag.retriever import analyze_query, content_terms
from rag.table_calc import answer_from_tables, calculate, detect_ops, parse_number


# ---------------------------------------------------------------- query analysis
def test_query_intents():
    assert analyze_query("Summarize this document").is_summary
    assert analyze_query("What is the total revenue in the table?").is_table
    assert analyze_query("Explain the diagram on page 5").is_visual
    assert analyze_query("Explain the diagram on page 5").pages == [5]
    assert analyze_query("Compare the two reports").is_compare
    assert analyze_query("see pages 3-5").pages == [3, 4, 5]
    assert "text" in analyze_query("Who leads the migration?").intents


def test_content_terms_drop_stopwords_and_stem():
    assert content_terms("What are the project objectives?") == ["project", "objective"]


# ---------------------------------------------------------------- retrieval
def top(service, q, ids=None, k=6):
    return service.retriever.retrieve(q, ids, k=k)


def test_text_retrieval_points_to_right_file_and_page(service):
    rr = top(service, "What are the project objectives of Acme Corp?")
    h = rr.hits[0]
    assert h.metadata["filename"] == "acme_report_2025.pdf" and h.metadata["page"] == 1
    assert rr.sufficient


def test_txt_and_docx_retrieval(service):
    assert top(service, "Who will lead the PostgreSQL migration?").hits[0].metadata["filename"] == "notes.txt"
    assert top(service, "Which employee works in Sales?").hits[0].metadata["filename"] == "employees.docx"


def test_scanned_document_is_searchable(service):
    h = top(service, "What budget was approved for the Kerala pilot project?").hits[0]
    assert h.metadata["filename"] == "scanned_memo.pdf" and h.metadata["content_type"] == "ocr_text"


def test_image_ocr_is_searchable(service):
    h = top(service, "What is the total due on the invoice?").hits[0]
    assert h.metadata["filename"] == "invoice.png" and h.metadata["content_type"] == "image"


def test_document_filter_is_respected(service, doc_ids):
    rr = top(service, "What was the revenue in Q2?", [doc_ids["beta_ltd_report.pdf"]])
    assert rr.hits and {h.metadata["filename"] for h in rr.hits} == {"beta_ltd_report.pdf"}


def test_compare_balances_documents(service, doc_ids):
    ids = [doc_ids["acme_report_2025.pdf"], doc_ids["beta_ltd_report.pdf"]]
    rr = top(service, "Compare the project objectives of both reports", ids, k=4)
    assert {h.metadata["filename"] for h in rr.hits} == {"acme_report_2025.pdf", "beta_ltd_report.pdf"}


def test_page_targeting_pulls_requested_page(service, doc_ids):
    rr = top(service, "Explain the chart on page 3", [doc_ids["acme_report_2025.pdf"]])
    assert any(h.metadata["page"] == 3 and h.metadata["content_type"] == "image" for h in rr.hits)
    assert rr.sufficient


def test_irrelevant_question_is_flagged_insufficient(service):
    rr = top(service, "What is the airspeed velocity of an unladen swallow?")
    assert not rr.sufficient


# ---------------------------------------------------------------- end-to-end (extractive LLM)
def test_ask_refuses_when_evidence_missing(service):
    a = service.ask("What is the airspeed velocity of an unladen swallow?")
    assert a.insufficient and "do not contain sufficient information" in a.text


def test_ask_without_documents_is_graceful(settings):
    from rag.embeddings import HashingEmbedder
    from rag.llm import ExtractiveLLM
    from rag.pipeline import RAGService

    svc = RAGService(settings, embedder=HashingEmbedder(), llm=ExtractiveLLM())
    assert "upload" in svc.ask("anything?").text.lower()


def test_ask_table_total_uses_calculation_and_cites_source(service, doc_ids):
    a = service.ask("What is the total revenue shown in the table?", doc_ids=[doc_ids["acme_report_2025.pdf"]])
    assert a.calc and "5,250,000" in a.calc[0].text
    assert "5,250,000" in a.text
    assert any(s.page == 2 and s.filename == "acme_report_2025.pdf" for s in a.sources)


def test_ask_employee_with_highest_salary(service, doc_ids):
    a = service.ask("Find the employee with the highest salary", doc_ids=[doc_ids["employees.docx"]])
    assert "Carol Diaz" in a.text and "135,000" in a.text


def test_summary_covers_selected_document(service, doc_ids):
    a = service.ask("Summarize this document", doc_ids=[doc_ids["notes.txt"]])
    assert a.intents == ["summary"] and "PostgreSQL" in a.text


def test_multi_doc_does_not_mix_up_files(service, doc_ids):
    a = service.ask("What was Beta Ltd's revenue in Q1?", doc_ids=[doc_ids["acme_report_2025.pdf"], doc_ids["beta_ltd_report.pdf"]])
    assert a.sources[0].filename == "beta_ltd_report.pdf"


def test_remove_document_deletes_vectors_and_files(tmp_path_factory, samples):
    from rag.embeddings import HashingEmbedder
    from rag.llm import ExtractiveLLM
    from rag.pipeline import RAGService
    from tests.conftest import make_settings

    s = make_settings(tmp_path_factory.mktemp("rm"))
    svc = RAGService(s, embedder=HashingEmbedder(), llm=ExtractiveLLM())
    info = svc.ingest_bytes("notes.txt", samples["txt"].read_bytes())
    assert svc.store.count() > 0 and svc.ingest_bytes("notes.txt", samples["txt"].read_bytes())["duplicate"]
    svc.remove_document(info["doc_id"])
    assert svc.store.count() == 0 and not svc.list_documents()
    assert not (s.upload_dir / info["doc_id"]).exists()


def test_failed_ingest_leaves_no_residue(tmp_path_factory):
    from rag.embeddings import HashingEmbedder
    from rag.llm import ExtractiveLLM
    from rag.pipeline import RAGService
    from tests.conftest import make_settings

    s = make_settings(tmp_path_factory.mktemp("bad"))
    svc = RAGService(s, embedder=HashingEmbedder(), llm=ExtractiveLLM())
    with pytest.raises(ValueError):
        svc.ingest_bytes("bad.pdf", b"%PDF-1.4 broken")
    assert not svc.list_documents() and not any(s.upload_dir.iterdir())


# ---------------------------------------------------------------- deterministic table maths
@pytest.mark.parametrize("raw,expected", [("$1,234.50", 1234.5), ("(500)", -500.0), ("-3.2%", -3.2), ("12", 12.0),
                                          ("−7", -7.0), ("1.2M", None), ("N/A", None), ("", None), ("2024-01-05", None)])
def test_parse_number(raw, expected):
    assert parse_number(raw) == expected


def tbl():
    return TableRecord("t1", "f.pdf", 2, ["Region", "Sales 2023", "Sales 2024", "Growth"],
                       [["North", "100", "130", "30%"], ["South", "(50)", "40", "-5%"], ["East", "80", "60", "-25%"],
                        ["Total", "130", "230", "0%"]])


def test_sum_excludes_total_row_and_cross_checks():
    r = calculate("What are the total sales 2024?", tbl())
    assert "= 230" in r.text and "own 'Total' row states" in r.text


def test_negative_numbers_in_parentheses_are_summed_correctly():
    assert "= 130" in calculate("total sales 2023", tbl()).text  # 100 - 50 + 80


def test_max_min_return_the_row_label():
    assert "North" in calculate("which region has the highest sales 2024", tbl()).text
    assert "South" in calculate("lowest sales 2023", tbl()).text


def test_average_and_count():
    assert "= 76.67" in calculate("average sales 2024", tbl()).text
    assert ": 3" in calculate("how many regions are there", tbl()).text


def test_ordinal_column_comparison():
    r = calculate("Compare the values in the second and third columns", tbl())
    assert r.operation == "compare" and "Sales 2024 - Sales 2023" in r.text and "difference" in r.text


def test_row_lookup():
    r = calculate("What were the sales 2024 in East?", tbl())
    assert "East / Sales 2024: 60" in r.text


def test_ambiguous_numeric_column_asks_for_clarification():
    r = calculate("what is the average?", tbl())
    assert r.needs_clarification and "Sales 2023" in r.text


def test_irrelevant_question_gives_no_calculation():
    assert answer_from_tables("who founded the company?", [tbl()]) == []
    assert detect_ops("what is the highest mountain") == ["max"]  # op detection alone is harmless: no table -> no calc


def test_named_document_restricts_search_and_table_math(service, doc_ids):
    a = service.ask("What was Beta Ltd's revenue in Q1?")  # no explicit filter: the name in the question decides
    assert {s.filename for s in a.sources} == {"beta_ltd_report.pdf"}
    assert "800,000" in a.text and "1,200,000" not in a.text
    assert any("beta_ltd_report.pdf" in n for n in a.notes)


def test_both_names_keep_both_documents(service):
    a = service.ask("Which had higher revenue in Q1, Acme Corp or Beta Ltd?")
    assert {s.filename for s in a.sources} >= {"acme_report_2025.pdf", "beta_ltd_report.pdf"}
    assert "1,200,000" in a.text and "800,000" in a.text
