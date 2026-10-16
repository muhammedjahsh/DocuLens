import io
import zipfile

import pytest

from ingestion.pipeline import extract_document
from rag.chunking import build_chunks, split_text
from utils.validation import ValidationError, sanitize_filename, validate_upload


def run(samples, key, settings, ocr):
    p = samples[key]
    return extract_document(p, doc_id=key, filename=p.name, settings=settings, ocr=ocr)


# ---------------------------------------------------------------- validation
def test_validation_rejects_bad_uploads(settings):
    with pytest.raises(ValidationError, match="Unsupported"):
        validate_upload("x.exe", b"MZ....", settings)
    with pytest.raises(ValidationError, match="empty"):
        validate_upload("a.pdf", b"", settings)
    with pytest.raises(ValidationError, match="valid PDF"):
        validate_upload("a.pdf", b"not a pdf", settings)
    settings.max_file_mb = 1
    with pytest.raises(ValidationError, match="limit"):
        validate_upload("a.pdf", b"%PDF" + b"0" * 2_000_000, settings)
    with pytest.raises(ValidationError, match="corrupted"):
        validate_upload("a.docx", b"PK\x03\x04garbage", settings)
    with pytest.raises(ValidationError, match="binary"):
        validate_upload("a.txt", b"abc\x00\x00def", settings)


def test_filename_sanitised():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("C:\\evil\\report<>.pdf") == "report__.pdf"
    assert sanitize_filename("") == "upload"


def test_docx_without_document_xml_rejected(settings):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("hello.txt", "x")
    with pytest.raises(ValidationError, match="not a valid Word"):
        validate_upload("a.docx", buf.getvalue(), settings)


# ---------------------------------------------------------------- PDF
def test_pdf_pages_headings_and_text(samples, settings, ocr):
    r = run(samples, "acme", settings, ocr)
    assert r.n_pages == 4
    p1 = next(e for e in r.elements if e.kind == "text" and e.page == 1)
    assert "Project Objectives" in p1.text and "churn" in p1.text
    assert {e.heading for e in r.elements if e.page == 4} == {"Risks and Outlook"}


def test_pdf_table_structure_and_values(samples, settings, ocr):
    r = run(samples, "acme", settings, ocr)
    assert len(r.tables) == 1
    t = r.tables[0]
    assert t.page == 2 and t.heading == "Quarterly Financials"
    assert t.headers == ["Quarter", "Revenue (USD)", "Expenses (USD)", "Net Profit (USD)"]
    q3 = next(row for row in t.rows if row[0] == "Q3")
    assert q3 == ["Q3", "1,100,000", "1,250,000", "(150,000)"]  # negative in parentheses preserved verbatim
    page2_prose = " ".join(e.text for e in r.elements if e.kind == "text" and e.page == 2)
    assert "1,200,000" not in page2_prose  # table text is not duplicated as prose


def test_pdf_embedded_image_extracted_and_page_rendered(samples, settings, ocr):
    r = run(samples, "acme", settings, ocr)
    assert len(r.images) == 1 and r.images[0].page == 3
    assert set(r.page_images) == {1, 2, 3, 4}


def test_chart_ocr_reads_labels(samples, settings, ocr):
    if not ocr.available:
        pytest.skip(ocr.reason)
    r = run(samples, "acme", settings, ocr)
    assert "42%" in r.images[0].ocr_text and "Europe" in r.images[0].ocr_text


def test_scanned_pdf_uses_ocr(samples, settings, ocr):
    if not ocr.available:
        pytest.skip(ocr.reason)
    r = run(samples, "scanned", settings, ocr)
    assert r.ocr_pages == [1]
    txt = r.elements[0].text
    assert "$75,000" in txt and "Kerala" in txt


def test_scanned_pdf_without_ocr_warns(samples, settings):
    settings.ocr_enabled = False
    from ingestion.ocr_processor import OCREngine

    r = extract_document(samples["scanned"], doc_id="s", filename="scanned_memo.pdf", settings=settings, ocr=OCREngine(settings))
    assert any("OCR" in w for w in r.warnings)


def test_corrupted_pdf_raises(tmp_path, settings, ocr):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4\nthis is not really a pdf")
    with pytest.raises(ValueError):
        extract_document(bad, doc_id="x", filename="bad.pdf", settings=settings, ocr=ocr)


# ---------------------------------------------------------------- DOCX / images / txt
def test_docx_structure(samples, settings, ocr):
    r = run(samples, "docx", settings, ocr)
    assert r.tables[0].headers == ["Name", "Department", "Salary (USD)", "Years"]
    assert len(r.tables[0].rows) == 5
    assert r.tables[0].heading == "Employee Directory"
    assert r.elements[0].page is None
    assert len(r.images) == 1


def test_image_ocr_invoice(samples, settings, ocr):
    if not ocr.available:
        pytest.skip(ocr.reason)
    r = run(samples, "invoice", settings, ocr)
    t = r.images[0].ocr_text
    assert "1,284.50" in t and "30 April 2025" in t


def test_txt(samples, settings, ocr):
    r = run(samples, "txt", settings, ocr)
    assert "PostgreSQL" in r.elements[0].text


# ---------------------------------------------------------------- chunking
def test_chunks_never_cross_pages_and_carry_metadata(samples, settings, ocr):
    r = run(samples, "acme", settings, ocr)
    chunks = build_chunks(r, settings)
    assert chunks and all(c.metadata["filename"] == "acme_report_2025.pdf" for c in chunks)
    table_chunks = [c for c in chunks if c.metadata["content_type"] == "table"]
    assert table_chunks and table_chunks[0].metadata["page"] == 2 and table_chunks[0].metadata["table_id"]
    assert "Revenue (USD)" in table_chunks[0].text
    assert all(c.metadata["page_image"] for c in chunks)


def test_split_text_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    parts = split_text(text, 300, 60)
    assert all(len(p) <= 400 for p in parts) and len(parts) > 5
    assert split_text("short", 300, 60) == ["short"]
    assert split_text("x" * 2000, 300, 0)[0] == "x" * 300  # no-punctuation fallback


def test_large_table_is_split_with_repeated_header(samples, settings, ocr):
    r = run(samples, "acme", settings, ocr)
    settings.table_rows_per_chunk = 2
    tc = [c for c in build_chunks(r, settings) if c.metadata["content_type"] == "table"]
    assert len(tc) == 3 and all("| Quarter |" in c.text for c in tc)
