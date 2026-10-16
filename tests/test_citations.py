from rag.answer_generator import AnswerGenerator, INSUFFICIENT
from rag.citations import build_sources, find_unsupported_numbers, format_location, validate_citations
from rag.vector_store import Hit
from tests.conftest import FakeLLM


def hit(i, text, page=3, ctype="text", fname="a.pdf", heading="Intro"):
    return Hit(f"id{i}", text, {"doc_id": "d", "filename": fname, "page": page, "content_type": ctype, "heading": heading,
                                "table_id": "", "image_path": "", "page_image": ""})


def test_validate_citations_strips_invented_labels():
    clean, cited, bad = validate_citations("Revenue grew [S1]. Costs fell [S2, S9]. Also [S7].", {"S1", "S2"})
    assert cited == ["S1", "S2"] and set(bad) == {"S9", "S7"}
    assert "S9" not in clean and "S7" not in clean and "[S1]" in clean and "[S2]" in clean


def test_validate_citations_leaves_normal_brackets_alone():
    clean, cited, _ = validate_citations("See [1] and [note] for details [S1].", {"S1"})
    assert "[1]" in clean and "[note]" in clean and cited == ["S1"]


def test_location_formatting():
    assert format_location("a.pdf", 3) == "a.pdf, p. 3"
    assert format_location("a.docx", None, "Methods") == "a.docx, section “Methods”"
    assert format_location("a.txt", None) == "a.txt"


def test_build_sources_labels_and_pages():
    src = build_sources([hit(1, "x" * 600), hit(2, "y", page=-1)])
    assert [s.label for s in src] == ["S1", "S2"] and src[0].page == 3 and src[1].page is None
    assert src[0].excerpt.endswith("…") and len(src[0].excerpt) < 500


def test_unsupported_numbers_detected():
    ev = "Revenue was $1,200,000 in Q1 and 12.5% growth."
    assert find_unsupported_numbers("Revenue was $1,200,000 [S1] with 12.5% growth.", ev) == []
    assert find_unsupported_numbers("Revenue was $1,300,000 with 12.5% growth.", ev) == ["1,300,000"]
    assert find_unsupported_numbers("There were 3 items in list 1) [S1]", ev) == []  # small ints ignored


def gen(settings, llm):
    return AnswerGenerator(settings, llm)


def test_generator_removes_fake_citation_and_warns(settings):
    llm = FakeLLM("Revenue was 1,200 [S1] and costs 900 [S5].")
    ans = gen(settings, llm).answer("rev?", [hit(1, "Revenue was 1,200 and costs 900.")], True, [], [], [], ["text"])
    assert "[S5]" not in ans.text and ans.cited_sources[0].label == "S1"
    assert any("did not match" in w for w in ans.warnings)


def test_generator_warns_on_unsupported_number_and_missing_citation(settings):
    ans = gen(settings, FakeLLM("Revenue was 9,999,999.")).answer("rev?", [hit(1, "Revenue was 1,200.")], True, [], [], [], ["text"])
    assert ans.unsupported_numbers == ["9,999,999"] and not ans.cited_sources
    assert any("did not cite" in w for w in ans.warnings)


def test_generator_does_not_call_llm_without_evidence(settings):
    llm = FakeLLM("should not be used")
    ans = gen(settings, llm).answer("q", [hit(1, "unrelated")], False, [], [], [], ["text"])
    assert ans.insufficient and ans.text == INSUFFICIENT and llm.calls == []


def test_generator_recognises_model_refusal(settings):
    ans = gen(settings, FakeLLM(INSUFFICIENT)).answer("q", [hit(1, "t")], True, [], [], [], ["text"])
    assert ans.insufficient


def test_generator_falls_back_to_evidence_when_llm_down(settings):
    ans = gen(settings, FakeLLM(fail=True)).answer("q", [hit(1, "The sky is blue.")], True, [], [], [], ["text"])
    assert "The sky is blue" in ans.text and any("unavailable" in w for w in ans.warnings)


def test_prompt_contains_rules_context_history_and_labels(settings):
    llm = FakeLLM("ok [S1]")
    history = [{"role": "user", "content": "earlier q"}, {"role": "assistant", "content": "earlier a"}]
    gen(settings, llm).answer("what?", [hit(1, "Fact A.")], True, history, [], [], ["text"])
    msgs, _ = llm.calls[0]
    assert msgs[0]["role"] == "system" and "Never invent" in msgs[0]["content"]
    assert [m["content"] for m in msgs[1:3]] == ["earlier q", "earlier a"]
    assert "[S1] file=a.pdf | page=3" in msgs[-1]["content"] and "Fact A." in msgs[-1]["content"]


def test_vision_model_receives_images_only_when_available(settings, tmp_path):
    img = tmp_path / "i.png"
    img.write_bytes(b"x")
    main, vis = FakeLLM("main [S1]"), FakeLLM("vision [S1]")
    g = AnswerGenerator(settings, main, vis)
    ans = g.answer("what is shown?", [hit(1, "chart", ctype="image")], True, [], [], [str(img)], ["visual"])
    assert ans.text.startswith("vision") and vis.calls[0][1] == [str(img)] and not main.calls
    ans2 = AnswerGenerator(settings, main, None).answer("what is shown?", [hit(1, "chart", ctype="image")], True, [], [], [str(img)], ["visual"])
    assert any("No vision model" in w for w in ans2.warnings)


def test_condense_question_only_for_followups(settings):
    llm = FakeLLM("What was Acme revenue in Q2?")
    g = gen(settings, llm)
    hist = [{"role": "user", "content": "Tell me about Acme"}, {"role": "assistant", "content": "Acme is a firm."}]
    assert g.condense_question("And in Q2?", hist) == "What was Acme revenue in Q2?"
    assert g.condense_question("What was Acme revenue in Q2 according to the quarterly table?", hist).startswith("What was Acme revenue in Q2 according")
    assert len(llm.calls) == 1
