"""Streamlit UI for the multimodal RAG chatbot.   Run:  streamlit run app.py"""
from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from config import get_settings
from rag.pipeline import RAGService
from utils.logging_config import setup_logging
from utils.validation import ValidationError

st.set_page_config(page_title="Document Chat (Multimodal RAG)", page_icon="📄", layout="wide")
setup_logging(get_settings().log_level)
log = logging.getLogger("app")

st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 1.5rem;}
[data-testid="stChatMessage"] {border-radius: 12px; padding: .6rem .9rem;}
.src-label {font-weight: 600; color: #1d4ed8;}
.doc-meta {color: #6b7280; font-size: .78rem; margin-top: -.4rem;}
</style>""", unsafe_allow_html=True)

EXAMPLES = ["Summarize this document.", "What are the main findings in this PDF?", "What is the total revenue shown in the table?",
            "Explain the diagram on page 3.", "Compare the information from the two uploaded documents.",
            "Show the page number and source supporting your answer."]


@st.cache_resource(show_spinner="Loading models and index…")
def get_service() -> RAGService:
    return RAGService()


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("seen_files", set())
    st.session_state.setdefault("errors", {})
    st.session_state.setdefault("uploader_key", 0)


def ingest_uploads(svc: RAGService, files) -> None:
    for f in files:
        key = (f.name, f.size)
        if key in st.session_state.seen_files:
            continue
        st.session_state.seen_files.add(key)
        with st.status(f"Processing {f.name}…", expanded=True) as status:
            try:
                info = svc.ingest_bytes(f.name, f.getvalue(), progress=lambda m: status.write(m))
                msg = "already indexed" if info.get("duplicate") else f"indexed ({info['n_chunks']} chunks)"
                status.update(label=f"{f.name}: {msg}", state="complete", expanded=False)
                st.session_state.errors.pop(f.name, None)
            except ValidationError as exc:
                st.session_state.errors[f.name] = str(exc)
                status.update(label=f"{f.name}: rejected", state="error")
            except ValueError as exc:  # corrupted / unreadable / empty
                st.session_state.errors[f.name] = str(exc)
                status.update(label=f"{f.name}: could not be processed", state="error")
            except Exception as exc:  # pragma: no cover - last resort, never crash the UI
                log.exception("Unexpected ingestion failure")
                st.session_state.errors[f.name] = f"Unexpected error while processing '{f.name}': {type(exc).__name__}"
                status.update(label=f"{f.name}: failed", state="error")


def sidebar(svc: RAGService) -> list[str]:
    with st.sidebar:
        st.header("📄 Documents")
        s = svc.s
        files = st.file_uploader("Upload PDF, DOCX, TXT, PNG, JPG", type=[e[1:] for e in s.allowed_extensions],
                                 accept_multiple_files=True, key=f"up{st.session_state.uploader_key}",
                                 help=f"Up to {s.max_file_mb} MB per file.")
        if files:
            ingest_uploads(svc, files)
        for name, err in list(st.session_state.errors.items()):
            st.error(err)
        docs = svc.list_documents()
        selected = []
        if docs:
            st.subheader("Indexed documents")
            st.caption("Tick the documents to search.")
        for d in docs:
            c1, c2 = st.columns([5, 1])
            if c1.checkbox(d["filename"], value=True, key=f"sel_{d['doc_id']}"):
                selected.append(d["doc_id"])
            if c2.button("🗑", key=f"rm_{d['doc_id']}", help=f"Remove {d['filename']}"):
                svc.remove_document(d["doc_id"])
                st.rerun()
            pages = f"{d['n_pages']} pages · " if d.get("n_pages") else ""
            st.markdown(f"<div class='doc-meta'>{pages}{d['n_chunks']} chunks · {d['n_tables']} tables · {d['n_images']} images"
                        f"{' · OCR p.' + ','.join(map(str, d['ocr_pages'])) if d['ocr_pages'] else ''}</div>", unsafe_allow_html=True)
            for w in d.get("warnings", []):
                st.caption(f"⚠️ {w}")
        if docs:
            with st.expander("Preview a page / image"):
                pick = st.selectbox("Document", docs, format_func=lambda d: d["filename"], key="prev_doc")
                pages = sorted(int(p) for p in pick.get("page_images", {}))
                if pages:
                    pg = st.number_input("Page", min_value=pages[0], max_value=pages[-1], value=pages[0], step=1) if len(pages) > 1 else pages[0]
                    path = svc.page_preview(pick["doc_id"], int(pg))
                    if path:
                        st.image(path, use_container_width=True)
                else:
                    st.caption("No page images for this file type.")
        st.divider()
        b1, b2 = st.columns(2)
        if b1.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        if b2.button("Remove all", use_container_width=True, disabled=not docs, help="Delete every document and its index"):
            svc.remove_all()
            st.session_state.seen_files, st.session_state.errors = set(), {}
            st.session_state.uploader_key += 1
            st.rerun()
        with st.expander("Configuration"):
            for k, v in svc.config_summary().items():
                st.markdown(f"**{k}:** {v}")
            if not svc.ocr.available:
                st.warning(svc.ocr.reason)
    return selected


def render_sources(ans) -> None:
    shown = ans.cited_sources or ans.sources[:3]
    if not shown:
        return
    title = "Sources" if ans.cited_sources else "Closest retrieved passages (not cited by the model)"
    with st.expander(f"📎 {title} ({len(shown)})"):
        for s in shown:
            st.markdown(f"<span class='src-label'>[{s.label}]</span> **{s.location}** · {s.content_type}"
                        f"{' · ' + s.heading if s.heading else ''}", unsafe_allow_html=True)
            if s.content_type == "table":
                st.markdown(s.excerpt.split("\n", 1)[-1] if "\n" in s.excerpt else s.excerpt)
            else:
                st.caption(s.excerpt)
            img = s.image_path or (s.page_image if s.content_type in ("image", "ocr_text") else "")
            if img and Path(img).exists():
                st.image(img, width=360)


def render_message(m: dict) -> None:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        ans = m.get("answer")
        if ans:
            for n in ans.notes:
                st.caption(f"ℹ️ {n}")
            for w in ans.warnings:
                st.warning(w, icon="⚠️")
            render_sources(ans)
            if ans.calc:
                with st.expander("🧮 Deterministic table calculation"):
                    for c in ans.calc:
                        st.markdown(c.text)
            t = ans.timings
            if t:
                st.caption(f"⏱ {t.get('total_s', 0):.1f}s total · retrieval {t.get('retrieval_s', 0):.2f}s · generation {t.get('generation_s', 0):.1f}s"
                           + (f" · rewritten as: “{ans.standalone_question}”" if ans.standalone_question and ans.standalone_question != m.get('question') else ""))


def main() -> None:
    init_state()
    try:
        svc = get_service()
    except Exception as exc:
        st.error(f"Could not start the application: {exc}")
        st.info("Check your .env settings (see README → Troubleshooting).")
        st.stop()
    selected = sidebar(svc)

    st.title("Chat with your documents")
    st.caption("Answers are grounded in your uploaded files, with file and page citations.")
    if not st.session_state.messages:
        if not svc.list_documents():
            st.info("👈 Upload one or more documents to begin.")
        st.markdown("**Try asking:**")
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
                st.session_state.pending = ex
                st.rerun()
    for m in st.session_state.messages:
        render_message(m)

    prompt = st.chat_input("Ask a question about your documents…") or st.session_state.pop("pending", None)
    if prompt:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching your documents…"):
                try:
                    ans = svc.ask(prompt, history=history, doc_ids=selected or None if svc.list_documents() else None)
                    text = ans.text
                except Exception as exc:
                    log.exception("Answer generation failed")
                    ans, text = None, f"Sorry, something went wrong while answering ({type(exc).__name__}). Please try again."
        st.session_state.messages.append({"role": "assistant", "content": text, "answer": ans, "question": prompt})
        st.rerun()


main()
