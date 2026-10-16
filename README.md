# Multimodal RAG Chatbot (PDF · DOCX · TXT · images · tables · scans)

Upload documents, ask questions, get answers grounded in the files with **file + page citations**.
Tables are queried with **pandas** (not LLM arithmetic), scanned pages are OCR'd, and charts/diagrams can
be sent to a **vision model** as the original image.

---------------------------------------------------------------------------------------------------

## 1. Architecture and data flow

```
                  ┌──────────────────────────── INGESTION ───────────────────────────────┐
 upload ─► validate ─► extract ─► normalise ─► chunk ─► embed ─► ChromaDB (vectors + metadata)
          (ext, size,   │                                  │
          magic bytes,  ├─ PDF  : PyMuPDF text+headings, tables (PyMuPDF→pdfplumber),
          zip-bomb)     │         embedded images, page renders, OCR for scanned pages
                        ├─ DOCX : python-docx body order, tables, images
                        ├─ Image: Pillow → OCR (+ optional vision caption)
                        └─ TXT
                                   └─► DocumentRegistry (JSON): structured TABLES, image paths, page renders
                  └───────────────────────────────────────────────────────────────────────┘

                  ┌──────────────────────────── QUERY ───────────────────────────────────┐
 question ─► follow-up rewrite ─► name-hint (“Beta Ltd” ⇒ that file) ─► intent analysis
   (history)    (LLM, only if needed)                               text│table│visual│summary│compare│page N
        ├─ summary intent ─► map-reduce over the document's chunks
        └─ otherwise ─► hybrid retrieval: vector (Chroma, cosine) + BM25 ─► RRF fusion
                         ─► modality boosts, page targeting ─► rerank ─► per-document balancing
                         ─► evidence check (vector similarity OR keyword overlap) ── fail ─► "not enough information" (no LLM call)
              ├─ table/aggregate intent ─► pandas on the FULL stored table ─► VERIFIED CALCULATIONS
              ├─ visual intent ─► attach original image / rendered page to the vision model
              └─► prompt (rules + [S#]-labelled context + calculations) ─► LLM
                  ─► strip invented [S#] ─► flag numbers absent from evidence ─► answer + sources
                  └───────────────────────────────────────────────────────────────────────┘
```

| Content | How it is extracted | How it is indexed | How it is answered |
|---|---|---|---|
| **Text** | PyMuPDF blocks (tables excluded), heading detection by font size; python-docx paragraphs/headings | Paragraph→sentence chunks (900 chars, 150 overlap), never crossing pages; prefixed with `[file \| page \| section]` | Retrieved passages with `[S#]` citations |
| **Tables** | PyMuPDF `find_tables()` → pdfplumber fallback; python-docx tables (merged cells handled) | Markdown chunks (header repeated every 25 rows) **and** the full structured table in the registry | Deterministic pandas: sum/mean/min/max/count/compare columns/row lookup; `Total` rows are never double-counted; ambiguity ⇒ clarifying question |
| **Images / diagrams / charts** | Embedded images saved (size-filtered); standalone images normalised | OCR text (+ optional caption) as a searchable chunk; original path kept in metadata | Original image (or rendered page) is sent to the **vision model**; OCR text alone is *not* treated as a full description |
| **Scanned pages** | Page has < 40 chars of text ⇒ render at 300 dpi ⇒ OpenCV denoise/threshold/deskew ⇒ Tesseract (best of raw vs. pre-processed) | `ocr_text` chunks, low-confidence warning stored | Same as text; user sees a warning when OCR confidence is low |

Design choices worth knowing
* **No FastAPI service**: the Streamlit app calls `RAGService` directly (one process, simpler to deploy). `rag/pipeline.py` is the seam if you later want an API.
* **Core RAG logic is plain Python** (no LangChain/LlamaIndex) so every step is testable.
* **DOCX has no pages** → citations show the section heading instead; TXT shows the file name only. Page numbers are never invented.
* Switching embedding model creates a new Chroma collection automatically (dimensions differ) – re-upload files after changing it.

## 2. Model compatibility (read this)

* **Phi-3 Mini is text-only.** `phi3:mini` (Ollama) / `microsoft/Phi-3-mini-4k-instruct` (HF) cannot see images. For charts, diagrams and photos configure a separate vision model: `VISION_PROVIDER=ollama VISION_MODEL=llava:7b` (or `llama3.2-vision`, `minicpm-v`), or use OpenAI `gpt-4o-mini` (vision works out of the box with `LLM_PROVIDER=openai`). Without one, image questions are answered from OCR text and the UI says so.
* Phi-3-mini-4k has a **4k-token window**; defaults (`MAX_CONTEXT_CHARS=7000`, `NUM_CTX=4096`) fit. For longer contexts use the 128k variant and raise both.
* `BAAI/bge-small-en-v1.5` is English-oriented; for other languages set `EMBEDDING_MODEL=BAAI/bge-m3` (and `OCR_LANG` accordingly).
* Tesseract is a system binary, not a pip package.

## 3. Installation

Python 3.10–3.12 recommended.

### Linux / macOS
```bash
sudo apt-get install tesseract-ocr          # macOS: brew install tesseract
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# LLM (default: Ollama + Phi-3 Mini)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3:mini
ollama pull llava:7b                         # optional: vision
streamlit run app.py
```

### Windows (PowerShell)
```powershell
# 1. Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki  (default path below)
# 2. Install Ollama: https://ollama.com/download/windows   then:  ollama pull phi3:mini
python -m venv .venv ; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# in .env set:  TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
streamlit run app.py
```

### Google Colab
```python
!apt-get -qq install -y tesseract-ocr zstd
!pip -q install -r requirements.txt pyngrok
!curl -fsSL https://ollama.com/install.sh | sh
import subprocess, time; subprocess.Popen(["ollama","serve"]); time.sleep(5)
!ollama pull phi3:mini
# expose Streamlit (needs a free ngrok token):
from pyngrok import ngrok; ngrok.set_auth_token("YOUR_TOKEN"); print(ngrok.connect(8501))
!streamlit run app.py --server.port 8501 --server.headless true &
```
No GPU/Ollama? `LLM_PROVIDER=openai` with a key in Colab secrets, or `LLM_PROVIDER=huggingface` (`pip install -r requirements-hf.txt`; needs a GPU for usable speed).

### No LLM at all
`LLM_PROVIDER=extractive` returns verbatim evidence + computed table results with citations (also the automatic fallback if the LLM is down).

## 4. Configuration
All settings live in `.env` (see `.env.example`, every key documented). Secrets are read only from the environment, never logged (a logging filter masks `sk-…` keys), and never sent to the browser.

## 5. Using it
Upload files in the sidebar → they are indexed automatically. Tick/untick documents to filter; 🗑 removes one; **Remove all** wipes files, vectors and registry; **Clear chat** resets the conversation. Naming a document in the question (“Beta Ltd's revenue…”) restricts the search to it (shown as a note).

| Question | Expected behaviour |
|---|---|
| “Summarize this document.” | Map-reduce summary of the selected document(s), cited per section |
| “What is the total revenue shown in the table?” | pandas total (excluding the table's own *Total* row, which is cross-checked), cited to the table's page |
| “Find the employee with the highest salary.” | Returns the row (name + value) computed from the full table |
| “Compare the values in the second and fourth columns.” | Row-wise differences + sums, computed deterministically |
| “What is the average?” (several numeric columns) | Asks which column |
| “Explain the diagram on page 5.” | Pulls page 5 chunks + sends page/figure image to the vision model; OCR-only warning if none configured |
| “Compare the two uploaded PDFs.” | Retrieval is balanced across the selected documents; answer attributes facts per file |
| “What is the CEO's favourite colour?” | “The uploaded documents do not contain sufficient information…”, no LLM call |
| “And in Q2?” (follow-up) | Rewritten into a standalone query using chat history |
| “Show the page number and source.” | Sources expander lists `file, p. N`, excerpt, table or image preview |

## 6. Tests and evaluation
```bash
pip install -r requirements-dev.txt
python scripts/generate_samples.py          # writes tests/samples/ (PDFs, DOCX, scanned PDF, PNGs, TXT)
pytest                                      # 75 tests
python -m evaluation.run_eval               # offline: hashing embeddings + extractive answers
python -m evaluation.run_eval --embedding sentence-transformers --llm ollama --vision ollama --out report.json
```
* `tests/` – ingestion (text/tables/OCR/scans/corruption), validation, chunking, retrieval (filters, page targeting, multi-doc, refusal), table maths, citations (invented-citation stripping, number checks), LLM clients against a mock server, and the Streamlit UI via `AppTest`.
* `evaluation/eval_dataset.json` – 19 questions across factual, table lookup, calculation, multi-document, OCR, visual, multi-citation and refusal categories. It reports retrieval hit rate, answer correctness, citation accuracy, refusal correctness, answers containing unsupported numbers, and latency (mean/p95).
* **Offline mode measures retrieval, calculation, citation plumbing and refusal – not language-model quality.** `llm_only` / `requires_vision` cases are skipped unless a generating/vision model is configured.
* **Calibrate thresholds** for your embedding model: `MIN_VECTOR_SIMILARITY` decides when to say “not enough information”. Run the eval with your real embedder and adjust until answerable questions pass and unanswerable ones refuse.

## 7. Troubleshooting
| Symptom | Fix |
|---|---|
| “OCR unavailable … tesseract not found” | Install Tesseract; on Windows set `TESSERACT_CMD` |
| “Cannot reach Ollama” / “model not installed” | `ollama serve`; `ollama pull phi3:mini` |
| Image questions answered “from OCR text only” | Configure `VISION_PROVIDER`/`VISION_MODEL` |
| Everything answers “not enough information” | `MIN_VECTOR_SIMILARITY` too high for your embedder – lower it (run the eval) |
| It answers things that are not in the docs | Raise `MIN_VECTOR_SIMILARITY`; use a stronger LLM; check the ⚠️ “figures not found in evidence” warnings |
| Table not detected / columns merged | Try `TABLE_BACKEND=pdfplumber`; borderless tables are the hardest case |
| Scanned PDF returns little text | Check the OCR warning; confirm `OCR_LANG`; scans below ~200 dpi OCR poorly |
| Slow first run | `sentence-transformers` downloads the embedding model once |
| “Dimension mismatch” after changing the embedding model | Not expected (new collection per model); delete `data/chroma` if you changed models mid-run |
| Upload rejected | Message states why (type, size, magic bytes, corrupted DOCX); limit is `MAX_FILE_MB` and Streamlit's `.streamlit/config.toml` `maxUploadSize` |

## 8. Deployment
* **Docker**: `docker compose up --build`, then `docker compose exec ollama ollama pull phi3:mini` (and `llava:7b`). Data persists in the `appdata` volume. The image runs as a non-root user.
* **Streamlit Community Cloud**: works with `packages.txt` (installs Tesseract) but there is no GPU/Ollama – use `LLM_PROVIDER=openai` and put `OPENAI_API_KEY` in *Secrets*. Storage is ephemeral.
* **Hugging Face Spaces (Docker SDK), Render, Fly.io, Azure/AWS/GCP containers**: deploy the Dockerfile; mount a persistent volume at `/data`; set env vars as secrets; put an authenticating reverse proxy (or the platform's auth) in front – **the app itself has no user accounts**, and all users share one document index.
* For multi-user production: add authentication, per-user `DATA_DIR`/collections, a client/server vector DB (Qdrant) behind the `VectorStore` interface, and a job queue for ingestion.

## 9. Security notes
Extension allow-list + magic-byte check + size cap + zip-bomb check (DOCX) + page/pixel limits; filenames sanitised (no path traversal); uploads stored `0600` under a random-id directory and only ever *parsed* (PyMuPDF/python-docx/Pillow never execute macros, scripts or embedded files); failed ingests are cleaned up; document text is labelled untrusted in the prompt (prompt-injection mitigation – reduces but does not eliminate the risk); no secrets in logs or the frontend.

## 10. Limitations (honest list)
* **OCR**: Tesseract struggles with handwriting, low-resolution scans, stylised fonts, and short labels inside boxes (the sample flow-diagram loses “CEO”/“CTO”). Check OCR warnings before trusting figures.
* **Charts/diagrams**: understanding depends entirely on the vision model; small local models misread axes and small numbers. Treat chart-derived values as interpretations, not extractions.
* **Tables**: borderless/rotated/spanning tables may be mis-detected. The calculator is a **heuristic** (matches column names, ordinals, row labels); values like `1.2M` are deliberately *not* parsed rather than mis-scaled; European `1.234,50` formatting is unsupported.
* **Refusal** uses similarity/keyword thresholds plus the LLM's grounding rule; a question that overlaps a document's vocabulary but asks for something absent (e.g. a year not present) is caught only by the LLM.
* DOCX/TXT have no page numbers; very long documents are summarised from an evenly spaced sample (and the UI says so).
* English-first defaults. Single shared index (no per-user isolation).
* Camelot is not wired in (pdfplumber is the fallback); ColPali-style page-image retrieval is not implemented.

## 11. What has and has not been verified
**Run and passing in development (Python 3.12, Tesseract 5.3):** all 75 tests; offline evaluation; PyMuPDF 1.28, pdfplumber 0.11, python-docx 1.2, ChromaDB 1.5, Streamlit 1.66; OCR on a rotated scanned PDF, a PNG invoice and an embedded chart; Ollama/OpenAI client wire formats against a mock server.
**Not run in that environment (no model downloads available):** the real `BAAI/bge-small-en-v1.5` embeddings, a real Phi-3/LLaVA/OpenAI response, the cross-encoder reranker, and the HuggingFace Transformers path. Those code paths are implemented to the documented APIs but you should run `python -m evaluation.run_eval --embedding sentence-transformers --llm ollama` on your machine and tune `MIN_VECTOR_SIMILARITY` before relying on them.

## 12. Project layout
```
app.py                      Streamlit UI
config.py                   settings (env-driven)
ingestion/                  models, pdf/docx/image processors, ocr_processor, table_extractor, pipeline (dispatch)
rag/                        chunking, embeddings, vector_store (+registry), retriever, reranker, table_calc,
                            llm, answer_generator, citations, pipeline (RAGService)
utils/                      validation, logging_config
scripts/generate_samples.py sample corpus generator
evaluation/                 eval_dataset.json, run_eval.py
tests/                      75 automated tests
```
