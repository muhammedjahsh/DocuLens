"""Evaluation harness: retrieval relevance, answer correctness, citation accuracy,
refusal correctness, unsupported numbers and latency.

    python -m evaluation.run_eval                       # offline: hashing embeddings + extractive answers
    python -m evaluation.run_eval --embedding sentence-transformers --llm ollama
    python -m evaluation.run_eval --llm openai --vision auto --out report.json

Offline mode measures the retrieval + table-calculation + citation machinery, NOT language-model
answer quality. 'llm_only' and 'requires_vision' cases are skipped unless a generating
(and, for the latter, vision) model is configured.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Settings  # noqa: E402
from rag.embeddings import build_embedder  # noqa: E402
from rag.pipeline import RAGService  # noqa: E402
from scripts.generate_samples import generate_all  # noqa: E402


def run(args) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="rag_eval_"))
    s = Settings()
    s.data_dir = tmp / "data"
    s.embedding_backend, s.llm_provider = args.embedding, args.llm
    if args.vision:
        s.vision_provider = args.vision
    s.ensure_dirs()
    samples = generate_all(tmp / "samples")
    svc = RAGService(s, embedder=build_embedder(s))
    ids = {}
    for p in samples.values():
        if p.suffix:  # all generated deliverables
            info = svc.ingest_bytes(p.name, p.read_bytes())
            ids[info["filename"]] = info["doc_id"]
    cases = json.loads((Path(__file__).parent / "eval_dataset.json").read_text())
    generating, has_vision = svc.llm.generates, svc.vision is not None

    rows, skipped = [], []
    for c in cases:
        if c.get("llm_only") and not generating:
            skipped.append((c["id"], "needs a generating LLM")); continue
        if c.get("requires_vision") and not has_vision:
            skipped.append((c["id"], "needs a vision model")); continue
        doc_filter = [ids[f] for f in c["docs"]] if c.get("docs") else None
        a = svc.ask(c["question"], doc_ids=doc_filter)
        exp = [(e["file"], e["page"]) for e in c.get("expect_sources", [])]
        got = [(x.filename, x.page) for x in a.sources]
        cited = [(x.filename, x.page) for x in a.cited_sources]
        r = {"id": c["id"], "category": c["category"], "latency_s": a.timings.get("total_s", 0.0),
             "unsupported_numbers": len(a.unsupported_numbers)}
        if c.get("expect_insufficient"):
            r["refusal_ok"] = a.insufficient
        else:
            r["retrieval_ok"] = all(e in got for e in exp) if c.get("require_all_sources") else any(e in got for e in exp)
            subs = c.get("expect_substrings", []) + (c.get("expect_substrings_vision", []) if has_vision else [])
            low = a.text.lower()
            r["answer_ok"] = all(x.lower() in low for x in subs) and not any(x.lower() in low for x in c.get("expect_absent", []))
            r["citation_ok"] = (all(e in cited for e in exp) if c.get("require_all_sources") else any(e in cited for e in exp)) if exp else True
        rows.append(r)

    def rate(key):
        v = [r[key] for r in rows if key in r]
        return (sum(v) / len(v)) if v else None

    lat = [r["latency_s"] for r in rows]
    summary = {"config": {"embedding": args.embedding, "llm": svc.llm.name, "vision": svc.vision.name if svc.vision else None},
               "n_cases": len(rows), "skipped": skipped,
               "retrieval_hit_rate": rate("retrieval_ok"), "answer_correctness": rate("answer_ok"),
               "citation_accuracy": rate("citation_ok"), "refusal_correctness": rate("refusal_ok"),
               "answers_with_unsupported_numbers": sum(1 for r in rows if r["unsupported_numbers"]),
               "latency_mean_s": statistics.mean(lat) if lat else None,
               "latency_p95_s": sorted(lat)[max(0, int(0.95 * len(lat)) - 1)] if lat else None,
               "cases": rows}
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(all(v for k, v in r.items() if k.endswith("_ok")))
    summary["pass_by_category"] = {k: f"{sum(v)}/{len(v)}" for k, v in by_cat.items()}
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", default="hashing", choices=["hashing", "sentence-transformers"])
    ap.add_argument("--llm", default="extractive", choices=["extractive", "ollama", "openai", "huggingface"])
    ap.add_argument("--vision", default=None, help="auto|none|ollama|openai")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-pass", type=float, default=None, help="exit 1 if the overall pass rate is below this")
    args = ap.parse_args()
    res = run(args)
    pct = lambda x: "n/a" if x is None else f"{x:.0%}"
    print(f"\nConfig: {res['config']}\nCases run: {res['n_cases']}  skipped: {res['skipped']}")
    print(f"Retrieval hit rate : {pct(res['retrieval_hit_rate'])}\nAnswer correctness : {pct(res['answer_correctness'])}")
    print(f"Citation accuracy  : {pct(res['citation_accuracy'])}\nRefusal correctness: {pct(res['refusal_correctness'])}")
    print(f"Answers w/ unsupported numbers: {res['answers_with_unsupported_numbers']}")
    print(f"Latency mean/p95   : {res['latency_mean_s']:.2f}s / {res['latency_p95_s']:.2f}s")
    print("Pass by category   :", res["pass_by_category"])
    for r in res["cases"]:
        bad = [k for k, v in r.items() if k.endswith("_ok") and not v]
        if bad:
            print(f"  FAIL {r['id']}: {', '.join(bad)}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
    if args.min_pass is not None:
        flags = [all(v for k, v in r.items() if k.endswith("_ok")) for r in res["cases"]]
        return 0 if flags and sum(flags) / len(flags) >= args.min_pass else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
