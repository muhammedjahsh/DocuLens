"""Query analysis + hybrid retrieval (vector + BM25, fused with Reciprocal Rank Fusion),
reranking, page targeting, multi-document balancing and an evidence-sufficiency check."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from config import Settings
from .embeddings import Embedder
from .reranker import Reranker
from .vector_store import Hit, VectorStore

STOPWORDS = set("""a an the of to in on at for from by with about as is are was were be been being this that these those it its
and or but if then than so such what which who whom whose when where why how do does did can could should would will may might
me my we our you your they their them he she his her i not no yes any all each both per into over under between among document
documents file files pdf uploaded upload according please tell show give find explain describe list state say shown""".split())


def stem(tok: str) -> str:
    """Plural-only stemming (predictable: 'objectives' == 'objective', 'companies' == 'company').
    Deeper morphology is left to the embedding model."""
    if tok.isdigit() or len(tok) <= 3 or "." in tok:
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith(("sses", "xes", "zes", "ches", "shes")):
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def content_terms(text: str) -> list[str]:
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


# --------------------------------------------------------------------------- query analysis
@dataclass
class QueryPlan:
    intents: set = field(default_factory=set)   # text | table | visual | summary | compare
    pages: list[int] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)

    @property
    def is_summary(self): return "summary" in self.intents
    @property
    def is_compare(self): return "compare" in self.intents
    @property
    def is_visual(self): return "visual" in self.intents
    @property
    def is_table(self): return "table" in self.intents


_RX = {
    "summary": re.compile(r"\b(summari[sz]e|summary|overview|tl;?dr|gist|main (findings?|points?|ideas?|takeaways?)|key (findings?|points?|takeaways?)|in brief)\b", re.I),
    "table": re.compile(r"\b(tables?|columns?|rows?|total|sum|average|mean|avg|highest|lowest|maximum|minimum|largest|smallest|how many|count|revenue|salary|salaries|profit|cost|price|percent(age)?|growth|%)\b", re.I),
    "visual": re.compile(r"\b(image|images|picture|photo|diagram|chart|graph|figure|illustration|screenshot|visual|plot|drawing|logo|depict|looks? like|shown in the (image|figure|chart))\b", re.I),
    "compare": re.compile(r"\b(compare|comparison|versus|vs\.?|differences?|differ|both|contrast)\b", re.I),
}
_PAGE_RX = re.compile(r"\bpages?\s+(\d+(?:\s*(?:,|and|&|-|to)\s*\d+)*)", re.I)


def analyze_query(q: str) -> QueryPlan:
    plan = QueryPlan(terms=content_terms(q))
    for name, rx in _RX.items():
        if rx.search(q):
            plan.intents.add(name)
    if not plan.intents - {"compare"}:
        plan.intents.add("text")
    for m in _PAGE_RX.finditer(q):
        nums = [int(n) for n in re.findall(r"\d+", m.group(1))]
        if re.search(r"-|to", m.group(1)) and len(nums) == 2 and nums[0] < nums[1] <= nums[0] + 30:
            nums = list(range(nums[0], nums[1] + 1))
        plan.pages.extend(nums)
    return plan


# --------------------------------------------------------------------------- BM25
class BM25:
    def __init__(self, hits: list[Hit], k1: float = 1.5, b: float = 0.75):
        self.hits, self.k1, self.b = hits, k1, b
        self.docs = [Counter(content_terms(h.text + " " + h.metadata.get("heading", ""))) for h in hits]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def top(self, terms: list[str], n: int) -> list[tuple[Hit, float]]:
        scored = []
        for hit, d, ln in zip(self.hits, self.docs, self.lens):
            s = 0.0
            for t in terms:
                f = d.get(t, 0)
                if f:
                    s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            if s > 0:
                scored.append((hit, s))
        return sorted(scored, key=lambda x: x[1], reverse=True)[:n]


# --------------------------------------------------------------------------- retriever
@dataclass
class RetrievalResult:
    hits: list[Hit]
    plan: QueryPlan
    sufficient: bool
    best_vector_sim: float = 0.0
    best_lexical: float = 0.0
    note: str = ""


class Retriever:
    def __init__(self, settings: Settings, store: VectorStore, embedder: Embedder, reranker: Reranker):
        self.s, self.store, self.embedder, self.reranker = settings, store, embedder, reranker
        self._bm25_cache: dict[tuple, BM25] = {}

    def _bm25(self, doc_ids: Optional[list[str]]) -> BM25:
        key = (getattr(self.store, "version", 0), tuple(sorted(doc_ids or [])))
        if key not in self._bm25_cache:
            self._bm25_cache = {key: BM25(self.store.get_all(doc_ids))}  # keep only the latest
        return self._bm25_cache[key]

    @staticmethod
    def _overlap(terms: list[str], text: str) -> float:
        if not terms:
            return 0.0
        present = set(content_terms(text))
        return sum(1 for t in set(terms) if t in present) / len(set(terms))

    def retrieve(self, query: str, doc_ids: Optional[list[str]] = None, k: Optional[int] = None,
                 doc_names: Optional[dict[str, str]] = None) -> RetrievalResult:
        k = k or self.s.top_k
        plan = analyze_query(query)
        n_docs = len(doc_ids) if doc_ids else None

        # 1. vector candidates
        vec_hits = self.store.query(self.embedder.embed_query(query), self.s.candidate_k, doc_ids)
        # 2. keyword candidates
        bm = self._bm25(doc_ids).top(plan.terms, self.s.candidate_k)
        # 3. reciprocal-rank fusion
        pool: dict[str, Hit] = {}
        prior: dict[str, float] = defaultdict(float)
        for rank, h in enumerate(vec_hits):
            pool[h.id] = h
            prior[h.id] += 1.0 / (60 + rank)
        for rank, (h, _) in enumerate(bm):
            if h.id in pool:
                pass
            else:
                pool[h.id] = h
            prior[h.id] += 1.0 / (60 + rank)
        # 4. page targeting ("explain the diagram on page 5")
        if plan.pages:
            for h in self.store.get_all(doc_ids):
                if h.metadata.get("page") in plan.pages:
                    pool.setdefault(h.id, h)
                    prior[h.id] += 0.05
        # 5. modality boosts
        for h in pool.values():
            ctype = h.metadata.get("content_type")
            boost = 1.0
            if plan.is_table and ctype == "table":
                boost = 1.35
            if plan.is_visual and ctype == "image":
                boost = 1.5
            prior[h.id] *= boost
            h.lexical = self._overlap(plan.terms, h.text + " " + h.metadata.get("heading", "")
                                      + " " + h.metadata.get("filename", ""))
        # 6. rerank
        ranked = self.reranker.rerank(query, list(pool.values()), dict(prior))

        # 7. selection: balanced across documents when comparing / multiple files named
        names_in_query = [d for d, n in (doc_names or {}).items() if n.lower() in query.lower()]
        balance = (plan.is_compare and (n_docs or 0) > 1) or len(names_in_query) > 1
        if balance:
            by_doc: dict[str, list[Hit]] = defaultdict(list)
            for h in ranked:
                by_doc[h.metadata["doc_id"]].append(h)
            selected, i = [], 0
            groups = list(by_doc.values())
            while len(selected) < k and any(i < len(g) for g in groups):
                for g in groups:
                    if i < len(g) and len(selected) < k:
                        selected.append(g[i])
                i += 1
        else:
            selected = ranked[:k]
        # always keep explicitly requested pages visible
        if plan.pages:
            forced = [h for h in ranked if h.metadata.get("page") in plan.pages]
            for h in forced[:k]:
                if h not in selected:
                    selected.append(h)
            selected = selected[: k + 2]

        best_vec = max((h.vector_sim for h in selected), default=0.0)
        best_lex = max((h.lexical for h in selected), default=0.0)
        sufficient = bool(selected) and (best_vec >= self.s.min_vector_similarity or best_lex >= self.s.min_lexical_overlap)
        note = ""
        if plan.is_summary and selected:
            sufficient = True
        if plan.is_visual and any(h.metadata.get("content_type") == "image" for h in selected):
            sufficient = True
        if plan.pages and any(h.metadata.get("page") in plan.pages for h in selected):
            sufficient = True
        if not selected:
            note = "No indexed content matched."
        return RetrievalResult(selected, plan, sufficient, best_vec, best_lex, note)
