"""Deterministic table calculations with pandas.

The LLM is never trusted to do arithmetic. For questions on totals, averages, min/max,
counts, column comparisons and row lookups, this module computes the answer from the
structured table and hands the verified result to the LLM (or shows it directly).

Heuristic by design: it matches column names / ordinals / row labels in the question.
When it cannot decide which column is meant, it says so (needs_clarification) instead of guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from ingestion.models import TableRecord

_NUM = re.compile(r"^\d{1,3}(,\d{3})+(\.\d+)?$|^\d+(\.\d+)?$|^\.\d+$")
_CURRENCY = re.compile(r"[$€£₹¥]|\b(USD|EUR|GBP|INR|Rs\.?)\b", re.I)
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
            "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6, "7th": 7, "8th": 8}
OPS = {
    "sum": r"\b(total|sum|add(ed)? up|combined|overall|altogether)\b",
    "mean": r"\b(average|mean|avg)\b",
    "max": r"\b(highest|largest|maximum|max|greatest|biggest|top|most)\b",
    "min": r"\b(lowest|smallest|minimum|min|least|fewest)\b",
    "count": r"\b(how many|count|number of (rows|entries|records|employees|items|people))\b",
    "compare": r"\b(compare|comparison|difference|differ|versus|vs\.?)\b",
}


def parse_number(raw: object) -> Optional[float]:
    """'$1,234.50' -> 1234.5 ; '(500)' -> -500 ; '-3.2%' -> -3.2 ; 'N/A' -> None.
    Values with scale suffixes ('1.2M') or other text deliberately return None (no silent mis-scaling)."""
    s = str(raw if raw is not None else "").strip()
    if not s:
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    s = s.replace("−", "-").replace("–", "-").replace("\u00a0", " ")
    s = _CURRENCY.sub("", s).replace("%", "").replace(" ", "")
    if s.startswith(("-", "+")):
        neg = neg ^ (s[0] == "-")
        s = s[1:]
    if s.endswith("-"):  # trailing minus (accounting style)
        neg, s = not neg, s[:-1]
    if not _NUM.match(s):
        return None
    v = float(s.replace(",", ""))
    return -v if neg else v


def _unit(values: list[str]) -> str:
    joined = " ".join(values)
    m = re.search(r"[$€£₹¥]", joined)
    if m:
        return m.group(0)
    return "%" if values and sum("%" in v for v in values) > len(values) / 2 else ""


def fmt(v: float, unit: str = "") -> str:
    txt = f"{v:,.2f}".rstrip("0").rstrip(".") if abs(v) < 1e15 else str(v)
    if unit == "%":
        return f"{txt}%"
    return f"{unit}{txt}" if unit else txt


@dataclass
class CalcResult:
    table_id: str
    filename: str
    page: Optional[int]
    operation: str
    text: str                       # human-readable, deterministic result
    needs_clarification: bool = False
    table_markdown: str = ""


@dataclass
class _Prepared:
    rec: TableRecord
    df: pd.DataFrame
    num: dict[str, pd.Series]       # numeric columns -> floats (NaN where unparsable)
    units: dict[str, str]
    label_cols: list[str]
    total_mask: pd.Series


def _prepare(rec: TableRecord) -> _Prepared:
    df = rec.to_dataframe()
    num, units = {}, {}
    for c in df.columns:
        vals = [v for v in df[c].tolist() if str(v).strip()]
        parsed = df[c].map(parse_number)
        if vals and parsed.notna().sum() / len(vals) >= 0.6:
            num[c] = parsed.astype(float)
            units[c] = _unit(vals)
    label_cols = [c for c in df.columns if c not in num]
    is_total = pd.Series(False, index=df.index)
    for c in label_cols:
        is_total |= df[c].str.strip().str.lower().str.match(r"^(grand\s+)?(sub)?total\b|^sum$|^overall$")
    return _Prepared(rec, df, num, units, label_cols, is_total)


def detect_ops(q: str) -> list[str]:
    ops = [name for name, rx in OPS.items() if re.search(rx, q, re.I)]
    if "count" in ops and len(ops) > 1:
        ops.remove("count")
    return ops


def _stem(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") else t


def _tokens(s: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", s.lower())]


def _match_columns(q: str, p: _Prepared) -> list[str]:
    qt = set(_tokens(q))
    # ordinal columns: "second and fourth columns"
    ords = []
    if re.search(r"\bcolumns?\b", q, re.I):
        for m in re.finditer(r"\b(" + "|".join(ORDINALS) + r")\b", q, re.I):
            idx = ORDINALS[m.group(1).lower()] - 1
            if idx < len(p.df.columns):
                ords.append(p.df.columns[idx])
    if ords:
        return list(dict.fromkeys(ords))
    scored = []
    for c in p.df.columns:
        ht = [t for t in _tokens(c) if t not in {"usd", "eur", "inr", "gbp"}] or _tokens(c)
        if ht:
            scored.append((sum(t in qt for t in ht) / len(ht), c))
    best = max((s for s, _ in scored), default=0)
    if best < 0.5:
        return []
    return [c for s, c in scored if s == best]


def _match_rows(q: str, p: _Prepared) -> pd.Series:
    ql = " " + re.sub(r"[^a-z0-9 ]", " ", q.lower()) + " "
    mask = pd.Series(False, index=p.df.index)
    for c in p.label_cols:
        for i, v in p.df[c].items():
            vv = re.sub(r"[^a-z0-9 ]", " ", str(v).lower()).strip()
            if len(vv) >= 2 and not p.total_mask[i] and f" {vv} " in ql:
                mask[i] = True
    return mask


def _row_label(p: _Prepared, i) -> str:
    for c in p.label_cols:
        if str(p.df.at[i, c]).strip():
            return f"{p.df.at[i, c]}"
    return f"row {i + 1}"


def calculate(question: str, rec: TableRecord) -> Optional[CalcResult]:
    p = _prepare(rec)
    if not p.num:
        return None
    ops = detect_ops(question)
    cols = _match_columns(question, p)
    rows_m = _match_rows(question, p)
    base = dict(table_id=rec.table_id, filename=rec.filename, page=rec.page)
    where = f"table {rec.table_id} (page {rec.page})" if rec.page else f"table {rec.table_id}"
    md_all = rec.to_markdown()

    if not ops and not rows_m.any():
        return None
    if not ops and rows_m.any() and not cols:
        sub = p.df[rows_m]
        return CalcResult(**base, operation="lookup", text=f"Row(s) matching the question in {where}:\n" + rec.to_markdown(sub.values.tolist()))

    if not cols and ops != ["count"]:
        nums = list(p.num)
        if len(nums) == 1:
            cols = nums
        else:
            return CalcResult(**base, operation="clarify", needs_clarification=True,
                              text=f"Numeric columns in {where}: {', '.join(nums)}. The question does not say which one to use.")
    if not ops:  # row + column lookup
        out = []
        for i in p.df.index[rows_m]:
            for c in cols:
                out.append(f"{_row_label(p, i)} / {c}: {p.df.at[i, c]}")
        return CalcResult(**base, operation="lookup", text=f"Lookup in {where}:\n" + "\n".join(out))

    use = rows_m & ~p.total_mask if rows_m.any() else ~p.total_mask  # never double count 'Total' rows
    scope = f"rows matching the question" if rows_m.any() else "all data rows (excluding any 'Total' row)"
    lines: list[str] = []
    for op in ops:
        if op == "count":
            lines.append(f"Count of {scope} in {where}: {int(use.sum())}")
            continue
        num_cols = [c for c in cols if c in p.num]
        if not num_cols:
            return CalcResult(**base, operation="clarify", needs_clarification=True,
                              text=f"Column(s) {cols} in {where} are not numeric; cannot compute '{op}'.")
        if op == "compare":
            if len(num_cols) < 2:
                continue
            a, b = num_cols[0], num_cols[1]
            sub = p.df[use]
            diff = (p.num[b][use] - p.num[a][use])
            tbl = ["| " + " | ".join([(p.label_cols[0] if p.label_cols else "row"), a, b, f"{b} - {a}"]) + " |",
                   "| --- | --- | --- | --- |"]
            for i in sub.index:
                tbl.append(f"| {_row_label(p, i)} | {p.df.at[i, a]} | {p.df.at[i, b]} | {fmt(diff[i], p.units.get(b, ''))} |")
            sa, sb = p.num[a][use].sum(), p.num[b][use].sum()
            lines.append(f"Comparison of '{a}' vs '{b}' in {where} over {scope}:\n" + "\n".join(tbl) +
                         f"\nSum of {a} = {fmt(sa, p.units.get(a, ''))}; sum of {b} = {fmt(sb, p.units.get(b, ''))}; "
                         f"difference ({b} - {a}) = {fmt(sb - sa, p.units.get(b, ''))}.")
            continue
        for c in num_cols:
            s, u = p.num[c][use].dropna(), p.units.get(c, "")
            if s.empty:
                lines.append(f"No numeric values in column '{c}' for {scope}.")
            elif op == "sum":
                lines.append(f"Sum of '{c}' over {scope} in {where} = {fmt(s.sum(), u)} ({len(s)} values)")
            elif op == "mean":
                lines.append(f"Average of '{c}' over {scope} in {where} = {fmt(s.mean(), u)} ({len(s)} values)")
            elif op in ("max", "min"):
                idx = s.idxmax() if op == "max" else s.idxmin()
                ties = s[s == s.loc[idx]].index.tolist()
                who = ", ".join(_row_label(p, i) for i in ties)
                full = rec.to_markdown([p.df.loc[i].tolist() for i in ties])
                lines.append(f"{'Highest' if op == 'max' else 'Lowest'} '{c}' in {where} = {fmt(s.loc[idx], u)} "
                             f"for: {who}\n{full}")
    for i in p.df.index[p.total_mask]:  # cross-check against the table's own total row
        own = [f"{c}={p.df.at[i, c]}" for c in p.num if str(p.df.at[i, c]).strip()]
        if own and any(o in ops for o in ("sum",)):
            lines.append(f"(The table's own '{_row_label(p, i)}' row states: {', '.join(own)})")
    if not lines:
        return None
    return CalcResult(**base, operation="+".join(ops), text="\n".join(lines), table_markdown=md_all)


def answer_from_tables(question: str, tables: list[TableRecord]) -> list[CalcResult]:
    results = [r for r in (calculate(question, t) for t in tables) if r]
    good = [r for r in results if not r.needs_clarification]
    return good if good else results[:1]
