"""Rubric-aligned features (PLAN_RUBRIC Phase B), from the cached transcripts. -> cache/rubric_feats.csv

Sources: cache/asr_c.jsonl (Whisper: clean text with punctuation, segment times) for STRUCTURE, and
cache/ctc.jsonl (CTC: raw lowercase text with word times, keeps the speaker's errors) for ACCURACY.
Rules: every feature is a rate (test clips are shorter), fillers are removed before anything is
counted (the rubric never mentions them), and self-correction is measured, not penalised.

Whisper side, spaCy en_core_web_sm (MIT), one doc per Whisper sentence (Whisper's own segmentation:
text split at . ! ?). "Finite" = tag VBD / VBP / VBZ / MD.
    finite_clause_ratio   sentences with >= 1 finite verb / sentences                      (+)
    mean_len_clause       words / finite clauses                                           (+)
    dep_clause_ratio      (advcl + ccomp + csubj + relcl + finite acl) / finite clauses     (+)
    clauses_per_sentence  finite clauses / sentences                                       (+)
    mean_dep_distance     mean |head - token| over non-punctuation, non-root tokens        (+)
    mattr50               moving-average type-token ratio, window 50 words                 (+)
    trigram_repeat_rate   share of word trigrams that occur more than once                 (-)
    fragment_ratio        sentences whose root is not VERB/AUX and that have no finite verb (-)
CTC side:
    lt_grammar_per100     LanguageTool 6.5 (LGPL, local server) GRAMMAR-category matches per 100 words,
                          on the CTC words cut into chunks by Whisper segment times (TYPOS, CASING,
                          PUNCTUATION and STYLE disabled)                                  (-)
    self_repair_per100    retraces with change (a word repeated 2-3 tokens later with a different
                          continuation) + partial-word restarts (a token of >= 2 characters that is a
                          prefix of the next token), per 100 words                         (data decides)

    python src/rubric_feats.py          # builds the cache if inputs or this version changed
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
OUT = CACHE / "rubric_feats.csv"
VERSION = "rubric-v1"                      # bump when a definition changes
LT_VERSION = "6.5"                         # 6.6+ needs Java 17; this machine has Java 11
NON_LEXICAL = {"uh", "uhh", "um", "umm", "erm", "er", "ah", "ahh", "eh", "mm", "hmm", "hm", "uhm"}
FINITE = {"VBD", "VBP", "VBZ", "MD"}
SUB_DEPS = {"advcl", "ccomp", "csubj", "relcl"}
EXPECTED = {"finite_clause_ratio": +1, "mean_len_clause": +1, "dep_clause_ratio": +1,
            "clauses_per_sentence": +1, "mean_dep_distance": +1, "mattr50": +1,
            "trigram_repeat_rate": -1, "fragment_ratio": -1, "lt_grammar_per100": -1,
            "self_repair_per100": 0}
NEW = list(EXPECTED)
_W = re.compile(r"[a-z']+")


def words(text: str) -> list[str]:
    return [t for t in _W.findall((text or "").lower()) if t not in NON_LEXICAL]


def load(p: Path) -> dict:
    d = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            d[(r["split"], r["filename"])] = r
        except json.JSONDecodeError:
            pass
    return d


# ---- Whisper side ---------------------------------------------------------------------------------
def is_finite(tok) -> bool:
    return tok.tag_ in FINITE or any(c.dep_ in ("aux", "auxpass") and c.tag_ in FINITE for c in tok.children)


def mattr(ws: list[str], win: int = 50) -> float:
    if not ws:
        return np.nan
    if len(ws) <= win:
        return len(set(ws)) / len(ws)
    c = Counter(ws[:win])
    vals = [len(c)]
    for i in range(win, len(ws)):
        c[ws[i]] += 1
        c[ws[i - win]] -= 1
        if c[ws[i - win]] == 0:
            del c[ws[i - win]]
        vals.append(len(c))
    return float(np.mean(vals) / win)


def trigram_repeat(ws: list[str]) -> float:
    tg = list(zip(ws, ws[1:], ws[2:]))
    if not tg:
        return np.nan
    c = Counter(tg)
    return sum(c[t] > 1 for t in tg) / len(tg)


def structure(text: str, nlp) -> dict:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if words(s)]
    ws = words(text)
    out = {"mattr50": mattr(ws), "trigram_repeat_rate": trigram_repeat(ws)}
    if not sents:
        return {**out, **{k: np.nan for k in ("finite_clause_ratio", "mean_len_clause", "dep_clause_ratio",
                                               "clauses_per_sentence", "mean_dep_distance", "fragment_ratio")}}
    n_fin = n_sub = n_sent_fin = n_frag = 0
    dists = []
    for doc in nlp.pipe(sents):
        toks = [t for t in doc if not t.is_punct and not t.is_space and t.lower_ not in NON_LEXICAL]
        fin = sum(t.tag_ in FINITE for t in toks)
        n_fin += fin
        n_sent_fin += fin > 0
        n_sub += sum(t.dep_ in SUB_DEPS or (t.dep_ == "acl" and is_finite(t)) for t in toks)
        roots = [t for t in doc if t.dep_ == "ROOT"]
        if fin == 0 and not any(r.pos_ in ("VERB", "AUX") for r in roots):
            n_frag += 1
        dists += [abs(t.head.i - t.i) for t in toks if t.dep_ != "ROOT"]
    n = len(sents)
    return {**out,
            "finite_clause_ratio": n_sent_fin / n,
            "mean_len_clause": len(ws) / n_fin if n_fin else np.nan,
            "dep_clause_ratio": n_sub / n_fin if n_fin else np.nan,
            "clauses_per_sentence": n_fin / n,
            "mean_dep_distance": float(np.mean(dists)) if dists else np.nan,
            "fragment_ratio": n_frag / n}


# ---- CTC side --------------------------------------------------------------------------------------
def ctc_chunks(ctc_words: list, segments: list) -> list[list[str]]:
    """CTC words (fillers removed) grouped by the Whisper segment their midpoint falls in (nearest if none)."""
    segs = [(float(s["start"]), float(s["end"])) for s in segments] or [(0.0, 1e9)]
    mids = np.array([(a + b) / 2 for a, b in segs])
    chunks = [[] for _ in segs]
    for w, st, en, *_ in ctc_words:
        w = (w or "").lower()
        if not _W.fullmatch(w) or w in NON_LEXICAL:
            continue
        m = (float(st) + float(en)) / 2
        inside = [i for i, (a, b) in enumerate(segs) if a <= m <= b]
        chunks[inside[0] if inside else int(np.argmin(np.abs(mids - m)))].append(w)
    return [c for c in chunks if c]


def self_repairs(ws: list[str]) -> int:
    n = 0
    for i, t in enumerate(ws):
        for j in (i + 2, i + 3):                       # j = i+1 is plain repetition, not a correction
            if j < len(ws) and ws[j] == t and j + 1 < len(ws) and ws[j + 1] != ws[i + 1]:
                n += 1
                break
        if len(t) >= 2 and i + 1 < len(ws) and ws[i + 1] != t and ws[i + 1].startswith(t):
            n += 1
    return n


def grammar_hits(tool, chunks: list[list[str]]) -> int:
    text = " ".join(" ".join(c).capitalize() + "." for c in chunks)
    return sum(m.category == "GRAMMAR" for m in tool.check(text)) if text else 0


# ---- build -----------------------------------------------------------------------------------------
def signature() -> str:
    h = hashlib.sha256(VERSION.encode() + LT_VERSION.encode())
    for p in (CACHE / "asr_c.jsonl", CACHE / "ctc.jsonl"):
        h.update(f"{p.name}:{p.stat().st_size}".encode())
        h.update(p.read_bytes()[:4096])
    return h.hexdigest()[:16]


def build() -> pd.DataFrame:
    import language_tool_python as ltp
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    tool = ltp.LanguageTool("en-US", language_tool_download_version=LT_VERSION)   # local server only
    tool.disabled_categories = {"TYPOS", "CASING", "PUNCTUATION", "STYLE"}
    W, C = load(CACHE / "asr_c.jsonl"), load(CACHE / "ctc.jsonl")
    rows = []
    try:
        for i, k in enumerate(sorted(set(W) & set(C)), 1):
            chunks = ctc_chunks(C[k]["words"], W[k]["segments"])
            cw = [w for c in chunks for w in c]
            n = max(len(cw), 1)
            rows.append({"split": k[0], "filename": k[1], **structure(W[k]["text"], nlp),
                         "lt_grammar_per100": 100 * grammar_hits(tool, chunks) / n,
                         "self_repair_per100": 100 * self_repairs(cw) / n})
            if i % 200 == 0:
                print(f"  rubric features {i}/{len(W)}", flush=True)
    finally:
        tool.close()
    return pd.DataFrame(rows)


def features(recompute: bool = False) -> pd.DataFrame:
    """-> DataFrame with split, filename and the NEW columns; rebuilt when inputs or VERSION change."""
    sig, meta = signature(), OUT.with_suffix(".sig")
    if recompute or not OUT.exists() or not meta.exists() or meta.read_text().strip() != sig:
        build().to_csv(OUT, index=False)
        meta.write_text(sig)
    return pd.read_csv(OUT)


if __name__ == "__main__":
    df = features()
    print(df[NEW].describe().T.round(4).to_string())
