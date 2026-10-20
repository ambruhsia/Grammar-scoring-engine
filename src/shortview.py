"""Short views of long clips (PLAN_RUBRIC G4): the text side of a long clip cut to a test-like short duration.

For each long training clip, a cut time is drawn (seeded) from the durations of the test short batch. Whisper segments that
end by the cut are kept (their words too); CTC words that end by the last kept segment are kept. The notebook's section-4
features are then recomputed on the cut, with the notebook's own code (ported below): duration, the pause features, the
text features, repair_wer_nofiller and no_punct. GPT-2 / CoLA features and the learned *_pred columns can't be recomputed
here (they need the GPU or the audio); a view keeps its clip's values for those, and that is stated wherever views are used.

    python src/shortview.py        # checks the port reproduces cache/features.csv on the uncut transcripts
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 0

# ---- ported from the notebook's section 4 (feature build) ---------------------------------------------
MIN_PAUSE_S, PAUSE_CAP_S, BOUNDARY_TOL_S, THOUGHT_GRACE = 0.25, 3.00, 0.20, 1
SUBORDINATORS = {"because", "although", "though", "while", "whereas", "since", "unless", "until", "whether", "if", "when",
                 "after", "before", "that", "which", "who", "whom", "whose"}
CONNECTIVES = {"however", "therefore", "moreover", "furthermore", "nevertheless", "consequently", "meanwhile", "otherwise",
               "instead", "thus"}
MODALS = {"would", "could", "should", "might", "must", "may", "shall", "ought"}
NON_LEXICAL = {"uh", "uhh", "um", "umm", "erm", "er", "ah", "ahh", "eh", "mm", "hmm", "hm", "uhm"}
_W = re.compile(r"[a-z']+")
norm = lambda t: _W.findall((t or "").lower())


def lev(a, b):
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def pause_feats(words, bounds, dur):
    out = dict(midclause_pause_ratio=0.0, midclause_pause_per_min=0.0, boundary_share=1.0, articulation_rate=0.0,
               phonation_ratio=0.0)
    if len(words) < 3 or dur <= 0:
        return out
    st = np.array([w[1] for w in words], float)
    en = np.array([w[2] for w in words], float)
    span = max(en[-1] - st[0], 1e-6)
    phon = float(np.clip(en - st, 0, None).sum())
    B = np.asarray(bounds, float)
    natural, mid = [], []
    for g, t in zip(st[1:] - en[:-1], en[:-1]):
        if g < MIN_PAUSE_S:
            continue
        at_b = B.size and np.min(np.abs(B - t)) <= BOUNDARY_TOL_S
        (natural if at_b else mid).append(min(float(g), PAUSE_CAP_S))
    mid.sort(reverse=True)
    counted = mid[THOUGHT_GRACE:]
    n_all = len(mid) + len(natural)
    out["midclause_pause_ratio"] = sum(counted) / span
    out["midclause_pause_per_min"] = len(counted) / (span / 60.0)
    out["boundary_share"] = len(natural) / n_all if n_all else 1.0
    out["articulation_rate"] = len(words) / max(phon, 1e-6)
    out["phonation_ratio"] = phon / span
    return out


def text_feats(clean):
    toks = norm(clean)
    content = [t for t in toks if t not in NON_LEXICAL]
    n = max(len(toks), 1)
    sents = [s for s in re.split(r"[.!?]+", clean or "") if s.strip()]
    rep1 = sum(1 for a, b in zip(toks, toks[1:]) if a == b and len(a) > 1)
    rep2 = sum(1 for i in range(len(toks) - 3) if toks[i:i + 2] == toks[i + 2:i + 4])
    return {"n_words": float(len(toks)), "text_ttr": len(set(content)) / max(len(content), 1),
            "mean_sent_len": len(toks) / max(len(sents), 1),
            "std_sent_len": float(np.std([len(norm(s)) for s in sents])) if sents else 0.0,
            "subordinator_rate": sum(t in SUBORDINATORS for t in toks) / n,
            "connective_rate": sum(t in CONNECTIVES for t in toks) / n,
            "modal_rate": sum(t in MODALS for t in toks) / n, "repetition_rate": (rep1 + rep2) / n,
            "mean_word_len": float(np.mean([len(t) for t in content])) if content else 0.0,
            "long_word_rate": sum(len(t) >= 8 for t in content) / max(len(content), 1)}


def section4(w_text, w_words, segments, ctc_text, duration):
    bounds = [float(s["end"]) for s in segments if s["text"].strip().endswith((".", "!", "?"))]
    f = {"duration": float(duration), **pause_feats(w_words, bounds, float(duration)), **text_feats(w_text)}
    wt = [t for t in norm(w_text) if t not in NON_LEXICAL]
    ct = [t for t in norm(ctc_text) if t not in NON_LEXICAL]
    f["repair_wer_nofiller"] = min(lev(ct, wt) / max(len(ct), 1), 2.0)
    f["no_punct"] = float(not any(s["text"].strip().endswith((".", "!", "?")) for s in segments))
    return f


RECOMPUTED = ["duration", "midclause_pause_ratio", "midclause_pause_per_min", "boundary_share", "articulation_rate",
              "phonation_ratio", "n_words", "text_ttr", "mean_sent_len", "std_sent_len", "subordinator_rate",
              "connective_rate", "modal_rate", "repetition_rate", "mean_word_len", "long_word_rate",
              "repair_wer_nofiller", "no_punct"]


# ---- the cut ------------------------------------------------------------------------------------------
def load_jsonl(p: Path) -> dict:
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[(r["split"], r["filename"])] = r
    return out


def cut(w, c, cut_s):
    """-> (whisper text, whisper words, segments, ctc text) of the first cut_s seconds."""
    segs = [s for s in w["segments"] if float(s["end"]) <= cut_s] or w["segments"][:1]
    t_end = max(float(s["end"]) for s in segs)
    words = [x for x in w["words"] if float(x[2]) <= t_end + 1e-6]
    ctc = " ".join(x[0] for x in c["words"] if float(x[2]) <= t_end + 1e-6)
    return "".join(s["text"] for s in segs).strip(), words, segs, ctc


def build_views(ctx) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """-> (views: one row per long training clip with RECOMPUTED columns, source row index in ctx.T, cut seconds)."""
    W, C = load_jsonl(ctx.cache / "asr_c.jsonl"), load_jsonl(ctx.cache / "ctc.jsonl")
    G = pd.read_csv(ctx.cache / "speaker_groups.csv")
    test_short = G[(G.split == "test") & (G.batch == "short")].duration.values
    rng = np.random.default_rng(SEED)
    src = np.where(ctx.batch == "long")[0]
    cuts = rng.choice(test_short, size=len(src), replace=True)
    rows = []
    for i, cs in zip(src, cuts):
        k = ("train", ctx.T.filename.iloc[i])
        wt, ww, segs, ct = cut(W[k], C[k], float(cs))
        rows.append(section4(wt, ww, segs, ct, cs))
    return pd.DataFrame(rows), src, cuts


if __name__ == "__main__":
    W, C = load_jsonl(ROOT / "cache" / "asr_c.jsonl"), load_jsonl(ROOT / "cache" / "ctc.jsonl")
    F = pd.read_csv(ROOT / "cache" / "features.csv")
    rows = [section4(W[k]["text"], W[k]["words"], W[k]["segments"], C[k]["raw_text"], W[k]["duration"])
            for k in zip(F.split, F.filename)]
    R = pd.DataFrame(rows)
    for c in RECOMPUTED:
        print(f"{c:<26} max|diff| vs features.csv {np.nanmax(np.abs(R[c].values - F[c].values)):.2e}")
