"""Laya rubric checklist (Amrita's method; approved idea #4, 2026-10-14): per-clip features from the kaggle/laya_checklist job.

Per clip, for each of the 8 yes/no questions asked of every sentence unit: the mean probability, the share of units above 0.5
and the max (rates, so robust to clip length); plus the verbatim rubric score's expected level and entropy, on the CTC and the
Whisper text. No labels are used to build them. Then, on the 732 labelled train clips (fixed before seeing the data):
  - the Phase B selection checks (sign, monotone by level, |Spearman with duration| < 0.2 per batch, |partial Spearman vs
    repair_wer| >= 0.05), with the expected signs below;
  - the rubric conformance tests per batch (one-sided Mann-Whitney, rank-biserial r) on the mean aggregates;
  - each feature's Spearman with deberta_pair_pred (redundancy) and with the label;
  - the CTC-failure check: on clips whose repair rate > 0.6, is the "same message?" probability lower than elsewhere?

    python src/laya_checklist.py     # cache/laya/laya_checklist_feats.csv, reports/phaseH/laya_*.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bootstrap  # noqa: E402,F401  — switches to .venv's python if needed
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from rubric_select import LEVELS, partial_spearman, sp

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
OUT = ROOT / "reports" / "phaseH"
QUESTIONS = {"error": -1, "basic_error": -1, "incomplete": -1, "complex_ok": +1, "formulaic": -1, "selfcorrect": +1,
             "same_meaning": 0, "grammar_only": 0}                  # 0 = the data decides the sign
AGGS = ("mean", "share", "max")
DUR_MAX, PARTIAL_MIN, MAX_INVERSIONS, ALPHA, CTC_FAIL_CUT = 0.2, 0.05, 1, 0.05, 0.6
# rubric tests on the mean aggregates: (descriptor, feature, group expected higher, the other group)
TESTS = [
    ("L3 vs L4 errors", "chk_error_mean", lambda v: (v >= 3.0) & (v <= 3.5), lambda v: (v >= 4.0) & (v <= 4.5)),
    ("L2 basic mistakes", "chk_basic_error_mean", lambda v: v <= 2.5, lambda v: v >= 3.0),
    ("L2 incomplete sentences", "chk_incomplete_mean", lambda v: v <= 2.5, lambda v: v >= 3.0),
    ("L5 complex structures", "chk_complex_ok_mean", lambda v: v == 5.0, lambda v: v == 4.0),
    ("L1-2 memorised patterns", "chk_formulaic_mean", lambda v: v <= 2.5, lambda v: v >= 3.0),
    ("L4-5 self-correction", "chk_selfcorrect_mean", lambda v: v >= 4.0, lambda v: v <= 3.0),
    ("overall level (CTC text)", "rub_ctc_ev", lambda v: v >= 4.0, lambda v: v <= 3.0),
    ("overall level (Whisper text)", "rub_whisper_ev", lambda v: v >= 4.0, lambda v: v <= 3.0),
]


def features() -> pd.DataFrame:
    U = pd.read_csv(CACHE / "laya" / "laya_units.csv")
    g = U.groupby(["split", "filename"])
    cols = {}
    for q in QUESTIONS:
        p = f"p_{q}"
        cols[f"chk_{q}_mean"] = g[p].mean()
        cols[f"chk_{q}_share"] = g[p].apply(lambda s: float((s > 0.5).mean()))
        cols[f"chk_{q}_max"] = g[p].max()
    F = pd.DataFrame(cols)
    S = pd.read_csv(CACHE / "laya" / "laya_rubric_score.csv").set_index(["split", "filename"])
    for text in ("ctc", "whisper"):
        F[f"rub_{text}_ev"] = S[f"{text}_ev"]
        F[f"rub_{text}_entropy"] = S[f"{text}_entropy"]
    F = F.reset_index()
    F.to_csv(CACHE / "laya" / "laya_checklist_feats.csv", index=False)
    return F


def expected_sign(col: str) -> int:
    if col.startswith("rub_"):
        return +1 if col.endswith("_ev") else 0
    q = col[len("chk_"):].rsplit("_", 1)[0]
    return QUESTIONS[q]


def selection(T: pd.DataFrame, cols) -> pd.DataFrame:
    y, rw, dur, short = T.label.values, T.repair_wer_nofiller.values, T.duration.values, (T.batch == "short").values
    rows = []
    for f in cols:
        x, exp = T[f].values.astype(float), expected_sign(f)
        rho = sp(x, y)
        direction = exp if exp else np.sign(rho)
        means = [np.nanmean(x[y == lv]) for lv in LEVELS]
        inv = int(sum(np.sign(b - a) == -direction for a, b in zip(means, means[1:])))
        d_s, d_l = sp(x[short], dur[short]), sp(x[~short], dur[~short])
        part = partial_spearman(x, y.astype(float), rw)
        c1 = True if exp == 0 else bool(np.sign(rho) == exp)
        c2, c3, c4 = inv <= MAX_INVERSIONS, max(abs(d_s), abs(d_l)) < DUR_MAX, abs(part) >= PARTIAL_MIN
        rows.append({"feature": f, "expected": {1: "+", -1: "-", 0: "data"}[exp], "spearman": rho, "1_sign": c1,
                     "inversions": inv, "2_monotone": c2, "dur_rho_short": d_s, "dur_rho_long": d_l, "3_length": c3,
                     "partial_vs_repair": part, "4_partial": c4, "passes": c1 and c2 and c3 and c4,
                     "rho_deberta_pair": sp(x, T.deberta_pair_pred.values.astype(float))})
    return pd.DataFrame(rows)


def rubric_tests(T: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for batch in ("short", "long"):
        d = T[T.batch == batch]
        yb = d.label.values
        for desc, feat, ga, gb in TESTS:
            a, b = d.loc[ga(yb), feat].dropna().values, d.loc[gb(yb), feat].dropna().values
            if len(a) < 3 or len(b) < 3:
                p, r = np.nan, np.nan
            else:
                u, p = mannwhitneyu(a, b, alternative="greater")
                r = 2 * u / (len(a) * len(b)) - 1
            rows.append({"batch": batch, "descriptor": desc, "test": feat, "n_a": len(a), "n_b": len(b), "p": p, "effect_r": r,
                         "pass": bool(p < ALPHA) if p == p else None})
    return pd.DataFrame(rows)


def main():
    F = features()
    tr = pd.read_csv(CACHE / "train.csv")
    G = pd.read_csv(CACHE / "speaker_groups.csv")
    Fe = pd.read_csv(CACHE / "features.csv")
    D = pd.read_csv(CACHE / "deberta_grouped.csv")
    T = (F[F.split == "train"].merge(tr[tr.label > 0], on="filename")
         .merge(Fe[Fe.split == "train"][["filename", "duration", "repair_wer_nofiller"]], on="filename")
         .merge(G[G.split == "train"][["filename", "batch"]], on="filename")
         .merge(D[D.split == "train"][["filename", "deberta_pair_pred"]], on="filename"))
    assert len(T) == 732, len(T)
    cols = [c for c in F.columns if c.startswith(("chk_", "rub_"))]
    sel = selection(T, cols)
    tests = rubric_tests(T)
    fail = T.repair_wer_nofiller > CTC_FAIL_CUT
    ctc = pd.DataFrame({"group": ["CTC failure (repair rate > 0.6)", "other clips"], "n": [int(fail.sum()), int((~fail).sum())],
                        "same_meaning_mean": [T.loc[fail, "chk_same_meaning_mean"].mean(), T.loc[~fail, "chk_same_meaning_mean"].mean()],
                        "grammar_only_mean": [T.loc[fail, "chk_grammar_only_mean"].mean(), T.loc[~fail, "chk_grammar_only_mean"].mean()]})
    OUT.mkdir(parents=True, exist_ok=True)
    sel.to_csv(OUT / "laya_selection.csv", index=False)
    tests.to_csv(OUT / "laya_rubric_tests.csv", index=False)
    ctc.to_csv(OUT / "laya_ctc_failure.csv", index=False)
    with pd.option_context("display.width", 230, "display.max_rows", 100):
        print(sel.round(3).to_string(index=False))
        print("\npassing all four checks:", sel[sel.passes].feature.tolist())
        print("\n", tests.pivot_table(index=["descriptor", "test"], columns="batch", values=["pass", "effect_r"],
                                      aggfunc="first").round(3).to_string())
        print("\n", ctc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
