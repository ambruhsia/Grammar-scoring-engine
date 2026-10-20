"""Rubric conformance tests (PLAN_RUBRIC G3), run separately for short and long clips. Diagnostics and report material only:
nothing is tuned to make a test pass.

Each descriptor: a one-sided Mann-Whitney U test (scipy) between two label groups, plus the rank-biserial effect size
r = 2U / (n1 n2) - 1, signed so that r > 0 supports the rubric. Pass = p < ALPHA.
The model test: the mean grouped OOF prediction must rise across every adjacent label-level pair (2.0 ... 5.0).

    python src/rubric_tests.py        # reads cache/insight.csv (src/insight.py), writes reports/phaseG/rubric_tests.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.05
LEVELS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
# (descriptor, feature, group A (expected higher), group B, rubric wording)
TESTS = [
    ("L2 incomplete sentences", "fragment_ratio", lambda y: y <= 2.5, lambda y: y >= 3.0, "might leave sentences incomplete"),
    ("L2 incomplete sentences", "no_punct", lambda y: y <= 2.5, lambda y: y >= 3.0, "might leave sentences incomplete"),
    ("L1-2 memorized / simple", "trigram_repeat_rate", lambda y: y <= 2.5, lambda y: y >= 3.0, "memorized patterns, simple structures"),
    ("L1-2 memorized / simple", "mattr50", lambda y: y >= 3.0, lambda y: y <= 2.5, "memorized patterns, simple structures (lower range at <=2.5)"),
    ("L3 vs L4 error density", "lt_grammar_per100", lambda y: (y >= 3.0) & (y <= 3.5), lambda y: (y >= 4.0) & (y <= 4.5), "errors (3) vs occasional minor errors (4)"),
    ("L3 vs L4 error density", "llm_err_per100", lambda y: (y >= 3.0) & (y <= 3.5), lambda y: (y >= 4.0) & (y <= 4.5), "errors (3) vs occasional minor errors (4)"),
    ("L5 complex structures", "dep_clause_ratio", lambda y: y == 5.0, lambda y: y == 4.0, "complex grammar, complex language structures"),
    ("L5 complex structures", "clauses_per_sentence", lambda y: y == 5.0, lambda y: y == 4.0, "complex grammar, complex language structures"),
    ("L4-5 self-correction", "corrected_error_ratio", lambda y: y >= 4.0, lambda y: y <= 3.0, "can correct most of them / correct themselves"),
]


def mw(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan, len(a), len(b)
    u, p = mannwhitneyu(a, b, alternative="greater")
    return float(p), float(2 * u / (len(a) * len(b)) - 1), len(a), len(b)


def run(T: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for batch in ("short", "long"):
        d = T[T.batch == batch]
        y = d.label.values
        for desc, feat, ga, gb, words in TESTS:
            p, r, na, nb = mw(d.loc[ga(y), feat], d.loc[gb(y), feat])
            rows.append({"batch": batch, "descriptor": desc, "test": feat, "rubric": words, "n_a": na, "n_b": nb,
                         "p": p, "effect_r": r, "pass": bool(p < ALPHA) if not np.isnan(p) else None})
        means = [d.loc[d.label == lv, "pred"].mean() for lv in LEVELS]
        bad = [f"{a:g}->{b:g}" for a, b, ma, mb in zip(LEVELS, LEVELS[1:], means, means[1:]) if not mb > ma]
        rows.append({"batch": batch, "descriptor": "Model monotone in level", "test": "mean OOF prediction by level",
                     "rubric": "higher level -> higher predicted score", "n_a": len(d), "n_b": np.nan, "p": np.nan,
                     "effect_r": np.nan, "pass": not bad,
                     "detail": "means " + " ".join(f"{m:.2f}" for m in means) + (f"; not rising at {', '.join(bad)}" if bad else "")})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    D = pd.read_csv(ROOT / "cache" / "insight.csv")
    T = D[D.split == "train"]
    R = run(T)
    out = ROOT / "reports" / "phaseG"
    out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "rubric_tests.csv", index=False)
    piv = R.pivot_table(index=["descriptor", "test"], columns="batch", values=["pass", "effect_r", "p"], aggfunc="first")
    with pd.option_context("display.width", 220):
        print(piv[[("pass", "short"), ("pass", "long"), ("effect_r", "short"), ("effect_r", "long"), ("p", "short"), ("p", "long")]]
              .to_string(float_format=lambda v: f"{v:.3f}"))
        print()
        for r in R[R.descriptor == "Model monotone in level"].itertuples():
            print(f"model monotone [{r.batch}]: {'PASS' if r.pass_ else 'FAIL'}  {r.detail}" if hasattr(r, "pass_") else "")
        for _, r in R[R.descriptor == "Model monotone in level"].iterrows():
            print(f"model monotone [{r.batch}]: {'PASS' if r['pass'] else 'FAIL'}  {r['detail']}")
