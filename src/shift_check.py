"""G8: label-free shift check. Is "short clip => weaker speaker" still true of test's short clips? No model change.

Signals that never saw a label: llm_rubric_expected (zero-shot judge), llm_err_per100, mattr50, cola_frac_bad, compared for
train-short, train-long, test-short and test-long (standardised mean differences on the pooled train sd).
Batch-blind model: StandardScaler -> RidgeCV on the rubric layer (mattr50, trigram_repeat_rate, llm_err_per100, cola_frac_bad,
i.e. the features that passed the Phase B checks) plus llm_rubric_expected. No audio embeddings, no batch, no amount
features. OOF on the speaker-grouped folds (3 seeds); test = mean of the fold models.
The batch prior is SUSPECT if both hold: (A) the signals put test-short closer to train-long than to train-short, and
(B) the main model predicts test-short more than 0.1 below the batch-blind model on average.

    python src/shift_check.py      # needs cache/insight.csv (src/insight.py); writes reports/phaseG/shift_check.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CLIP, FULL_SEEDS, rmse, wrmse
from data import load

ROOT = Path(__file__).resolve().parents[1]
SIGNALS = ["llm_rubric_expected", "llm_err_per100", "mattr50", "cola_frac_bad"]
BLIND = ["mattr50", "trigram_repeat_rate", "llm_err_per100", "cola_frac_bad", "llm_rubric_expected"]
GAP = 0.1


def main():
    D = pd.read_csv(ROOT / "cache" / "insight.csv")
    D["group4"] = D.split + "-" + D.batch
    T = D[D.split == "train"]
    sd = T[SIGNALS].std()
    means = D.groupby("group4")[SIGNALS].mean()
    smd = lambda a, b: ((means.loc[a] - means.loc[b]) / sd).abs()
    to_short, to_long = smd("test-short", "train-short"), smd("test-short", "train-long")
    dist = pd.DataFrame({"test-short vs train-short": to_short, "test-short vs train-long": to_long})
    dist.loc["mean |SMD|"] = dist.mean()
    closer_to_long = bool(dist.loc["mean |SMD|", "test-short vs train-long"] < dist.loc["mean |SMD|", "test-short vs train-short"])

    ctx = load(FULL_SEEDS)
    tab = D.set_index(["split", "filename"])
    X = tab.loc[[("train", f) for f in ctx.T.filename], BLIND]
    Xt = tab.loc[[("test", f) for f in ctx.TE.filename], BLIND]
    med = X.median()
    X, Xt = X.fillna(med).values, Xt.fillna(med).values
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
    oofs, tests = [], []
    for s in FULL_SEEDS:
        o, t = np.zeros(len(ctx.y)), np.zeros(len(Xt))
        for a, b in ctx.folds[s]:
            m = clone(model).fit(X[a], ctx.y[a])
            o[b] = m.predict(X[b]); t += m.predict(Xt) / len(ctx.folds[s])
        oofs.append(np.clip(o, *CLIP)); tests.append(np.clip(t, *CLIP))
    blind_oof, blind_test = np.mean(oofs, 0), np.mean(tests, 0)
    short = ctx.batch == "short"
    print(f"batch-blind model OOF: grouped {rmse(ctx.y, blind_oof):.4f}  testmix {wrmse(ctx.y, blind_oof, ctx.w):.4f}  "
          f"short {rmse(ctx.y[short], blind_oof[short]):.4f}  long {rmse(ctx.y[~short], blind_oof[~short]):.4f}")

    bl = pd.concat([pd.DataFrame({"split": "train", "filename": ctx.T.filename, "blind": blind_oof}),
                    pd.DataFrame({"split": "test", "filename": ctx.TE.filename, "blind": blind_test})])
    D = D.merge(bl, on=["split", "filename"], how="left")
    cmp = D.groupby("group4").agg(n=("pred", "size"), true_label=("label", "mean"), main_model=("pred", "mean"),
                                  batch_blind=("blind", "mean")).round(3)
    cmp["main_minus_blind"] = (cmp.main_model - cmp.batch_blind).round(3)
    gap = float(cmp.loc["test-short", "main_minus_blind"])
    suspect = closer_to_long and gap < -GAP

    out = ROOT / "reports" / "phaseG"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat({"signal_means": means.T.round(3), "smd": dist.round(3)}, axis=1).to_csv(out / "shift_check_signals.csv")
    cmp.to_csv(out / "shift_check_models.csv")
    with pd.option_context("display.width", 200):
        print("\nlabel-free signal means:\n", means.round(3).to_string())
        print("\nstandardised distance of test-short (|mean diff| / train sd):\n", dist.round(3).to_string())
        print("\nmean prediction by group:\n", cmp.to_string())
    print(f"\n(A) test-short closer to train-long than to train-short: {closer_to_long}")
    print(f"(B) main model below batch-blind on test-short by more than {GAP}: {gap < -GAP} (main - blind = {gap:+.3f})")
    print(f"batch prior SUSPECT: {suspect}")


if __name__ == "__main__":
    main()
