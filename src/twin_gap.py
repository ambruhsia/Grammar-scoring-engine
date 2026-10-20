"""How much each candidate column rides on speaker twins (PLAN_RUBRIC A5 table).

Families: seed-averaged OOF RMSE from a random-fold run vs a grouped-fold run (their board.csv files).
Learned columns: the *_pred columns are fixed OOF from the notebook's random seed-42 folds, so their RMSE
is the same under any local folds. The ones that can be rebuilt on CPU are rebuilt with the notebook's own
protocol, first on its random folds (must reproduce the cached column), then on the grouped folds:
  wavlm_pred / wavlm_large_pred / whisper_med_pred  per-layer z-score, uniform layer mean, ridge
  spk_knn_pred                                      kernel kNN on speaker x-vectors (K 10, tau 0.05)
DeBERTa, fine-tuned WavLM and emb_pred need the GPU or uncached embeddings and are marked as such.

    python src/twin_gap.py <random_run_id> <grouped_run_id>   # writes reports/phaseA/twin_gap.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CLIP, FULL_SEEDS, RUNS, rmse, wrmse  # noqa: E402
from data import load  # noqa: E402

FROZEN = {"wavlm_pred": "wavlm_layers.npz", "wavlm_large_pred": "ssl_wavlm_large.npz",
          "whisper_med_pred": "ssl_whisper_med.npz"}
K_NN, TAU = 10, 0.05
NOT_LOCAL = {"ft_wavlm_pred": "GPU (fine-tune)", "deberta_pred": "GPU", "deberta_ctc_pred": "GPU",
             "deberta_pair_pred": "GPU", "emb_pred": "embeddings not cached"}


def pooled(ctx, file):
    z = np.load(ctx.cache / file)
    idx = {k: i for i, k in enumerate(zip(z["split"], z["filename"]))}
    H = z["H"]
    tr = [idx[("train", f)] for f in ctx.T.filename]
    mu, sd = H[tr].mean(0, keepdims=True), H[tr].std(0, keepdims=True) + 1e-6
    return ((H[tr] - mu) / sd).mean(1)


def ridge_oof(X, y, folds):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 6, 31)))
    o = np.zeros(len(y))
    for a, b in folds:
        o[b] = clone(m).fit(X[a], y[a]).predict(X[b])
    return o


def knn_oof(E, y, folds):
    o = np.zeros(len(y))
    for a, b in folds:
        S = E[b] @ E[a].T
        idx = np.argpartition(-S, K_NN - 1, axis=1)[:, :K_NN]
        s = np.take_along_axis(S, idx, 1)
        w = np.exp((s - s.max(1, keepdims=True)) / TAU)
        o[b] = (w * y[a][idx]).sum(1) / w.sum(1)
    return o


def main(random_run, grouped_run):
    rnd, grp = load((42,), folds="random"), load(FULL_SEEDS, folds="grouped")
    y, w = grp.y, grp.w
    rows = []

    br = pd.read_csv(RUNS / random_run / "board.csv").set_index("candidate")
    bg = pd.read_csv(RUNS / grouped_run / "board.csv").set_index("candidate")
    for c in bg.index:
        if c.endswith("_pred"):
            continue
        rows.append({"column": c, "kind": "family", "random": br.loc[c, "oof_rmse"], "grouped": bg.loc[c, "oof_rmse"],
                     "grouped_testmix": bg.loc[c, "oof_testmix"], "note": "local fit; uses *_pred as features"})

    def add(col, oof_r, oof_g_list, note):
        g = np.mean(oof_g_list, 0)
        rows.append({"column": col, "kind": "rebuilt", "random": rmse(y, np.clip(oof_r, *CLIP)),
                     "grouped": rmse(y, np.clip(g, *CLIP)), "grouped_testmix": wrmse(y, np.clip(g, *CLIP), w),
                     "note": note})

    for col, file in FROZEN.items():
        X = pooled(grp, file)
        o_r = ridge_oof(X, y, rnd.folds[42])
        repro = np.abs(o_r - grp.T[col].values).max()
        add(col, o_r, [ridge_oof(X, y, grp.folds[s]) for s in FULL_SEEDS], f"reproduces cached column: max|diff| {repro:.1e}")

    z = np.load(grp.cache / "spk_xvec.npz")
    idx = {k: i for i, k in enumerate(zip(z["split"], z["filename"]))}
    E = z["E"][[idx[("train", f)] for f in grp.T.filename]].astype(np.float64)
    o_r = knn_oof(E, y, rnd.folds[42])
    repro = np.abs(o_r - grp.T["spk_knn_pred"].values).max()
    add("spk_knn_pred", o_r, [knn_oof(E, y, grp.folds[s]) for s in FULL_SEEDS], f"reproduces cached column: max|diff| {repro:.1e}")

    for col, why in NOT_LOCAL.items():
        rows.append({"column": col, "kind": "fixed", "random": rmse(y, np.clip(grp.T[col], *CLIP)), "grouped": np.nan,
                     "grouped_testmix": np.nan, "note": f"cannot rebuild locally ({why}); grouped OOF unknown"})

    df = pd.DataFrame(rows)
    df["gap"] = df.grouped - df.random
    df = df.sort_values(["kind", "gap"], ascending=[True, False])
    out = Path(__file__).resolve().parents[1] / "reports" / "phaseA"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "twin_gap.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(df.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main(*sys.argv[1:3])
