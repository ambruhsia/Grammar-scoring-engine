"""Replay one experiment-loop run from its saved candidate matrices (the final notebook's stacking code, also runnable locally).

A run's cands.npz holds every member's out-of-fold predictions (one matrix per fold seed), its test predictions and its
topic-held-out predictions. From those, this recomputes exactly what src/run.py did after the members:
    1. blend weights per recording batch, non-negative least squares on topic-held-out predictions, test-mix weights;
       "shrink": the short-clip weights also see the long clips at SHRINK_LAMBDA x their weight. Refit inside every fold for the
       out-of-fold blend; fitted on all rows for the test blend.
    2. optional examiner correction: a ridge on [21 Qwen3.5-9B judge numbers, prediction] predicts the residual; refit inside every
       fold for the out-of-fold predictions, fitted on all rows for the test predictions; test-mix sample weights.
    3. optional same-speaker pooling: p' = (1 - s) p + s * mean of p over the clip's speaker group (train: the speaker groups;
       test: the same clustering of the test x-vectors).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import RidgeCV

CLIP = (1.5, 5.0)
SHRINK_LAMBDA = 0.25
JUDGE_VIEWS, JUDGE_PROMPTS = ("verbatim", "ctc", "whisper"), ("holistic", "accuracy")


def blend_weights(Cm, y, w, batch, shrink):
    """Per-batch NNLS on sqrt(weight)-scaled rows -> (short weights, long weights)."""
    sh, lo = np.where(batch == "short")[0], np.where(batch == "long")[0]
    s_lo = np.sqrt(w[lo])
    w_long = nnls(Cm[lo] * s_lo[:, None], y[lo] * s_lo)[0]
    R = np.r_[sh, lo] if shrink else sh
    ww = np.r_[w[sh], SHRINK_LAMBDA * w[lo]] if shrink else w[sh]
    s_r = np.sqrt(ww)
    return nnls(Cm[R] * s_r[:, None], y[R] * s_r)[0], w_long


def apply_blend(Cm, batch, wb):
    return np.where(batch == "short", Cm @ wb[0], Cm @ wb[1])


def judge_features(J1: pd.DataFrame, J2: pd.DataFrame, keys) -> np.ndarray:
    """The 21 examiner numbers of the correction: round 1 (expected grade + log-odds of grade >= 4 for 3 transcripts x 2 questions)
    and round 2 (expected grade for 9 question x transcript pairs). J1 / J2: the judge CSVs indexed by (split, filename)."""
    cols = {}
    for v in JUDGE_VIEWS:
        for p in JUDGE_PROMPTS:
            P = J1.loc[keys, [f"{v}_{p}_p{i}" for i in range(1, 6)]].values
            P = P / P.sum(1, keepdims=True)
            cols[f"judge35_{v}_{p}_ev"] = P @ np.arange(1, 6)
            cols[f"judge35_{v}_{p}_lo"] = np.log(P[:, 3:].sum(1) + 1e-9) - np.log(P[:, :3].sum(1) + 1e-9)
    for c in sorted({c.rsplit("_p", 1)[0] for c in J2.columns if "_p" in c and not c.endswith("_top")}):
        nd = 6 if c.endswith("cefr") else 5
        P = J2.loc[keys, [f"{c}_p{i}" for i in range(1, nd + 1)]].values
        cols[f"j2_{c}_ev"] = (P / P.sum(1, keepdims=True)) @ np.arange(1, nd + 1)
    return pd.DataFrame(cols).values


def ridge_correction(Z, e, sw):
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    m = RidgeCV(alphas=np.logspace(-1, 4, 30)).fit((Z - mu) / sd, e, sample_weight=sw)
    return lambda Zn: m.predict((Zn - mu) / sd)


def test_speaker_groups(E_train, E_test, pct=99):
    """Same-voice groups inside the test set: average-linkage clustering at the train speaker groups' cosine threshold."""
    A = E_train / (np.linalg.norm(E_train, axis=1, keepdims=True) + 1e-12)
    B = E_test / (np.linalg.norm(E_test, axis=1, keepdims=True) + 1e-12)
    t = float(np.percentile((B @ A.T).max(1), pct))
    return AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=1.0 - t).fit_predict(B)


def pool(p, groups, s):
    return (1 - s) * p + s * pd.Series(p).groupby(np.asarray(groups)).transform("mean").values


def replay(cands, folds, *, shrink=False, correction=None, pool_s=0.0, groups=None, groups_test=None):
    """-> (out-of-fold prediction (seed mean), test prediction, per-seed OOF list). folds: {seed: [(train idx, val idx), ...]}.
    correction: None or (X train, X test) extra features. groups / groups_test: speaker groups for pooling."""
    CO_s, CT_s, TO = cands["oof"], cands["test"], cands["topic"]
    y, w, bt, btt = cands["y"], cands["w"], cands["batch"], cands["batch_test"]
    seeds = [int(s) for s in cands["seeds"]]
    CT = np.mean(CT_s, 0)
    test = np.clip(apply_blend(CT, btt, blend_weights(TO, y, w, bt, shrink)), *CLIP)
    seed_oofs = []
    for i, s in enumerate(seeds):
        o = np.zeros(len(y))
        for a, b in folds[s]:
            o[b] = apply_blend(CO_s[i][b], bt[b], blend_weights(TO[a], y[a], w[a], bt[a], shrink))
        seed_oofs.append(np.clip(o, *CLIP))
    oof = np.clip(np.mean(seed_oofs, 0), *CLIP)
    if correction is not None:
        X, Xt = correction
        new = []
        for i, s in enumerate(seeds):
            o = seed_oofs[i]; oc = o.copy()
            for a, b in folds[s]:
                f = ridge_correction(np.c_[X[a], o[a]], y[a] - o[a], w[a])
                oc[b] = o[b] + f(np.c_[X[b], o[b]])
            new.append(np.clip(oc, *CLIP))
        f = ridge_correction(np.c_[X, oof], y - oof, w)
        test = np.clip(test + f(np.c_[Xt, test]), *CLIP)
        seed_oofs = new
        oof = np.clip(np.mean(seed_oofs, 0), *CLIP)
    if pool_s > 0:
        seed_oofs = [np.clip(pool(o, groups, pool_s), *CLIP) for o in seed_oofs]
        oof = np.clip(np.mean(seed_oofs, 0), *CLIP)
        test = np.clip(pool(test, groups_test, pool_s), *CLIP)
    return oof, test, seed_oofs
