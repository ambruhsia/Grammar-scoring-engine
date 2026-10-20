"""Covariate-shift weights: how test-like each training clip is, within its recording batch (2026-10-19).

Test short clips are different recordings from the training ones (speech PCs: within-batch adversarial AUC 0.76). Density
ratio r(x) = p(test | x) / p(train | x) * n_train / n_test, from a logistic regression on the 48 speech PCs (src/emb_pcs.py)
fitted per recording batch (exact 45.06 s / other short / long) on train + test clips; each training clip's p(test | x) is
out-of-fold (5 folds), so a clip never scores itself. No labels are used. Weights are clipped to [CLIP_LO, CLIP_HI] and
normalised to mean 1 within each batch.
Uses: importance-weighted training (src/experiment.py IW_FAMILIES) and importance-weighted CV (validation clips weighted by
r on top of the batch test-mix weight): an estimate of the error on the test distribution rather than the training one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import emb_pcs

CLIP_LO, CLIP_HI = 0.2, 5.0


def ratios(cache, train_files) -> np.ndarray:
    """-> density ratio per training clip (in train_files order)."""
    P = emb_pcs.table(cache, 16)
    b3 = emb_pcs.batches(cache, P.index)
    split = P.index.get_level_values(0)
    r = pd.Series(np.nan, index=P.loc["train"].index)
    for b in ("b45", "short", "long"):
        m = b3 == b
        X, is_test = P.values[m], (split[m] == "test").astype(int)
        p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)), X, is_test,
                              cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        tr = is_test == 0
        rb = p[tr] / (1 - p[tr]) * tr.sum() / is_test.sum()
        rb = np.clip(rb / rb.mean(), CLIP_LO, CLIP_HI)
        r.loc[P.index[m][tr].get_level_values(1)] = rb / rb.mean()
    return r.loc[list(train_files)].values


def ess(w) -> float:
    """Effective sample size of a weight vector (Kish)."""
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum())
