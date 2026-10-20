"""Topic-held-out validation (PLAN_RUBRIC G11). About 40% of test answers prompts the training set barely has.

Topic folds: for each topic T (k-means topics from src/insight.py, cache/insight.csv), validate on T's training clips and train
on clips outside T whose speaker group has no clip in T. Every training clip is validated exactly once.
A member's topic-held-out OOF next to its speaker-grouped OOF shows how much it leans on familiar prompt content.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV


def topic_folds(ctx) -> tuple[list, np.ndarray]:
    I = pd.read_csv(ctx.cache / "insight.csv")
    topic = I[I.split == "train"].set_index("filename").topic
    t = topic.loc[ctx.T.filename].values
    folds = []
    for T in sorted(set(t)):
        b = np.where(t == T)[0]
        a = np.where((t != T) & ~np.isin(ctx.groups, np.unique(ctx.groups[b])))[0]
        folds.append((a, b))
    return folds, t


def fit(model, X, y, rows, groups, sw=None):
    from common import fit_weighted
    m = clone(model)
    w = None if sw is None else sw[rows]
    if isinstance(m, GridSearchCV) and groups is not None:
        return fit_weighted(m, X[rows], y[rows], w, groups=groups[rows])
    return fit_weighted(m, X[rows], y[rows], w)


def topic_oof(model, X, y, folds, groups=None, sw=None) -> np.ndarray:
    o = np.full(len(y), np.nan)
    for a, b in folds:
        o[b] = fit(model, X, y, a, groups, sw).predict(X[b])
    return o
