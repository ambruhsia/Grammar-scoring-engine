"""Laya rubric-question members (kaggle/laya job outputs in cache/laya/; approved 2026-10-14).

Learned columns (OOF on ctx.folds, test = mean of the fold models):
    laya_ctc_pred / laya_whisper_pred   StandardScaler -> Ridge on [the encoder's mean-pooled state while answering the rubric
                                        question (1024) + every question's level probabilities (15)]; alpha from logspace(0, 6, 31)
                                        by inner GroupKFold(3) on speaker groups. Fit on the denoised target like the other members
                                        (chosen before seeing any result).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import fit_weighted


def member_inputs(ctx) -> dict:
    """-> {"laya_ctc_pred" / "laya_whisper_pred": (X train, X test, model)}."""
    d = ctx.cache / "laya"
    z = np.load(d / "laya_hidden.npz")
    J = pd.read_csv(d / "laya_judge.csv").set_index(["split", "filename"])
    keys = list(zip(z["split"], z["filename"]))
    idx = {k: i for i, k in enumerate(keys)}
    tr = [idx[("train", f)] for f in ctx.T.filename]
    te = [idx[("test", f)] for f in ctx.TE.filename]
    out = {}
    for text in ("ctc", "whisper"):
        P = J.loc[keys, [c for c in J.columns if c.startswith(f"{text}_") and not c.endswith("_score")]].values
        X = np.concatenate([z[f"{text}_mean"].astype(np.float32), P.astype(np.float32)], axis=1)
        model = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": np.logspace(0, 6, 31)},
                             cv=GroupKFold(3), scoring="neg_root_mean_squared_error")
        out[f"laya_{text}_pred"] = (X[tr], X[te], model)
    return out


def heads(ctx, target) -> dict:
    """-> {"laya_ctc_pred": (oof per seed, test per seed), "laya_whisper_pred": ...}."""
    yt = np.asarray(target, float)
    out = {}
    for name, (X, Xt, model) in member_inputs(ctx).items():
        oofs, tests = [], []
        for s in ctx.seeds:
            o, t = np.zeros(len(yt)), np.zeros(len(Xt))
            for a, b in ctx.folds[s]:
                m = fit_weighted(clone(model), X[a], yt[a], None, groups=ctx.groups[a])
                o[b] = m.predict(X[b])
                t += m.predict(Xt) / len(ctx.folds[s])
            oofs.append(o); tests.append(t)
        out[name] = (oofs, tests)
    return out


def topic_heads(ctx, folds, target) -> dict:
    """Topic-held-out OOF of the two heads (G11)."""
    from topic_cv import topic_oof
    return {name: topic_oof(model, X, np.asarray(target, float), folds, ctx.groups)
            for name, (X, Xt, model) in member_inputs(ctx).items()}
