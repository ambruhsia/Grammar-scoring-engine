"""Frozen-encoder heads on speaker-grouped folds (PLAN_RUBRIC C1). Replaces the random-fold audio *_pred columns.

Per encoder (cache/*.npz: every hidden layer's mean + std over time), each layer is z-scored on train rows and
the layers of a fixed band are averaged:
    wavlm_large   layers 20-22   fixed a priori (plan)
    wavlm         layers 6-8     best layer 7 on seed-42 grouped OOF, +-1
    whisper_med   layers 22-24   best layer 24 on seed-42 grouped OOF, -2..0 (24 is the last layer)
Two heads per encoder, OOF on ctx.folds (test = mean of the fold models):
    <enc>_ridge_pred   StandardScaler -> Ridge, alpha from logspace(0, 6, 31)
    <enc>_svr_pred     StandardScaler -> RBF SVR (gamma "scale", epsilon 0.1), C from {1, 3, 10}
Hyperparameters are picked by an inner GroupKFold(3) on speaker groups, so twins can't leak into the
choice either. Results are cached in cache/audio_heads_<signature>.npz.
"""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ENCODERS = {"wavlm_large": ("ssl_wavlm_large.npz", (20, 21, 22)),
            "wavlm": ("wavlm_layers.npz", (6, 7, 8)),
            "whisper_med": ("ssl_whisper_med.npz", (22, 23, 24))}
REPLACES = ["wavlm_pred", "wavlm_large_pred", "whisper_med_pred"]
VERSION = "c1-v1"


def heads():
    ridge = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": np.logspace(0, 6, 31)},
                         cv=GroupKFold(3), scoring="neg_root_mean_squared_error")
    svr = GridSearchCV(make_pipeline(StandardScaler(), SVR(kernel="rbf", gamma="scale", epsilon=0.1)),
                       {"svr__C": [1.0, 3.0, 10.0]}, cv=GroupKFold(3), scoring="neg_root_mean_squared_error")
    return {"ridge": ridge, "svr": svr}


def band_features(ctx, file, band):
    z = np.load(ctx.cache / file)
    idx = {k: i for i, k in enumerate(zip(z["split"], z["filename"]))}
    tr = [idx[("train", f)] for f in ctx.T.filename]
    te = [idx[("test", f)] for f in ctx.TE.filename]
    H = z["H"][:, list(band)].astype(np.float64)
    mu, sd = H[tr].mean(0, keepdims=True), H[tr].std(0, keepdims=True) + 1e-6
    Hz = ((H - mu) / sd).mean(1)
    return Hz[tr], Hz[te]


def signature(ctx) -> str:
    h = hashlib.sha256(VERSION.encode() + repr(sorted(ENCODERS.items())).encode())
    for s in ctx.seeds:
        for a, b in ctx.folds[s]:
            h.update(np.asarray(b).tobytes())
    h.update(ctx.T.filename.str.cat().encode())
    return h.hexdigest()[:16]


def build(ctx) -> dict:
    """-> {column: (oof per seed, test per seed)}; cached per fold signature."""
    path = ctx.cache / f"audio_heads_{signature(ctx)}.npz"
    if path.exists():
        z = np.load(path)
        cols = sorted({k.rsplit("__", 2)[0] for k in z.files})
        return {c: ([z[f"{c}__oof__{i}"] for i in range(len(ctx.seeds))],
                    [z[f"{c}__test__{i}"] for i in range(len(ctx.seeds))]) for c in cols}
    out, y, g = {}, ctx.y, ctx.groups
    for enc, (file, band) in ENCODERS.items():
        X, Xt = band_features(ctx, file, band)
        for hname, model in heads().items():
            oofs, tests = [], []
            for s in ctx.seeds:
                o, t = np.zeros(len(y)), np.zeros(len(Xt))
                for a, b in ctx.folds[s]:
                    m = clone(model).fit(X[a], y[a], groups=g[a])
                    o[b] = m.predict(X[b])
                    t += m.predict(Xt) / len(ctx.folds[s])
                oofs.append(o); tests.append(t)
            out[f"{enc}_{hname}_pred"] = (oofs, tests)
            print(f"  audio head {enc}_{hname}: done", flush=True)
    np.savez(path, **{f"{c}__oof__{i}": o for c, (os_, _) in out.items() for i, o in enumerate(os_)},
             **{f"{c}__test__{i}": t for c, (_, ts) in out.items() for i, t in enumerate(ts)})
    return out
