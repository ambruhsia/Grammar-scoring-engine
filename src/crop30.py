"""First-30-s crop embeddings (PLAN_RUBRIC C4 / G6): adversarial batch check and ridge heads on speaker-grouped folds.

Embeddings: cache/crop30_wavlm_large.npz (25 layers) and cache/crop30_whisper_large_v3.npz (33 layers), mean + std per layer.
Layer bands fixed a priori (plan C1): WavLM-large 20-22, Whisper-large-v3 encoder 29-32. Per-layer z-score on train rows,
band mean. Heads: StandardScaler -> RidgeCV(logspace(0, 6, 31)), OOF on ctx.folds.

    python src/crop30.py        # adversarial AUC: can short vs long be told apart from equal-length crops?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CROPS = {"crop30_wavlm_large_pred": ("crop30_wavlm_large.npz", (20, 21, 22)),
         "crop30_whisper_v3_pred": ("crop30_whisper_large_v3.npz", (29, 30, 31, 32))}
FULL = {"full_wavlm_large": ("ssl_wavlm_large.npz", (20, 21, 22))}


def band(cache: Path, file: str, layers, train_keys: set):
    z = np.load(cache / file)
    keys = list(zip(z["split"], z["filename"]))
    H = z["H"][:, list(layers)].astype(np.float64)
    tr = np.array([k in train_keys for k in keys])
    mu, sd = H[tr].mean(0, keepdims=True), H[tr].std(0, keepdims=True) + 1e-6
    return dict(zip(keys, ((H - mu) / sd).mean(1)))


def member_inputs(ctx) -> dict:
    """-> {col: (X train, X test, model)} for the crop heads."""
    tr = [("train", f) for f in ctx.T.filename]
    te = [("test", f) for f in ctx.TE.filename]
    out = {}
    for col, (file, layers) in CROPS.items():
        B = band(ctx.cache, file, layers, set(tr))
        out[col] = (np.stack([B[k] for k in tr]), np.stack([B[k] for k in te]),
                    make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 6, 31))))
    return out


def adversarial():
    import pandas as pd
    G = pd.read_csv(ROOT / "cache" / "speaker_groups.csv")
    keys = list(zip(G.split, G.filename))
    y = (G.batch == "short").values.astype(int)
    grp = np.where(G.split == "train", G.group.astype(str), "test_" + G.filename)   # every test clip is its own speaker
    train_keys = set(k for k in keys if k[0] == "train")
    for name, (file, layers) in {**CROPS, **FULL}.items():
        B = band(ROOT / "cache", file, layers, train_keys)
        X = np.stack([B[k] for k in keys])
        p = np.zeros(len(y))
        for a, b in GroupKFold(5).split(X, y, grp):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(X[a], y[a])
            p[b] = m.predict_proba(X[b])[:, 1]
        print(f"{name:<26} short-vs-long AUC {roc_auc_score(y, p):.3f}   (train only {roc_auc_score(y[G.split == 'train'], p[G.split == 'train']):.3f})")


if __name__ == "__main__":
    adversarial()
