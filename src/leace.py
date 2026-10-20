"""LEACE: closed-form linear concept erasure (Belrose et al., 2023), used here to erase the prompt topic from text embeddings.

Text members lean on familiar prompt content (topic-held-out RMSE up to +0.10 worse), and test answers prompts training barely
has. LEACE removes every direction along which a linear classifier could detect the concept (here the 10 k-means prompt clusters
of cache/insight.csv, computed without labels on train + test), while moving the embeddings as little as possible:
    whiten X (so its covariance is the identity), project out the span of the whitened cross-covariance with the one-hot topic,
    unwhiten. With n < d this is done in the span of the data via the SVD X_c = U S V':  X'_c = X_c - U Q Q' S V',
    Q = an orthonormal basis of the column span of U' Z_c (Z_c = centred one-hot topics).
Fitted on all train + test clips (no grades involved). After erasure no linear classifier beats chance at predicting the topic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def topics_for(cache, keys) -> np.ndarray:
    I = pd.read_csv(cache / "insight.csv").set_index(["split", "filename"])
    return I.loc[list(keys), "topic"].values


def erase(X: np.ndarray, topic: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, np.float64)
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    keep = S > tol * S.max()
    U, S, Vt = U[:, keep], S[keep], Vt[keep]
    Z = pd.get_dummies(pd.Series(topic)).values.astype(np.float64)
    Zc = Z - Z.mean(0, keepdims=True)
    Q, R = np.linalg.qr(U.T @ Zc)                     # r x k; keep the directions with non-negligible weight
    Q = Q[:, np.abs(np.diag(R)) > 1e-10]
    return mu + Xc - (U @ Q) @ ((Q.T * S) @ Vt)
