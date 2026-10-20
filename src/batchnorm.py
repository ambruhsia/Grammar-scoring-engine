"""Per-recording-batch normalisation of audio embeddings (Phase I, 2026-10-17).

The two recording batches stay separable from equal-length 30-s crops (AUC 0.93-0.95): part of what the speech embeddings encode
is the recording, not the speaker. Speaker verification handles this kind of mismatch by re-centring embeddings with unlabelled
data from each recording condition. Here every audio embedding is z-scored WITHIN its recording batch, using all of that batch's
clips (train and test, no labels). Because centring also removes any genuine difference in grade between the batches, the ridge
heads get a short-batch indicator as an extra input (a per-batch intercept learned from the training labels).
"""
from __future__ import annotations

import numpy as np


def batch_groups(ctx):
    """-> (train batch labels, test batch labels)."""
    return np.asarray(ctx.batch), np.asarray(ctx.batch_test)


def z_by_batch(H_tr, H_te, g_tr, g_te):
    """z-score each feature within each batch over its train + test rows. Works for any trailing shape."""
    H_tr, H_te = np.asarray(H_tr, np.float64), np.asarray(H_te, np.float64)
    out_tr, out_te = np.empty_like(H_tr), np.empty_like(H_te)
    for g in np.unique(np.r_[g_tr, g_te]):
        A = np.concatenate([H_tr[g_tr == g], H_te[g_te == g]])
        mu, sd = A.mean(0, keepdims=True), A.std(0, keepdims=True) + 1e-6
        out_tr[g_tr == g] = (H_tr[g_tr == g] - mu) / sd
        out_te[g_te == g] = (H_te[g_te == g] - mu) / sd
    return out_tr, out_te


def with_indicator(X_tr, X_te, g_tr, g_te):
    """Append the short-batch indicator (the per-batch intercept)."""
    return np.c_[X_tr, (g_tr == "short").astype(float)], np.c_[X_te, (g_te == "short").astype(float)]
