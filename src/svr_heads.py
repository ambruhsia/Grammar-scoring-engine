"""RBF-SVR heads on single upper speech-encoder layers (approved idea #2, 2026-10-14).

The strongest audio views in other participants' write-ups (AMITABHSHUKLA, PJ2001-IND) are RBF-SVRs on one upper layer of a
frozen encoder, where our audio heads are ridges on the uniform mean of every layer. Fixed before running, not tuned on our CV:
    wavlm_large_l20_svr_pred   WavLM-large layer 20 (the leaderboard #1's choice), mean + std over time ->
                               StandardScaler -> SVR(rbf, C=3, epsilon=0.1, gamma="scale"), the settings of our `svr` family
Fit on the denoised target like the other members; OOF on ctx.folds (3 seeds), test = mean of the fold models.
"""
from __future__ import annotations

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import leakfree

HEADS = {"wavlm_large_l20_svr_pred": ("ssl_wavlm_large.npz", 20)}


def member_inputs(ctx) -> dict:
    """-> {col: (X train, X test, model)}."""
    out = {}
    for col, (file, layer) in HEADS.items():
        z, tr, te = leakfree._rows(ctx, file)
        H = z["H"][:, layer].astype(np.float64)
        Htr, Hte = H[tr], H[te]
        if getattr(ctx, "batch_norm_audio", False):        # Phase I: z-score within each recording batch (src/batchnorm.py)
            import batchnorm
            g_tr, g_te = batchnorm.batch_groups(ctx)
            Htr, Hte = batchnorm.z_by_batch(Htr, Hte, g_tr, g_te)
        out[col] = (Htr, Hte, make_pipeline(StandardScaler(), SVR(kernel="rbf", C=3.0, epsilon=0.1, gamma="scale")))
    return out
