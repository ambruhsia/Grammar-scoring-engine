"""Voxtral-Mini-3B audio-LLM members (approved idea #1, 2026-10-14; kaggle/voxtral job -> cache/voxtral/).

    voxtral_pred       language-model layers 6-18 (the lower-middle band): the mean hidden state over the audio tokens that
                       cover real audio, while the model is asked about the speaker's grammar. Same protocol as our audio heads:
                       z-score each layer on the training rows, uniform mean over the layers, StandardScaler -> RidgeCV
                       (alphas logspace(0, 6, 31)); fit on the denoised target.
    voxtral_enc_pred   the model's audio encoder (a Whisper-large-v3 encoder) layers 16-32, mean + std over real frames, the
                       same protocol (a separate gate test, after voxtral_pred).
Clips whose states were non-finite in fp16 are set to the training mean after z-scoring (0).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SOURCES = {"voxtral_pred": ("voxtral_lm.npz", "A"), "voxtral_enc_pred": ("voxtral_enc.npz", "H"),
           "voxtral_b_pred": ("voxtral_views.npz", "B"),      # Phase I: question BEFORE the audio (rubric question)
           "voxtral_c_pred": ("voxtral_views.npz", "C")}      # Phase I: question before the audio (error checklist)
AVERAGED = {"voxtral_abc_pred": ("voxtral_pred", "voxtral_b_pred", "voxtral_c_pred")}   # mean of the three views' inputs


def member_inputs(ctx, which=("voxtral_pred",)) -> dict:
    """-> {col: (X train, X test, model)} for the requested members."""
    out = {}
    for col in [c for c in which if c in AVERAGED]:
        parts = member_inputs(ctx, AVERAGED[col])
        out[col] = (np.mean([v[0] for v in parts.values()], 0), np.mean([v[1] for v in parts.values()], 0),
                    make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 6, 31))))
    for col in [c for c in which if c not in AVERAGED]:
        file, key = SOURCES[col]
        z = np.load(ctx.cache / "voxtral" / file)
        idx = {k: i for i, k in enumerate(zip(z["split"], z["filename"]))}
        tr = [idx[("train", f)] for f in ctx.T.filename]
        te = [idx[("test", f)] for f in ctx.TE.filename]
        H = z[key].astype(np.float64)                                     # [clips, layers, dims]
        if getattr(ctx, "batch_norm_audio", False):                       # Phase I: z-score within each recording batch
            import batchnorm
            g_tr, g_te = batchnorm.batch_groups(ctx)
            Ztr, Zte = batchnorm.z_by_batch(np.nan_to_num(H[tr]), np.nan_to_num(H[te]), g_tr, g_te)
            Xtr, Xte = batchnorm.with_indicator(Ztr.mean(1), Zte.mean(1), g_tr, g_te)
            out[col] = (Xtr, Xte, make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 6, 31))))
            continue
        mu = np.nanmean(H[tr], 0, keepdims=True)
        sd = np.nanstd(H[tr], 0, keepdims=True) + 1e-6
        X = np.nan_to_num(((H - mu) / sd).mean(1), nan=0.0)               # uniform mean over layers
        out[col] = (X[tr], X[te], make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 6, 31))))
    return out
