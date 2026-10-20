"""The leak-free stack (PLAN_RUBRIC C1/C2/C5 + the agreed rebaseline): every learned column OOF on speaker-grouped folds.

Rebuilt locally, per fold seed, with the notebook's own protocol (it beat the C1 band heads at stack level):
    wavlm_pred / wavlm_large_pred / whisper_med_pred   per-layer z-score, uniform layer mean, RidgeCV(logspace(0, 6, 31))
    emb_pred                                           all-mpnet-base-v2 on Whisper text (cache/emb_mpnet.npz),
                                                       RidgeCV(logspace(-1, 4, 30))
From the C2 GPU job (cache/deberta_grouped.csv; OOF on the grouped seed-42 folds):
    deberta_pred / deberta_pair_pred
Dropped (random-fold OOF that can't be rebuilt here, or only useful through speaker twins):
    spk_knn_pred, deberta_ctc_pred, and ft_wavlm_pred when DROP_FT_WAVLM (C5 rule)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from common import fit_weighted
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FROZEN = {"wavlm_pred": "wavlm_layers.npz", "wavlm_large_pred": "ssl_wavlm_large.npz",
          "whisper_med_pred": "ssl_whisper_med.npz"}
ALWAYS_DROP = {"spk_knn_pred", "deberta_ctc_pred"}


def _oof(X, Xt, y, model, ctx):
    oofs, tests = [], []
    for s in ctx.seeds:
        o, t = np.zeros(len(y)), np.zeros(len(Xt))
        for a, b in ctx.folds[s]:
            sw = getattr(ctx, "member_w", None)     # G5(b) test-mix sample weights, when on
            m = fit_weighted(clone(model), X[a], y[a], None if sw is None else sw[a])  # y = ctx.y_fit (D3)
            o[b] = m.predict(X[b])
            t += m.predict(Xt) / len(ctx.folds[s])
        oofs.append(o); tests.append(t)
    return oofs, tests


def _rows(ctx, file):
    z = np.load(ctx.cache / file)
    idx = {k: i for i, k in enumerate(zip(z["split"], z["filename"]))}
    return z, [idx[("train", f)] for f in ctx.T.filename], [idx[("test", f)] for f in ctx.TE.filename]


def member_inputs(ctx) -> dict:
    """-> {col: (X train, X test, model)} for the locally rebuilt columns."""
    out = {}
    for col, file in FROZEN.items():
        z, tr, te = _rows(ctx, file)
        H = z["H"]
        if getattr(ctx, "batch_norm_audio", False):        # Phase I: z-score within each recording batch (src/batchnorm.py)
            import batchnorm
            g_tr, g_te = batchnorm.batch_groups(ctx)
            Ztr, Zte = batchnorm.z_by_batch(H[tr], H[te], g_tr, g_te)
            Xtr, Xte = batchnorm.with_indicator(Ztr.mean(1), Zte.mean(1), g_tr, g_te)
            out[col] = (Xtr, Xte, make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 6, 31))))
            continue
        mu, sd = H[tr].mean(0, keepdims=True), H[tr].std(0, keepdims=True) + 1e-6
        out[col] = (((H[tr] - mu) / sd).mean(1), ((H[te] - mu) / sd).mean(1),
                    make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 6, 31))))
    z, tr, te = _rows(ctx, "emb_mpnet.npz")
    E = z["E"]
    if getattr(ctx, "leace_text", False):                   # Phase I: erase the prompt topic (src/leace.py)
        import leace
        E = leace.erase(E, leace.topics_for(ctx.cache, zip(z["split"], z["filename"])))
    out["emb_pred"] = (E[tr], E[te], make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30))))
    return out


def build(ctx, drop_ft_wavlm: bool) -> tuple[dict, dict, set]:
    """-> (per-seed columns {col: (oofs, tests)}, fixed leak-free columns {col: (train, test)}, dropped set)."""
    per_seed = {col: _oof(X, Xt, getattr(ctx, "y_fit", ctx.y), model, ctx)
                for col, (X, Xt, model) in member_inputs(ctx).items()}
    fixed = {}
    dg = ctx.cache / "deberta_grouped.csv"
    if dg.exists():
        D = pd.read_csv(dg).set_index(["split", "filename"])
        for col in ("deberta_pred", "deberta_pair_pred"):
            if col in D.columns:
                fixed[col] = (D.loc[[("train", f) for f in ctx.T.filename], col].values,
                              D.loc[[("test", f) for f in ctx.TE.filename], col].values)
    dropped = set(ALWAYS_DROP) | ({"ft_wavlm_pred"} if drop_ft_wavlm else set())
    if not dg.exists():
        dropped |= {"deberta_pred", "deberta_pair_pred"}    # no grouped DeBERTa yet: leave them out, not leaky
    return per_seed, fixed, dropped
