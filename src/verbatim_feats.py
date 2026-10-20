"""Members on the verbatim Whisper-large-v3-turbo transcripts (approved idea #3, 2026-10-14; kaggle/verbatim job -> cache/verbatim/).

    qwen_verbatim_pred   Qwen2.5-7B rubric-prompt hidden states on the verbatim text (last token + mean, layers 16/20/24):
                         exactly the recipe of qwen_ctc_pred / qwen_whisper_pred (StandardScaler -> Ridge, alpha from
                         logspace(0, 6, 31) by inner GroupKFold(3) on speaker groups, fit on the raw labels like them).
    deberta_vpair_pred   DeBERTa-v3-base fine-tuned on (verbatim, Whisper clean) on the grouped seed-42 folds (the job's OOF),
                         a fixed column like deberta_pair_pred.
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


def qwen_inputs(ctx) -> dict:
    z = np.load(ctx.cache / "verbatim" / "qwen_hidden_verbatim.npz")
    idx = {k: i for i, k in enumerate(zip(z["split"], z["filename"]))}
    tr = [idx[("train", f)] for f in ctx.T.filename]
    te = [idx[("test", f)] for f in ctx.TE.filename]
    H = np.concatenate([z["verbatim_last"], z["verbatim_mean"]], axis=1).astype(np.float32).reshape(len(z["split"]), -1)
    model = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": np.logspace(0, 6, 31)},
                         cv=GroupKFold(3), scoring="neg_root_mean_squared_error")
    return {"qwen_verbatim_pred": (H[tr], H[te], model)}


def qwen_heads(ctx) -> dict:
    """-> {"qwen_verbatim_pred": (oof per seed, test per seed)} on ctx.folds, raw labels (the Qwen-head recipe)."""
    out = {}
    for name, (X, Xt, model) in qwen_inputs(ctx).items():
        oofs, tests = [], []
        for s in ctx.seeds:
            o, t = np.zeros(len(ctx.y)), np.zeros(len(Xt))
            for a, b in ctx.folds[s]:
                m = fit_weighted(clone(model), X[a], ctx.y[a], None, groups=ctx.groups[a])
                o[b] = m.predict(X[b])
                t += m.predict(Xt) / len(ctx.folds[s])
            oofs.append(o); tests.append(t)
        out[name] = (oofs, tests)
    return out


def deberta_table(ctx) -> pd.DataFrame:
    return pd.read_csv(ctx.cache / "verbatim" / "deberta_vpair.csv").set_index(["split", "filename"])
