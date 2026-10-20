"""Qwen3.5-9B members (Phase I; kaggle/qwen35_judge job -> cache/qwen35/).

The job asked Qwen/Qwen3.5-9B (Apache-2.0, NF4) two questions (the official rubric; grammatical accuracy) about three transcripts
(verbatim Whisper-large-v3-turbo, CTC raw, Whisper clean), one forward pass each, and kept the next-token distribution over the
digits 1-5 and the state at the answer position (LM layers at 50 / 65 / 80% depth).
    judge numbers   per (transcript, question): expected grade and log-odds of grade >= 4 -> 12 columns (judge35_*)
    qwen35_judge_pred      StandardScaler -> RidgeCV on the 12 judge numbers
    qwen35_{view}_pred     per transcript: the answer-position states of both questions, layer-averaged and concatenated (8192
                           dims) -> GridSearchCV ridge with speaker-grouped inner folds (the protocol of our Qwen2.5 heads), raw y
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VERSION = "q35-v1"
VIEWS, PROMPTS = ("verbatim", "ctc", "whisper"), ("holistic", "accuracy")


def judge_table(cache) -> pd.DataFrame:
    J = pd.read_csv(cache / "qwen35" / "qwen35_judge.csv").set_index(["split", "filename"])
    out = pd.DataFrame(index=J.index)
    for v in VIEWS:
        for p in PROMPTS:
            P = J[[f"{v}_{p}_p{i}" for i in range(1, 6)]].values
            P = P / P.sum(1, keepdims=True)
            out[f"judge35_{v}_{p}_ev"] = P @ np.arange(1, 6)
            out[f"judge35_{v}_{p}_lo"] = np.log(P[:, 3:].sum(1) + 1e-9) - np.log(P[:, :3].sum(1) + 1e-9)
    return out


def judge_inputs(ctx):
    J = judge_table(ctx.cache)
    X = J.loc[[("train", f) for f in ctx.T.filename]].values
    Xt = J.loc[[("test", f) for f in ctx.TE.filename]].values
    return X, Xt, make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))


def member_inputs(ctx, views=VIEWS) -> dict:
    z = np.load(ctx.cache / "qwen35" / "qwen35_hidden.npz")
    idx = {k: i for i, k in enumerate(zip(z["split"], z["filename"]))}
    tr = [idx[("train", f)] for f in ctx.T.filename]
    te = [idx[("test", f)] for f in ctx.TE.filename]
    vlist = [str(v) for v in z["views"]]
    out = {}
    for v in views:
        H = z["H"][:, vlist.index(v)].astype(np.float32)          # [clips, prompts, layers, hidden]
        X = H.mean(2).reshape(len(H), -1)                           # layer mean, prompts concatenated
        if getattr(ctx, "leace_text", False):                      # Phase I: erase the prompt topic (src/leace.py)
            import leace
            X = leace.erase(X, leace.topics_for(ctx.cache, zip(z["split"], z["filename"]))).astype(np.float32)
        model = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": np.logspace(0, 6, 31)},
                             cv=GroupKFold(3), scoring="neg_root_mean_squared_error")
        out[f"qwen35_{v}_pred"] = (X[tr], X[te], model)
    return out


def _sig(ctx, folds_list, tag, y):
    h = hashlib.sha256((VERSION + tag).encode())
    for folds in folds_list:
        for _, b in folds:
            h.update(np.asarray(b).tobytes())
    h.update(np.asarray(y, float).tobytes())
    return h.hexdigest()[:16]


def heads(ctx) -> dict:
    """-> {col: (oof per seed, test per seed)} for the three hidden-state heads; fitted on raw y with speaker-grouped inner CV."""
    from common import fit_weighted
    tg = "grouped_leace" if getattr(ctx, "leace_text", False) else "grouped"
    path = ctx.cache / "qwen35" / f"q35_heads_{_sig(ctx, [ctx.folds[s] for s in ctx.seeds], tg, ctx.y)}.npz"
    inputs = member_inputs(ctx)
    if path.exists():
        z = np.load(path)
        return {c: ([z[f"{c}__oof__{i}"] for i in range(len(ctx.seeds))], [z[f"{c}__test__{i}"] for i in range(len(ctx.seeds))])
                for c in inputs}
    out = {}
    for c, (X, Xt, model) in inputs.items():
        oofs, tests = [], []
        for s in ctx.seeds:
            o, t = np.zeros(len(ctx.y)), np.zeros(len(Xt))
            for a, b in ctx.folds[s]:
                m = fit_weighted(clone(model), X[a], ctx.y[a], None, groups=ctx.groups[a])
                o[b] = m.predict(X[b]); t += m.predict(Xt) / len(ctx.folds[s])
            oofs.append(o); tests.append(t)
        out[c] = (oofs, tests)
        print(f"  {c}: OOF RMSE {np.mean([np.sqrt(np.mean((np.clip(o, 1.5, 5) - ctx.y) ** 2)) for o in oofs]):.4f}", flush=True)
    np.savez(path, **{f"{c}__oof__{i}": o for c, (os_, _) in out.items() for i, o in enumerate(os_)},
             **{f"{c}__test__{i}": t for c, (_, ts) in out.items() for i, t in enumerate(ts)})
    return out


def topic_heads(ctx, folds) -> dict:
    from topic_cv import topic_oof
    tg = "topic_leace" if getattr(ctx, "leace_text", False) else "topic"
    path = ctx.cache / "qwen35" / f"q35_topic_{_sig(ctx, [folds], tg, ctx.y)}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    out = {c: topic_oof(model, X, ctx.y, folds, ctx.groups) for c, (X, Xt, model) in member_inputs(ctx).items()}
    np.savez(path, **out)
    return out
