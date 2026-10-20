"""Phi-4 examiner numbers and hidden-state heads (kaggle/phi4_judge -> cache/phi4/), for src/variants.py.

The job asked microsoft/phi-4 (MIT, NF4) the Qwen3.5 round-1 questions (the official rubric; grammatical accuracy) about three
transcripts (verbatim, CTC raw, Whisper clean), one forward pass each, keeping the next-token distribution over the digits 1-5
and the state at the answer position (layers at 50 / 65 / 80% depth).
    judge numbers  per (transcript, question): expected grade and log-odds of grade >= 4 -> 12 columns phi4_*  (label-free)
    phi4_{view}_pred  per transcript: both questions' answer states, layer-averaged and concatenated (2 x 5120) -> standardise ->
                      ridge, alpha by an inner GroupKFold(3) on speaker groups (the Qwen3.5 heads' protocol), raw label;
                      out-of-fold on every fold seed of the 8 runs, test = mean of the fold models, plus topic-held-out OOF.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DIR = Path(__file__).resolve().parents[1] / "cache" / "phi4"
VIEWS, PROMPTS = ("verbatim", "ctc", "whisper"), ("holistic", "accuracy")
VERSION = "phi4-v1"


def judge_numbers(keys) -> pd.DataFrame:
    J = pd.read_csv(DIR / "phi4_judge.csv").set_index(["split", "filename"]).loc[keys]
    out = {}
    for v in VIEWS:
        for p in PROMPTS:
            P = J[[f"{v}_{p}_p{i}" for i in range(1, 6)]].values
            P = P / P.sum(1, keepdims=True)
            out[f"phi4_{v}_{p}_ev"] = P @ np.arange(1, 6)
            out[f"phi4_{v}_{p}_lo"] = np.log(P[:, 3:].sum(1) + 1e-9) - np.log(P[:, :3].sum(1) + 1e-9)
    return pd.DataFrame(out, index=J.index)


def head_inputs(keys) -> dict:
    z = np.load(DIR / "phi4_hidden.npz")
    idx = {k: i for i, k in enumerate(zip(z["split"], z["filename"]))}
    rows = [idx[k] for k in keys]
    vlist = [str(v) for v in z["views"]]
    return {f"phi4_{v}_pred": z["H"][rows, vlist.index(v)].astype(np.float32).mean(2).reshape(len(rows), -1) for v in VIEWS}


def model():
    return GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": np.logspace(0, 6, 31)},
                        cv=GroupKFold(3), scoring="neg_root_mean_squared_error")


def heads(fn, tfn, y, groups, folds: dict, tfolds) -> dict:
    """-> {col: {"oof": {seed: (n,)}, "test": {seed: (216,)}, "topic": (n,)}}, cached in cache/phi4/phi4_heads_<sig>.npz."""
    h = hashlib.sha256(VERSION.encode())
    for s in sorted(folds):
        for _, b in folds[s]:
            h.update(np.asarray(b).tobytes())
    h.update(np.asarray(y, float).tobytes())
    path = DIR / f"phi4_heads_{h.hexdigest()[:16]}.npz"
    keys, tkeys = [("train", f) for f in fn], [("test", f) for f in tfn]
    Xtr, Xte = head_inputs(keys), head_inputs(tkeys)
    if path.exists():
        z = np.load(path)
        return {c: {"oof": {s: z[f"{c}__oof__{s}"] for s in folds}, "test": {s: z[f"{c}__test__{s}"] for s in folds},
                    "topic": z[f"{c}__topic"]} for c in Xtr}
    out, save = {}, {}
    for c in Xtr:
        X, Xt = Xtr[c], Xte[c]
        res = {"oof": {}, "test": {}}
        for s, fl in folds.items():
            o, t = np.zeros(len(y)), np.zeros(len(Xt))
            for a, b in fl:
                m = clone(model()).fit(X[a], y[a], groups=groups[a])
                o[b] = m.predict(X[b]); t += m.predict(Xt) / len(fl)
            res["oof"][s], res["test"][s] = o, t
            save[f"{c}__oof__{s}"], save[f"{c}__test__{s}"] = o, t
        tp = np.full(len(y), np.nan)
        for a, b in tfolds:
            tp[b] = clone(model()).fit(X[a], y[a], groups=groups[a]).predict(X[b])
        res["topic"] = save[f"{c}__topic"] = tp
        out[c] = res
        print(f"  {c}: OOF RMSE {np.mean([np.sqrt(np.mean((np.clip(o, 1.5, 5) - y) ** 2)) for o in res['oof'].values()]):.4f} "
              f"(mean over {len(folds)} fold seeds), topic-held-out {np.sqrt(np.mean((np.clip(tp, 1.5, 5) - y) ** 2)):.4f}", flush=True)
    np.savez(path, **save)
    return out
