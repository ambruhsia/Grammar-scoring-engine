"""Local half of PLAN_RUBRIC C3: turn the Qwen2.5-7B-Instruct job outputs (cache/qwen/) into model inputs.

Learned columns (OOF on ctx.folds, test = mean of the fold models):
    qwen_ctc_pred / qwen_whisper_pred   StandardScaler -> Ridge on the rubric-prompt hidden states (last token +
                                        mean over transcript tokens, layers 16/20/24, concatenated); alpha from
                                        logspace(0, 6, 31) by inner GroupKFold(3) on speaker groups.
Per-clip features (no labels used, except the train median as the fallback for corrected_error_ratio):
    llm_rubric_expected / _entropy      zero-shot judge on the CTC text: sum d * P(d), and the entropy of P(1..5)
    llm_rubric_expected_whisper         the same judge on the Whisper text (interpretation)
    llm_err_per100                      word-level edits between the CTC text and Qwen's minimal grammar correction,
                                        fillers removed on both sides, per 100 words                   (expected -)
    corrected_error_ratio               errors the speaker corrected / (corrected + uncorrected), from Qwen's JSON
                                        counts; the train median when a clip has no errors               (expected +)
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NON_LEXICAL = {"uh", "uhh", "um", "umm", "erm", "er", "ah", "ahh", "eh", "mm", "hmm", "hm", "uhm"}
_W = re.compile(r"[a-z']+")
VERSION = "c3-v1"
RUBRIC_COLS = ["llm_err_per100"]          # corrected_error_ratio failed the Phase B checks (reported as a finding)
JUDGE_COLS = ["llm_rubric_expected", "llm_rubric_entropy"]


def words(t):
    return [w for w in _W.findall((t or "").lower()) if w not in NON_LEXICAL]


def lev(a, b):
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def parse_counts(s):
    m = re.search(r"\{.*?\}", s or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
        return int(d.get("corrected", 0)), int(d.get("uncorrected", 0))
    except (ValueError, TypeError):
        return None


def features(cache, train_files) -> pd.DataFrame:
    """Per-clip judge / correction / self-correction features, indexed by (split, filename)."""
    d = cache / "qwen"
    J = pd.read_csv(d / "qwen_judge.csv")
    P = J[[f"ctc_p{i}" for i in range(1, 6)]].values
    Pw = J[[f"whisper_p{i}" for i in range(1, 6)]].values
    out = J[["split", "filename"]].copy()
    out["llm_rubric_expected"] = P @ np.arange(1, 6)
    out["llm_rubric_entropy"] = -(P * np.log(P + 1e-12)).sum(1)
    out["llm_rubric_expected_whisper"] = Pw @ np.arange(1, 6)
    C = {}
    for line in (cache / "ctc.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        C[(r["split"], r["filename"])] = r["raw_text"]
    G = {}
    for line in (d / "qwen_gen.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        G[(r["split"], r["filename"])] = r
    err, ratio, parsed = [], [], 0
    for k in zip(out.split, out.filename):
        src, g = words(C[k]), G.get(k, {})
        err.append(100 * lev(src, words(g.get("correction", ""))) / max(len(src), 1))
        c = parse_counts(g.get("selfcorr_json", ""))
        parsed += c is not None
        ratio.append(np.nan if c is None or sum(c) == 0 else c[0] / sum(c))
    out["llm_err_per100"] = err
    out["corrected_error_ratio"] = ratio
    med = out.loc[out.split.eq("train") & out.filename.isin(train_files), "corrected_error_ratio"].median()
    out["corrected_error_ratio"] = out["corrected_error_ratio"].fillna(med)
    out.attrs["parsed_json"] = parsed
    return out.set_index(["split", "filename"])


def _signature(ctx) -> str:
    h = hashlib.sha256(VERSION.encode())
    for s in ctx.seeds:
        for _, b in ctx.folds[s]:
            h.update(np.asarray(b).tobytes())
    h.update((ctx.cache / "qwen" / "qwen_hidden.npz").read_bytes()[:1 << 16])
    return h.hexdigest()[:16]


def member_inputs(ctx) -> dict:
    """-> {"qwen_ctc_pred" / "qwen_whisper_pred": (X train, X test, model)}."""
    z = np.load(ctx.cache / "qwen" / "qwen_hidden.npz")
    idx = {k: i for i, k in enumerate(zip(z["split"], z["filename"]))}
    tr = [idx[("train", f)] for f in ctx.T.filename]
    te = [idx[("test", f)] for f in ctx.TE.filename]
    out = {}
    for text in ("ctc", "whisper"):
        H = np.concatenate([z[f"{text}_last"], z[f"{text}_mean"]], axis=1).astype(np.float32).reshape(len(z["split"]), -1)
        if getattr(ctx, "leace_text", False):               # Phase I: erase the prompt topic (src/leace.py)
            import leace
            H = leace.erase(H, leace.topics_for(ctx.cache, zip(z["split"], z["filename"]))).astype(np.float32)
        model = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": np.logspace(0, 6, 31)},
                             cv=GroupKFold(3), scoring="neg_root_mean_squared_error")
        out[f"qwen_{text}_pred"] = (H[tr], H[te], model)
    return out


def topic_heads(ctx, folds, y_fit) -> dict:
    """Topic-held-out OOF of the two Qwen heads (G11); cached on the topic folds and target."""
    from topic_cv import topic_oof
    sw = getattr(ctx, "member_w", None)
    h = hashlib.sha256(VERSION.encode() + b"topic" + (b"w" if sw is not None else b"") + (b"leace" if getattr(ctx, "leace_text", False) else b""))
    for a, b in folds:
        h.update(np.asarray(b).tobytes())
    h.update(np.asarray(y_fit, float).tobytes())
    path = ctx.cache / "qwen" / f"qwen_topic_{h.hexdigest()[:16]}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    out = {name: topic_oof(model, X, np.asarray(y_fit, float), folds, ctx.groups, sw)
           for name, (X, Xt, model) in member_inputs(ctx).items()}
    np.savez(path, **out)
    return out


def heads(ctx, target=None) -> dict:
    """-> {"qwen_ctc_pred": (oof per seed, test per seed), "qwen_whisper_pred": ...}; cached.
    target None = the raw labels ctx.y (as every run up to 2026-10-14 fitted them); else e.g. the denoised ctx.y_fit."""
    sw = getattr(ctx, "member_w", None)
    yt = ctx.y if target is None else np.asarray(target, float)
    tag = "" if target is None else "_t" + hashlib.sha256(yt.tobytes()).hexdigest()[:8]
    tag += "_leace" if getattr(ctx, "leace_text", False) else ""
    path = ctx.cache / "qwen" / f"qwen_heads_{_signature(ctx)}{'_w' if sw is not None else ''}{tag}.npz"
    names = ["qwen_ctc_pred", "qwen_whisper_pred"]
    if path.exists():
        z = np.load(path)
        return {c: ([z[f"{c}__oof__{i}"] for i in range(len(ctx.seeds))],
                    [z[f"{c}__test__{i}"] for i in range(len(ctx.seeds))]) for c in names}
    out = {}
    for name, (X, Xt, model) in member_inputs(ctx).items():
        text = name.split("_")[1]
        oofs, tests = [], []
        for s in ctx.seeds:
            o, t = np.zeros(len(ctx.y)), np.zeros(len(Xt))
            for a, b in ctx.folds[s]:
                from common import fit_weighted
                m = fit_weighted(clone(model), X[a], yt[a], None if sw is None else sw[a], groups=ctx.groups[a])
                o[b] = m.predict(X[b])
                t += m.predict(Xt) / len(ctx.folds[s])
            oofs.append(o); tests.append(t)
        out[f"qwen_{text}_pred"] = (oofs, tests)
    np.savez(path, **{f"{c}__oof__{i}": o for c, (os_, _) in out.items() for i, o in enumerate(os_)},
             **{f"{c}__test__{i}": t for c, (_, ts) in out.items() for i, t in enumerate(ts)})
    return out
