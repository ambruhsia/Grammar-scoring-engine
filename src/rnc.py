"""Rank-N-Contrast member (new method, approved by Amrita 2026-10-13): a grade-ordered embedding that ignores speaker and topic.

Rank-N-Contrast (Zha et al., NeurIPS 2023) learns a representation in which clips are ordered by label distance: for an anchor i
and a positive j, the negatives are every clip at least as far from i in grade as j. Here the positives must come from a
DIFFERENT speaker group and a DIFFERENT topic, and same-speaker clips are left out entirely, so the embedding cannot lean on
voice or prompt. Settings are fixed in advance (no grid):
    inputs     Qwen hidden states (Whisper and CTC text) + the three pooled audio encoders; per source StandardScaler -> PCA 64,
               fitted inside the fold; concatenated (320 dims)
    head       MLP 320 -> 256 -> 32 (ReLU, dropout 0.2), L2-normalised; Adam lr 1e-3, weight decay 1e-3, 300 full-batch epochs;
               cosine similarity, temperature 0.2
    member     RidgeCV(logspace(-2, 3, 30)) on the 32-d embedding -> rnc_pred
Trained on ctx.y_fit (denoised under D3). OOF on ctx.folds (test = fold mean) and on the G11 topic folds; cached.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

PCA_DIM, HIDDEN, OUT_DIM = 64, 256, 32
EPOCHS, LR, WD, TAU, DROPOUT, SEED = 300, 1e-3, 1e-3, 0.2, 0.2, 0
VERSION = "rnc-v1"


def sources(ctx):
    """-> list of (X train, X test) per embedding source."""
    import leakfree
    import qwen_feats
    out = [(X, Xt) for X, Xt, _ in qwen_feats.member_inputs(ctx).values()]
    out += [(X, Xt) for c, (X, Xt, _) in leakfree.member_inputs(ctx).items() if c != "emb_pred"]
    return out


def topics(ctx):
    I = pd.read_csv(ctx.cache / "insight.csv")
    return I[I.split == "train"].set_index("filename").topic.loc[ctx.T.filename].values


def rnc_loss(z, y, grp, top):
    """Rank-N-Contrast with speaker- and topic-crossing positives; labels on a 0.5 grid."""
    n = len(y)
    S = z @ z.T / TAU
    d = (y[:, None] - y[None, :]).abs()
    bucket = torch.round(d / 0.5).long()
    other = (grp[:, None] != grp[None, :])                      # same-speaker clips are neither positives nor negatives
    pos = other & (top[:, None] != top[None, :])
    E = torch.exp(S) * other
    B = torch.zeros(n, int(bucket.max()) + 1).scatter_add_(1, bucket, E)
    C = torch.flip(torch.cumsum(torch.flip(B, [1]), 1), [1])    # sum of E over all k with distance >= bucket
    D = C.gather(1, bucket)
    return (-(S - torch.log(D + 1e-12)) * pos).sum() / pos.sum().clamp(min=1)


def fit_predict(srcs, y, grp, top, a, b, n_test):
    """Train the head on rows a; return ridge predictions for rows b and for the test rows."""
    torch.manual_seed(SEED)
    rng_state = np.random.get_state(); np.random.seed(SEED)
    parts_a, parts_b, parts_t = [], [], []
    for X, Xt in srcs:
        sc = StandardScaler().fit(X[a])
        pca = PCA(PCA_DIM, random_state=SEED).fit(sc.transform(X[a]))
        parts_a.append(pca.transform(sc.transform(X[a])))
        parts_b.append(pca.transform(sc.transform(X[b])))
        parts_t.append(pca.transform(sc.transform(Xt)))
    np.random.set_state(rng_state)
    Za, Zb, Zt = (np.hstack(p) for p in (parts_a, parts_b, parts_t))
    mu, sd = Za.mean(0), Za.std(0) + 1e-6
    Za, Zb, Zt = ((Z - mu) / sd for Z in (Za, Zb, Zt))
    head = torch.nn.Sequential(torch.nn.Linear(Za.shape[1], HIDDEN), torch.nn.ReLU(), torch.nn.Dropout(DROPOUT),
                               torch.nn.Linear(HIDDEN, OUT_DIM))
    opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WD)
    xa = torch.tensor(Za, dtype=torch.float32)
    ya = torch.tensor(y[a], dtype=torch.float32)
    ga, ta = torch.tensor(grp[a]), torch.tensor(top[a])
    for _ in range(EPOCHS):
        head.train()
        opt.zero_grad()
        loss = rnc_loss(torch.nn.functional.normalize(head(xa), dim=1), ya, ga, ta)
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        emb = lambda Z: torch.nn.functional.normalize(head(torch.tensor(Z, dtype=torch.float32)), dim=1).numpy()
        Ea, Eb, Et = emb(Za), emb(Zb), emb(Zt)
    r = RidgeCV(alphas=np.logspace(-2, 3, 30)).fit(Ea, y[a])
    return r.predict(Eb), r.predict(Et)


def _key(ctx, folds_list, y, tag):
    h = hashlib.sha256((VERSION + tag).encode())
    for folds in folds_list:
        for _, b in folds:
            h.update(np.asarray(b).tobytes())
    h.update(np.asarray(y, float).tobytes())
    return h.hexdigest()[:16]


def oof(ctx) -> tuple[list, list]:
    """-> (oof per seed, test per seed) on ctx.folds; cached."""
    y = np.asarray(ctx.y_fit, float)
    path = ctx.cache / f"rnc_{_key(ctx, [ctx.folds[s] for s in ctx.seeds], y, 'grouped')}.npz"
    if path.exists():
        z = np.load(path)
        return [z[f"oof{i}"] for i in range(len(ctx.seeds))], [z[f"test{i}"] for i in range(len(ctx.seeds))]
    srcs, top = sources(ctx), topics(ctx)
    oofs, tests = [], []
    for s in ctx.seeds:
        o, t = np.zeros(len(y)), np.zeros(len(ctx.TE))
        for a, b in ctx.folds[s]:
            pb, pt = fit_predict(srcs, y, ctx.groups, top, a, b, len(ctx.TE))
            o[b] = pb; t += pt / len(ctx.folds[s])
        oofs.append(o); tests.append(t)
        print(f"  rnc seed {s}: done", flush=True)
    np.savez(path, **{f"oof{i}": o for i, o in enumerate(oofs)}, **{f"test{i}": t for i, t in enumerate(tests)})
    return oofs, tests


def topic_oof(ctx, tf) -> np.ndarray:
    y = np.asarray(ctx.y_fit, float)
    path = ctx.cache / f"rnc_{_key(ctx, [tf], y, 'topic')}.npz"
    if path.exists():
        return np.load(path)["oof"]
    srcs, top = sources(ctx), topics(ctx)
    o = np.full(len(y), np.nan)
    for a, b in tf:
        o[b] = fit_predict(srcs, y, ctx.groups, top, a, b, len(ctx.TE))[0]
    np.savez(path, oof=o)
    return o
