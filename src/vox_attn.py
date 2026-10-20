"""Attention pooling over Voxtral frames, trained on random test-length crops (Phase I; idea from Techy-A's public write-up,
code written from scratch).

Input: cache/voxtral/voxtral_frames.npz (kaggle/voxtral_frames job): per clip, Voxtral-Mini-3B's state at every audio token that
covers real audio (12.5 per second; audio first, then the grammar question, so the states depend on the audio alone), averaged
over LM layers 12-14 and projected to 256 dimensions by a PCA fitted on training frames (label-free).
Voxtral reads causally, so the first L seconds of a clip's frames are (almost exactly) the frames of an L-second recording: a
crop costs nothing. Settings fixed in advance (no grid):
    head      Linear 256 -> 128 + GELU (dropout 0.1); attention weights from Linear 128 -> 1 over time; pooled = [attention-weighted
              mean, plain mean, std] (384) -> dropout 0.2 -> Linear -> score (+ the training-fold mean)
    training  AdamW lr 1e-3, weight decay 1e-2, 40 epochs, batch 32, MSE on ctx.y_fit; every epoch each training clip is cut to
              a prefix whose length is drawn from the TEST clips' durations (a clip shorter than the draw stays whole)
    predict   full clips; 2 initialisations per fold (the cached run was trained on a T4: kaggle/vox_attn_gpu)
OOF on ctx.folds (test = fold mean) and on the G11 topic folds; cached.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import torch

FPS, HID, EPOCHS, BATCH, LR, WD, INITS = 12.5, 128, 40, 32, 1e-3, 1e-2, 2
VERSION = "vox-attn-v1-gpu2"   # trained on a T4 by kaggle/vox_attn_gpu (same code, CUDA)
torch.set_num_threads(8)


class Head(torch.nn.Module):
    def __init__(self, d, mean_y):
        super().__init__()
        self.proj = torch.nn.Sequential(torch.nn.Linear(d, HID), torch.nn.GELU(), torch.nn.Dropout(0.1))
        self.att = torch.nn.Linear(HID, 1)
        self.out = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(3 * HID, 1))
        self.mean_y = float(mean_y)

    def forward(self, x, mask):
        h = self.proj(x)
        a = self.att(h).squeeze(-1).masked_fill(~mask, -1e9)
        alpha = torch.softmax(a, 1).unsqueeze(-1)
        m = mask.float().unsqueeze(-1)
        n = m.sum(1).clamp(min=1)
        mean = (h * m).sum(1) / n
        std = torch.sqrt((((h - mean.unsqueeze(1)) ** 2) * m).sum(1) / n + 1e-6)
        return self.out(torch.cat([(alpha * h).sum(1), mean, std], 1)).squeeze(-1) + self.mean_y


def load(ctx):
    z = np.load(ctx.cache / "voxtral" / "voxtral_frames.npz")
    idx = {k: i for i, k in enumerate(zip(z["split"], z["filename"]))}
    F, off = z["F"], z["offsets"]
    get = lambda i: F[off[i]:off[i + 1]].astype(np.float32)
    tr = [get(idx[("train", f)]) for f in ctx.T.filename]
    te = [get(idx[("test", f)]) for f in ctx.TE.filename]
    G = pd.read_csv(ctx.cache / "speaker_groups.csv")
    durs = G[G.split == "test"].duration.values
    return tr, te, durs


def batches(clips, idx, rng, crop_durs=None):
    order = rng.permutation(idx) if crop_durs is not None else np.asarray(idx)
    for s in range(0, len(order), BATCH):
        rows = order[s:s + BATCH]
        xs = []
        for r in rows:
            x = clips[r]
            if crop_durs is not None:
                L = int(round(rng.choice(crop_durs) * FPS))
                x = x[:max(L, 8)]
            xs.append(x)
        T = max(len(x) for x in xs)
        X = np.zeros((len(xs), T, xs[0].shape[1]), np.float32)
        M = np.zeros((len(xs), T), bool)
        for i, x in enumerate(xs):
            X[i, :len(x)], M[i, :len(x)] = x, True
        yield rows, torch.from_numpy(X), torch.from_numpy(M)


def fit_predict(tr, te, y, a, b, durs, seed):
    n = sum(len(tr[i]) for i in a)
    mu = sum(tr[i].sum(0, dtype=np.float64) for i in a) / n
    sd = np.sqrt(sum(((tr[i] - mu) ** 2).sum(0, dtype=np.float64) for i in a) / n) + 1e-6
    mu, sd = mu.astype(np.float32), sd.astype(np.float32)
    norm = lambda x: (x - mu) / sd
    trn, ten = [norm(x) for x in tr], [norm(x) for x in te]
    pb, pt = np.zeros(len(b)), np.zeros(len(te))
    for k in range(INITS):
        torch.manual_seed(seed * 100 + k)
        rng = np.random.default_rng(seed * 100 + k)
        head = Head(trn[0].shape[1], y[a].mean())
        opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1 - e / EPOCHS)
        yt = torch.tensor(y, dtype=torch.float32)
        for _ in range(EPOCHS):
            head.train()
            for rows, X, M in batches(trn, a, rng, durs):
                opt.zero_grad()
                loss = torch.mean((head(X, M) - yt[rows]) ** 2)
                loss.backward()
                opt.step()
            sched.step()
        head.eval()
        with torch.no_grad():
            pb += np.concatenate([head(X, M).numpy() for _, X, M in batches(trn, b, rng)]) / INITS
            pt += np.concatenate([head(X, M).numpy() for _, X, M in batches(ten, np.arange(len(ten)), rng)]) / INITS
    return pb, pt


def _key(folds_list, y, tag):
    h = hashlib.sha256((VERSION + tag).encode())
    for folds in folds_list:
        for _, b in folds:
            h.update(np.asarray(b).tobytes())
    h.update(np.asarray(y, float).tobytes())
    return h.hexdigest()[:16]


def oof(ctx) -> tuple[list, list]:
    y = np.asarray(ctx.y_fit, float)
    path = ctx.cache / f"voxattn_{_key([ctx.folds[s] for s in ctx.seeds], y, 'grouped')}.npz"
    if path.exists():
        z = np.load(path)
        return [z[f"oof{i}"] for i in range(len(ctx.seeds))], [z[f"test{i}"] for i in range(len(ctx.seeds))]
    tr, te, durs = load(ctx)
    oofs, tests = [], []
    for s in ctx.seeds:
        o, t = np.zeros(len(y)), np.zeros(len(te))
        for k, (a, b) in enumerate(ctx.folds[s]):
            pb, pt = fit_predict(tr, te, y, a, b, durs, seed=s + k)
            o[b] = pb; t += pt / len(ctx.folds[s])
        oofs.append(o); tests.append(t)
        print(f"  vox_attn seed {s}: OOF RMSE {np.sqrt(np.mean((np.clip(o, 1.5, 5) - ctx.y) ** 2)):.4f}", flush=True)
    np.savez(path, **{f"oof{i}": o for i, o in enumerate(oofs)}, **{f"test{i}": t for i, t in enumerate(tests)})
    return oofs, tests


def topic_oof(ctx, tf) -> np.ndarray:
    y = np.asarray(ctx.y_fit, float)
    path = ctx.cache / f"voxattn_{_key([tf], y, 'topic')}.npz"
    if path.exists():
        return np.load(path)["oof"]
    tr, te, durs = load(ctx)
    o = np.full(len(y), np.nan)
    for k, (a, b) in enumerate(tf):
        o[b] = fit_predict(tr, te, y, a, b, durs, seed=1000 + k)[0]
    np.savez(path, oof=o)
    return o
