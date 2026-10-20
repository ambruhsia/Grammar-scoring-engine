# %% [markdown]
# # Attention pooling over Voxtral frames on random test-length crops, trained on a T4 (Phase I, 2026-10-15)
# The member of src/vox_attn.py (idea from Techy-A's public write-up; code from scratch), unchanged except: CUDA, and 2
# initialisations per fold. Input frames: the shl-voxtral-frames job's output (view A, LM layers 12-14, PCA 256).
# Embedded below (from the local loop, so the folds and target match the stack exactly): per training clip the label, the
# denoised target (D3, lambda 0.5), speaker group, topic and the speaker-grouped fold for seeds 42 / 1337 / 2024; the test order;
# the test durations the crops are drawn from. Writes /kaggle/working/voxattn_gpu.npz (oof/test per seed, topic OOF).

# %%
import io
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TRAIN_CSV = """<per-clip table (filename, label, fold assignments) removed for the public copy: rebuilt from the competition data by src/vox_attn.py>"""
TEST_CSV = """<per-clip table (filename, label, fold assignments) removed for the public copy: rebuilt from the competition data by src/vox_attn.py>"""
DURS = np.array([45.12,45.06,45.06,45.06,42.54,59.94,60.52,41.72,60.06,37.92,6.48,58.98,59.76,45.06,60.06,60.06,43.02,45.06,45.06,45.06,45.06,45.06,45.06,45.06,38.49,49.2,45.06,45.06,45.06,45.06,45.06,45.0,45.0,45.06,45.06,45.06,45.06,45.0,45.06,45.06,45.06,43.62,42.18,45.06,45.06,42.84,45.06,45.06,44.76,45.06,45.06,45.06,45.06,41.34,45.06,45.06,42.12,45.06,45.0,43.14,44.82,45.06,45.06,44.86,44.97,44.93,45.06,45.06,45.06,45.06,45.06,43.14,41.94,44.22,41.22,45.06,45.06,45.06,45.06,45.06,45.06,45.06,41.58,45.06,45.06,45.06,44.92,45.06,45.16,45.06,45.06,45.06,44.7,45.06,45.06,45.06,45.06,45.06,45.06,45.06,45.06,45.01,44.64,45.06,45.06,44.4,45.06,45.0,45.06,45.06,45.2,45.06,45.06,45.06,44.92,45.99,45.0,45.0,45.0,44.84,45.04,45.28,45.06,45.06,45.06,45.72,44.92,45.06,45.06,46.01,44.84,44.82,46.01,45.06,46.01,44.32,60.07,18.56,30.02,45.0,60.07,60.07,52.7,51.14,60.08,60.08,60.08,60.08,60.08,60.08,41.42,60.08,60.07,54.61,60.54,57.07,60.5,60.61,57.26,28.27,60.07,36.72,32.96,60.07,60.07,60.07,60.07,60.07,60.07,60.07,60.07,60.07,45.03,60.07,60.07,61.03,60.07,60.07,60.07,60.07,60.07,60.07,60.07,45.0,52.69,49.92,60.07,60.07,60.07,60.07,60.07,60.07,60.07,60.01,57.52,56.11,60.01,60.07,60.07,60.07,60.07,60.07,60.07,60.07,60.07,60.0,44.99,43.32,45.06,45.06,45.06,45.06,41.4,45.06,45.06,45.06])
FPS, HID, EPOCHS, BATCH, LR, WD, INITS = 12.5, 128, 40, 32, 1e-3, 1e-2, 2
dev = "cuda"
assert torch.cuda.is_available(), "no CUDA device: the accelerator must be a GPU"
D, TE = pd.read_csv(io.StringIO(TRAIN_CSV)), pd.read_csv(io.StringIO(TEST_CSV))
hits = [Path(d) / "voxtral_frames.npz" for d, _, f in os.walk("/kaggle/input", followlinks=True) if "voxtral_frames.npz" in f]
z = np.load(hits[0])
idx = {k: i for i, k in enumerate(zip(z["split"], z["filename"]))}
F, off = z["F"], z["offsets"]
get = lambda i: F[off[i]:off[i + 1]].astype(np.float32)
tr = [get(idx[("train", f)]) for f in D.filename]
te = [get(idx[("test", f)]) for f in TE.filename]
y_fit, y = D.y_fit.values, D.y.values
print(f"{len(tr)} train / {len(te)} test clips; frames dim {tr[0].shape[1]}; GPU {torch.cuda.get_device_name(0)}")


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
        yield rows, torch.from_numpy(X).to(dev), torch.from_numpy(M).to(dev)


def fit_predict(a, b, seed):
    n = sum(len(tr[i]) for i in a)
    mu = sum(tr[i].sum(0, dtype=np.float64) for i in a) / n
    sd = np.sqrt(sum(((tr[i] - mu) ** 2).sum(0, dtype=np.float64) for i in a) / n) + 1e-6
    mu, sd = mu.astype(np.float32), sd.astype(np.float32)
    trn, ten = [(x - mu) / sd for x in tr], [(x - mu) / sd for x in te]
    pb, pt = np.zeros(len(b)), np.zeros(len(te))
    yt = torch.tensor(y_fit, dtype=torch.float32, device=dev)
    for k in range(INITS):
        torch.manual_seed(seed * 100 + k)
        rng = np.random.default_rng(seed * 100 + k)
        head = Head(trn[0].shape[1], y_fit[a].mean()).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1 - e / EPOCHS)
        for _ in range(EPOCHS):
            head.train()
            for rows, X, M in batches(trn, a, rng, DURS):
                opt.zero_grad()
                loss = torch.mean((head(X, M) - yt[torch.as_tensor(rows, device=dev)]) ** 2)
                loss.backward()
                opt.step()
            sched.step()
        head.eval()
        with torch.no_grad():
            pb += np.concatenate([head(X, M).cpu().numpy() for _, X, M in batches(trn, b, rng)]) / INITS
            pt += np.concatenate([head(X, M).cpu().numpy() for _, X, M in batches(ten, np.arange(len(ten)), rng)]) / INITS
    return pb, pt


# %%
res, t0 = {}, time.time()
for i, s in enumerate((42, 1337, 2024)):
    fs = D[f"fold{s}"].values
    o, t = np.zeros(len(y)), np.zeros(len(te))
    for k in range(5):
        a, b = np.where(fs != k)[0], np.where(fs == k)[0]
        pb, pt = fit_predict(a, b, seed=s + k)
        o[b] = pb; t += pt / 5
    res[f"oof{i}"], res[f"test{i}"] = o, t
    print(f"seed {s}: OOF RMSE {np.sqrt(np.mean((np.clip(o, 1.5, 5) - y) ** 2)):.4f}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
top, grp = D.topic.values, D.group.values
ot = np.full(len(y), np.nan)
for k, T in enumerate(sorted(set(top))):
    b = np.where(top == T)[0]
    a = np.where((top != T) & ~np.isin(grp, np.unique(grp[b])))[0]
    ot[b] = fit_predict(a, b, seed=1000 + k)[0]
res["topic_oof"] = ot
print(f"topic-held-out OOF RMSE {np.sqrt(np.mean((np.clip(ot, 1.5, 5) - y) ** 2)):.4f}  [{(time.time() - t0) / 60:.1f} min]")
np.savez("/kaggle/working/voxattn_gpu.npz", **res)
print("done")
