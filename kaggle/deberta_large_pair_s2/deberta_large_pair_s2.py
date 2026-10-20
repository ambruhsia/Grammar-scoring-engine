# %% [markdown]
# # DeBERTa-v3-large on the (CTC raw, Whisper clean) pair, speaker-grouped folds (Phase I, 2026-10-15)
# The base model on this pair (`deberta_pair_pred`, 0.561 alone) carries the short-clip blend. Our first large run (v23) collapsed
# on 5 of 15 folds with the base recipe; this one uses the recipe from lijoraju's public write-up for large (lr 1e-5, layer-wise
# LR decay 0.9, 10% warm-up, 5 epochs, effective batch 16 = 4 x 4 accumulation with gradient checkpointing (v1 ran out of memory
# at batch 8), mean pooling, dropout off), but keeps the FINAL epoch of every fold
# (choosing the best validation epoch would tune on the validation fold). Collapse guard: a fold whose validation predictions are
# almost constant (sd < 0.15 of the training labels' sd; no label of the validation fold is used) is retrained with a new seed.
# Folds: the speaker-grouped seed-42 folds (folds_grouped.csv), like deberta_pair_pred. Seeds 42 / 7 / 2024, split over both T4s.
# DeBERTa-v3 is MIT-licensed. Transcripts come from the attached caches dataset; nothing leaves Kaggle.
# Writes /kaggle/working/deberta_large_pair.csv: split, filename, deberta_large_pair_pred (+ one column per seed).

# %%
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/kaggle/working")
SCRIPT = OUT / "train_dl.py"
SCRIPT.write_text(r'''
import gc, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, DataCollatorWithPadding, get_cosine_schedule_with_warmup

SEEDS = [int(s) for s in sys.argv[1].split(",")]
OUT = Path("/kaggle/working")
MODEL_NAME, N_EPOCHS, LR, LLRD, HEAD_LR, BS, ACCUM, WD, WARMUP = "microsoft/deberta-v3-large", 5, 1e-5, 0.9, 5e-5, 4, 4, 0.01, 0.10
CLIP, dev = (1.5, 5.0), "cuda"
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))

def find(name):
    hits = [Path(d) / name for d, _, files in os.walk("/kaggle/input", followlinks=True) if name in files]
    return min(hits, key=lambda p: len(p.parts))

def load_jsonl(p):
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line); out[(r["split"], r["filename"])] = r
        except json.JSONDecodeError:
            pass
    return out

W, C = load_jsonl(find("asr_c.jsonl")), load_jsonl(find("ctc.jsonl"))
train = pd.read_csv(find("train.csv")); train.columns = [c.strip().lower() for c in train.columns]
test = pd.read_csv(find("test.csv")); test.columns = [c.strip().lower() for c in test.columns]
FG = pd.read_csv(find("folds_grouped.csv")); FG = FG[FG.seed == 42]
T = train[train.label > 0].merge(FG[["filename", "fold"]], on="filename").reset_index(drop=True)
assert len(T) == 732
TE = test[["filename"]].reset_index(drop=True)
y = T.label.values.astype(np.float64)
folds = [(np.where(T.fold.values != k)[0], np.where(T.fold.values == k)[0]) for k in range(5)]
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
collate = DataCollatorWithPadding(tok)
pair = lambda s, f: (C[(s, f)]["raw_text"], W[(s, f)]["text"])
tr_txt, te_txt = [pair("train", f) for f in T.filename], [pair("test", f) for f in TE.filename]
lens = [len(tok(a, b)["input_ids"]) for a, b in tr_txt]
MAX_LEN = int(min(512, np.percentile(lens, 99) + 16))
print(f"seeds {SEEDS} GPU {torch.cuda.get_device_name(0)} tokens median {int(np.median(lens))} MAX_LEN {MAX_LEN}", flush=True)

class DS(Dataset):
    def __init__(self, pairs, yy=None): self.pairs, self.yy = pairs, yy
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        a, b = self.pairs[i]
        e = dict(tok(a, b, truncation=True, max_length=MAX_LEN)); e.pop("token_type_ids", None)
        if self.yy is not None: e["labels"] = float(self.yy[i])
        return e

class Reg(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = AutoConfig.from_pretrained(MODEL_NAME); cfg.hidden_dropout_prob = 0.0; cfg.attention_probs_dropout_prob = 0.0
        self.bb = AutoModel.from_pretrained(MODEL_NAME, config=cfg); self.head = nn.Linear(cfg.hidden_size, 1)
        self.bb.gradient_checkpointing_enable()   # v1 ran out of memory on a T4 at batch 8
        nn.init.zeros_(self.head.bias)
    def forward(self, input_ids, attention_mask):
        h = self.bb(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1e-6)).squeeze(-1)

def param_groups(model):
    n_layers = len(model.bb.encoder.layer)
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    groups = []
    for n, p in model.named_parameters():
        if n.startswith("head."):
            lr = HEAD_LR
        elif n.startswith("bb.embeddings."):
            lr = LR * LLRD ** n_layers
        elif n.startswith("bb.encoder.layer."):
            lr = LR * LLRD ** (n_layers - 1 - int(n.split(".")[3]))
        else:
            lr = LR
        groups.append({"params": [p], "lr": lr, "weight_decay": 0.0 if any(k in n for k in no_decay) else WD})
    return groups

def predict(model, dl):
    model.eval(); out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for b in dl: out.append(model(b["input_ids"].to(dev), b["attention_mask"].to(dev)).float().cpu().numpy())
    return np.concatenate(out)

def run_fold(seed, tr_i, va_i):
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = y[tr_i].mean(), y[tr_i].std(); yn = (y - mu) / sd
    g = torch.Generator(); g.manual_seed(seed)
    tr_dl = DataLoader(DS([tr_txt[i] for i in tr_i], yn[tr_i]), batch_size=BS, shuffle=True, collate_fn=collate, generator=g)
    va_dl = DataLoader(DS([tr_txt[i] for i in va_i]), batch_size=32, collate_fn=collate)
    te_dl = DataLoader(DS(te_txt), batch_size=32, collate_fn=collate)
    model = Reg().to(dev).float()
    opt = torch.optim.AdamW(param_groups(model))
    steps = N_EPOCHS * (len(tr_dl) // ACCUM)
    sch = get_cosine_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    scaler = torch.cuda.amp.GradScaler()
    for ep in range(N_EPOCHS):
        model.train()
        for it, b in enumerate(tr_dl):
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(b["input_ids"].to(dev), b["attention_mask"].to(dev))
            loss = nn.functional.mse_loss(pred.float(), b["labels"].float().to(dev)) / ACCUM
            scaler.scale(loss).backward()
            if (it + 1) % ACCUM == 0:
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step()
    va, te = predict(model, va_dl) * sd + mu, predict(model, te_dl) * sd + mu
    del model, opt, sch, scaler; gc.collect(); torch.cuda.empty_cache()
    return va, te, float(np.std(va) / y[tr_i].std())

for seed in SEEDS:
    cache_p = OUT / f"dl_pair_s{seed}.npz"
    cache = dict(np.load(cache_p)) if cache_p.exists() else {}
    t0 = time.time()
    for k, (a, b) in enumerate(folds):
        if f"f{k}_va" in cache: continue
        for attempt in range(3):
            va, te, spread = run_fold(seed + 1000 * attempt, a, b)
            ok = np.isfinite(va).all() and np.isfinite(te).all() and spread >= 0.15
            print(f"  seed {seed} fold {k} attempt {attempt}: val RMSE {rmse(y[b], np.clip(va, *CLIP)):.3f} "
                  f"pred spread {spread:.2f} {'ok' if ok else 'COLLAPSED -> retry'}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
            if ok: break
        cache[f"f{k}_va"], cache[f"f{k}_te"], cache[f"f{k}_att"] = va, te, np.array(attempt)
        np.savez(cache_p, **cache)
    o = np.zeros(len(y))
    for k, (a, b) in enumerate(folds): o[b] = cache[f"f{k}_va"]
    print(f"seed {seed}: OOF RMSE {rmse(y, np.clip(o, *CLIP)):.4f}", flush=True)
''')

# %%
import torch
from huggingface_hub import snapshot_download
snapshot_download("microsoft/deberta-v3-large")        # once, before two processes would race on the cache
n_gpu = torch.cuda.device_count()
assert n_gpu >= 1, "no CUDA device: the accelerator must be a GPU"
plan = ["11,33", "22"] if n_gpu >= 2 else ["11,22,33"]
procs = []
for g, seeds in enumerate(plan):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    log = open(OUT / f"train_gpu{g}.log", "w")
    procs.append((subprocess.Popen([sys.executable, str(SCRIPT), seeds], env=env, stdout=log, stderr=subprocess.STDOUT), log, g))
t0 = time.time()
while any(p.poll() is None for p, _, _ in procs):
    time.sleep(60)
    print(f"[{(time.time() - t0) / 60:.0f} min] " + " | ".join(
        (OUT / f"train_gpu{g}.log").read_text().strip().splitlines()[-1][:110] if (OUT / f"train_gpu{g}.log").stat().st_size else "-"
        for _, _, g in procs), flush=True)
for p, log, g in procs:
    log.close()
    print(f"--- GPU {g} exit {p.returncode}\n" + (OUT / f"train_gpu{g}.log").read_text()[-3000:])
assert all(p.returncode == 0 for p, _, _ in procs), "a training process failed"

# %%
def find(name):
    hits = [Path(d) / name for d, _, files in os.walk("/kaggle/input", followlinks=True) if name in files]
    return min(hits, key=lambda p: len(p.parts))


train = pd.read_csv(find("train.csv")); train.columns = [c.strip().lower() for c in train.columns]
test = pd.read_csv(find("test.csv")); test.columns = [c.strip().lower() for c in test.columns]
FG = pd.read_csv(find("folds_grouped.csv")); FG = FG[FG.seed == 42]
T = train[train.label > 0].merge(FG[["filename", "fold"]], on="filename").reset_index(drop=True)
folds = [(np.where(T.fold.values != k)[0], np.where(T.fold.values == k)[0]) for k in range(5)]
res = {"split": ["train"] * len(T) + ["test"] * len(test), "filename": list(T.filename) + list(test.filename)}
cols = []
for seed in (11, 22, 33):
    z = np.load(OUT / f"dl_pair_s{seed}.npz")
    o, t = np.zeros(len(T)), np.zeros(len(test))
    for k, (a, b) in enumerate(folds):
        o[b] = z[f"f{k}_va"]; t += z[f"f{k}_te"] / 5
    res[f"deberta_large_pair_s{seed}"] = list(o) + list(t); cols.append(f"deberta_large_pair_s{seed}")
    print(f"seed {seed}: OOF RMSE {np.sqrt(np.mean((np.clip(o, 1.5, 5) - T.label.values) ** 2)):.4f}, retried folds "
          f"{[k for k in range(5) if int(z[f'f{k}_att']) > 0]}")
R = pd.DataFrame(res)
R["deberta_large_pair_pred"] = R[cols].mean(1)
o = R.deberta_large_pair_pred.values[:len(T)]
print(f"deberta_large_pair_pred (3 seeds) OOF RMSE {np.sqrt(np.mean((np.clip(o, 1.5, 5) - T.label.values) ** 2)):.4f}")
R.to_csv(OUT / "deberta_large_pair_s2.csv", index=False)
print("done")
