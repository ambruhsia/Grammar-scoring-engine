# %% [markdown]
# # Laya's ModernBERT-large encoder fine-tuned on the (CTC raw, Whisper clean) pair (approved 2026-10-14)
# The recipe of `deberta_pair_pred` (mean pooling, linear head, target standardised per fold, final epoch, seeds 42 / 7 / 2024
# on the speaker-grouped seed-42 folds), with the stabilisers the DeBERTa-v3-large collapse called for, fixed before running:
# layer-wise lr decay (top layer 2e-5, x0.9 per layer below), head lr 1e-3, 20% warm-up, gradient clipping, micro-batches of 4
# with 2-step accumulation and gradient checkpointing. A fold "collapses" when its validation RMSE is >= 0.97 x the RMSE of
# predicting the training mean; if 2 of the first seed's 5 folds collapse, the job stops early to save GPU time.
# Backbone: convaiinnovations/laya (Apache-2.0), pinned revision; transcripts from the artefacts dataset stay on Kaggle.
# Writes /kaggle/working/laya_ft_pair.csv (laya_ft_pair_pred: OOF for train, mean of the 15 fold models for test) and
# laya_ft_folds.json (per-fold validation RMSE, baseline, collapse flag, minutes).

# %%
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("laya") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "laya==0.4.7"], check=True)

# %%
import copy
import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding, get_cosine_schedule_with_warmup

import laya

OUT = Path("/kaggle/working")
MODEL, REVISION = "convaiinnovations/laya", "7b928d828b7b0e022f929d9bd2e44165aa270148"
SEEDS, FOLD_SEED = (42, 7, 2024), 42
N_EPOCHS, LR_TOP, LR_DECAY, LR_HEAD, WD, WARMUP = 6, 2e-5, 0.9, 1e-3, 0.01, 0.20
MICRO_BS, ACCUM = 4, 2
COLLAPSE_RATIO, MAX_EARLY_COLLAPSES = 0.97, 2
CLIP = (1.5, 5.0)
dev = "cuda"
assert torch.cuda.is_available(), "no CUDA device: the accelerator must be a GPU"
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def find(name):
    hits = [Path(d) / name for d, _, files in os.walk("/kaggle/input", followlinks=True) if name in files]
    if not hits:
        raise FileNotFoundError(f"{name} not under /kaggle/input: attach the artefacts dataset")
    return min(hits, key=lambda p: len(p.parts))


def load_jsonl(p):
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            out[(r["split"], r["filename"])] = r
        except json.JSONDecodeError:
            pass
    return out


W, C = load_jsonl(find("asr_c.jsonl")), load_jsonl(find("ctc.jsonl"))
train, test = pd.read_csv(find("train.csv")), pd.read_csv(find("test.csv"))
FG = pd.read_csv(find("folds_grouped.csv"))
FG = FG[FG.seed == FOLD_SEED]
T = train[train.label > 0].merge(FG[["filename", "fold"]], on="filename").reset_index(drop=True)
assert len(T) == len(FG) == 732, (len(T), len(FG))
TE = test[["filename"]].reset_index(drop=True)
y = T.label.values.astype(np.float64)
folds = [(np.where(T.fold.values != k)[0], np.where(T.fold.values == k)[0]) for k in range(5)]
tr_txt = [(C[("train", f)]["raw_text"] or "", W[("train", f)]["text"] or "") for f in T.filename]
te_txt = [(C[("test", f)]["raw_text"] or "", W[("test", f)]["text"] or "") for f in TE.filename]
print(f"train {len(T)}  test {len(TE)}  fold sizes {[len(b) for _, b in folds]}  GPU {torch.cuda.get_device_name(0)}")

# %%
agent = laya.load(MODEL, device="cpu", revision=REVISION)
tok, ENC0 = agent.tok, agent.model.encoder          # Laya's ModernBERT-large backbone
ENC0.config.reference_compile = False                # no torch.compile inside training
del agent
gc.collect()
collate = DataCollatorWithPadding(tok)
lens = [len(tok(a, b)["input_ids"]) for a, b in tr_txt]
MAX_LEN = int(min(1024, np.percentile(lens, 99) + 16))
print(f"encoder {type(ENC0).__name__}, {len(ENC0.layers)} layers, hidden {ENC0.config.hidden_size}; pair tokens median "
      f"{int(np.median(lens))} p99 {int(np.percentile(lens, 99))} -> MAX_LEN {MAX_LEN}")


class DS(Dataset):
    def __init__(self, pairs, yv=None):
        self.pairs, self.yv = pairs, yv

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        a, b = self.pairs[i]
        e = dict(tok(a, b, truncation=True, max_length=MAX_LEN))
        e.pop("token_type_ids", None)
        if self.yv is not None:
            e["labels"] = float(self.yv[i])
        return e


class Reg(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = copy.deepcopy(ENC0)
        self.bb.gradient_checkpointing_enable()
        self.head = nn.Linear(self.bb.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        h = self.bb(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1e-6)).squeeze(-1)


def param_groups(model):
    """Layer-wise lr decay: the top encoder layer at LR_TOP, each layer below x LR_DECAY, embeddings lowest; head at LR_HEAD."""
    n = len(model.bb.layers)
    groups = [{"params": list(layer.parameters()), "lr": LR_TOP * LR_DECAY ** (n - 1 - i)} for i, layer in enumerate(model.bb.layers)]
    groups.append({"params": list(model.bb.embeddings.parameters()), "lr": LR_TOP * LR_DECAY ** n})
    rest = [p for name, p in model.bb.named_parameters() if not name.startswith(("layers.", "embeddings."))]
    groups.append({"params": rest, "lr": LR_TOP})
    groups.append({"params": list(model.head.parameters()), "lr": LR_HEAD})
    for g in groups:
        g["weight_decay"] = WD
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))
    return groups


def predict(model, pairs):
    model.eval()
    out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for b in DataLoader(DS(pairs), batch_size=16, collate_fn=collate):
            out.append(model(b["input_ids"].to(dev), b["attention_mask"].to(dev)).float().cpu().numpy())
    return np.concatenate(out)


def run_fold(seed, tr_i, va_i):
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = y[tr_i].mean(), y[tr_i].std()
    g = torch.Generator(); g.manual_seed(seed)
    tr_dl = DataLoader(DS([tr_txt[i] for i in tr_i], (y[tr_i] - mu) / sd), batch_size=MICRO_BS, shuffle=True,
                       collate_fn=collate, generator=g)
    model = Reg().to(dev).float()
    opt = torch.optim.AdamW(param_groups(model))
    steps = N_EPOCHS * math.ceil(len(tr_dl) / ACCUM)
    sch = get_cosine_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    for _ in range(N_EPOCHS):
        model.train()
        opt.zero_grad(set_to_none=True)
        for n, b in enumerate(tr_dl, 1):
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(b["input_ids"].to(dev), b["attention_mask"].to(dev))
            scaler.scale(nn.functional.mse_loss(pred.float(), b["labels"].float().to(dev)) / ACCUM).backward()
            if n % ACCUM == 0 or n == len(tr_dl):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sch.step()
                opt.zero_grad(set_to_none=True)
    va = predict(model, [tr_txt[i] for i in va_i]) * sd + mu
    te = predict(model, te_txt) * sd + mu
    assert np.isfinite(va).all() and np.isfinite(te).all(), "non-finite predictions"
    del model, opt, sch, scaler
    gc.collect(); torch.cuda.empty_cache()
    return va, te, rmse(y[va_i], np.full(len(va_i), mu))


# %%
oof, te_pred, log, t0 = np.zeros(len(y)), np.zeros(len(TE)), [], time.time()
for seed in SEEDS:
    oof_s, te_s = np.zeros(len(y)), np.zeros(len(TE))
    for k, (a, b) in enumerate(folds):
        va, tp, base = run_fold(seed, a, b)
        r = rmse(y[b], np.clip(va, *CLIP))
        log.append({"seed": seed, "fold": k, "val_rmse": r, "mean_baseline": base, "collapsed": bool(r >= COLLAPSE_RATIO * base),
                    "minutes": (time.time() - t0) / 60})
        print(f"  seed {seed:>4} fold {k}  val RMSE {r:.3f}  (predict-the-mean {base:.3f}){'  COLLAPSED' if log[-1]['collapsed'] else ''}"
              f"  [{log[-1]['minutes']:.1f} min]", flush=True)
        (OUT / "laya_ft_folds.json").write_text(json.dumps(log, indent=1))
        oof_s[b] = va
        te_s += tp / len(folds)
        if seed == SEEDS[0] and sum(e["collapsed"] for e in log) >= MAX_EARLY_COLLAPSES:
            raise RuntimeError(f"{MAX_EARLY_COLLAPSES} of the first seed's folds collapsed: unstable, stopping to save GPU time")
    print(f"  seed {seed}: OOF RMSE {rmse(y, np.clip(oof_s, *CLIP)):.4f}", flush=True)
    oof += oof_s / len(SEEDS)
    te_pred += te_s / len(SEEDS)
print(f"laya_ft_pair_pred (grouped folds, {len(SEEDS)} seeds) standalone OOF RMSE {rmse(y, np.clip(oof, *CLIP)):.4f}; "
      f"collapsed folds {sum(e['collapsed'] for e in log)}/{len(log)}; {(time.time() - t0) / 60:.1f} min")
pd.DataFrame({"split": ["train"] * len(T) + ["test"] * len(TE), "filename": list(T.filename) + list(TE.filename),
              "laya_ft_pair_pred": np.r_[oof, te_pred]}).to_csv(OUT / "laya_ft_pair.csv", index=False)
