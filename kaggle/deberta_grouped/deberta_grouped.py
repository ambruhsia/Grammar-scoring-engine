# %% [markdown]
# # DeBERTa-v3-base on speaker-grouped folds (PLAN_RUBRIC C2)
# Same recipe as the main notebook's section 7 (6 epochs, seeds 42/7/2024, final epoch only, target
# standardised per fold, mean pooling, dropout off, fp16 autocast with fp32 master weights), but the folds are
# the speaker-grouped seed-42 folds from `folds_grouped.csv`, so no speaker's twin sits on both sides of a split.
# - `deberta_pred`: Whisper text
# - `deberta_pair_pred`: the pair (CTC raw, Whisper clean). Whisper repairs grammar; the raw side keeps the errors.
# Every (input, seed, fold) is cached to /kaggle/working as it finishes. DeBERTa-v3 is MIT-licensed.

# %%
import gc, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, DataCollatorWithPadding, get_cosine_schedule_with_warmup

OUT = Path("/kaggle/working")
MODEL_NAME = "microsoft/deberta-v3-base"
RUNS = {"whisper": "deberta_pred", "ctc_pair": "deberta_pair_pred"}
N_EPOCHS, LR, BS, WD, WARMUP = 6, 2e-5, 8, 0.01, 0.10
SEEDS = (42, 7, 2024)
FOLD_SEED = 42                      # the grouped fold split (folds_grouped.csv, seed column)
USE_AMP = True
CLIP = (1.5, 5.0)
dev = "cuda"
assert torch.cuda.is_available(), "no CUDA device: the accelerator must be a GPU"
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def find(name):
    hits = [Path(d) / name for d, _, files in os.walk("/kaggle/input", followlinks=True) if name in files]
    if not hits:
        raise FileNotFoundError(f"{name} not under /kaggle/input")
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
train = pd.read_csv(find("train.csv"))
train.columns = [c.strip().lower() for c in train.columns]
test = pd.read_csv(find("test.csv"))
test.columns = [c.strip().lower() for c in test.columns]
FG = pd.read_csv(find("folds_grouped.csv"))
FG = FG[FG.seed == FOLD_SEED]
T = train[train.label > 0].merge(FG[["filename", "fold"]], on="filename").reset_index(drop=True)
assert len(T) == len(FG) == 732, (len(T), len(FG))
TE = test[["filename"]].reset_index(drop=True)
y = T.label.values.astype(np.float64)
folds = [(np.where(T.fold.values != k)[0], np.where(T.fold.values == k)[0]) for k in range(5)]
print(f"train {len(T)}  test {len(TE)}  fold sizes {[len(b) for _, b in folds]}  GPU {torch.cuda.get_device_name(0)}")

# %%
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
collate = DataCollatorWithPadding(tok)


def text_for(split, fn, inp):
    w, c = W[(split, fn)]["text"], C[(split, fn)]["raw_text"]
    return {"whisper": (w, None), "ctc_pair": (c, w)}[inp]


class DS(Dataset):
    def __init__(self, pairs, yy=None):
        self.pairs, self.yy = pairs, yy

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        a, b = self.pairs[i]
        e = dict(tok(a, b, truncation=True, max_length=MAX_LEN))
        e.pop("token_type_ids", None)
        if self.yy is not None:
            e["labels"] = float(self.yy[i])
        return e


class Reg(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = AutoConfig.from_pretrained(MODEL_NAME)
        cfg.hidden_dropout_prob = 0.0
        cfg.attention_probs_dropout_prob = 0.0
        self.bb = AutoModel.from_pretrained(MODEL_NAME, config=cfg)
        self.head = nn.Linear(cfg.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        h = self.bb(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1e-6)).squeeze(-1)


def predict(model, dl):
    model.eval()
    out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
        for b in dl:
            out.append(model(b["input_ids"].to(dev), b["attention_mask"].to(dev)).float().cpu().numpy())
    return np.concatenate(out)


def run_fold(seed, tr_i, va_i):
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = y[tr_i].mean(), y[tr_i].std()
    yn = (y - mu) / sd
    g = torch.Generator(); g.manual_seed(seed)
    tr_dl = DataLoader(DS([tr_txt[i] for i in tr_i], yn[tr_i]), batch_size=BS, shuffle=True, collate_fn=collate, generator=g)
    va_dl = DataLoader(DS([tr_txt[i] for i in va_i]), batch_size=32, collate_fn=collate)
    te_dl = DataLoader(DS(te_txt), batch_size=32, collate_fn=collate)
    model = Reg().to(dev).float()
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": WD},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0}], lr=LR)
    steps = N_EPOCHS * len(tr_dl)
    sch = get_cosine_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    curve = []
    for ep in range(N_EPOCHS):
        model.train()
        for b in tr_dl:
            with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
                pred = model(b["input_ids"].to(dev), b["attention_mask"].to(dev))
            loss = nn.functional.mse_loss(pred.float(), b["labels"].float().to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
        curve.append(rmse(y[va_i], np.clip(predict(model, va_dl) * sd + mu, *CLIP)))
    va = predict(model, va_dl) * sd + mu
    te = predict(model, te_dl) * sd + mu
    assert np.isfinite(va).all() and np.isfinite(te).all(), "non-finite predictions: set USE_AMP = False"
    del model, opt, sch, scaler
    gc.collect(); torch.cuda.empty_cache()
    return va, te, curve


# %%
result = {"split": ["train"] * len(T) + ["test"] * len(TE), "filename": list(T.filename) + list(TE.filename)}
for INPUT, COL in RUNS.items():
    tr_txt = [text_for("train", f, INPUT) for f in T.filename]
    te_txt = [text_for("test", f, INPUT) for f in TE.filename]
    lens = [len(tok(a, b)["input_ids"]) for a, b in tr_txt]
    MAX_LEN = int(min(512, np.percentile(lens, 99) + 16))
    print(f"\n[{INPUT}] -> {COL}   tokens median {int(np.median(lens))} p99 {int(np.percentile(lens, 99))} -> MAX_LEN {MAX_LEN}")
    CACHE = OUT / f"deberta_grouped_{MODEL_NAME.split('/')[-1]}_{INPUT}_e{N_EPOCHS}.npz"
    cache = dict(np.load(CACHE)) if CACHE.exists() else {}
    oof, te_pred, t0 = np.zeros(len(y)), np.zeros(len(TE)), time.time()
    for seed in SEEDS:
        oof_s, te_s = np.zeros(len(y)), np.zeros(len(TE))
        for k, (a, b) in enumerate(folds):
            kv, kt = f"s{seed}_f{k}_va", f"s{seed}_f{k}_te"
            if kv in cache:
                va, tp, note = cache[kv], cache[kt], "cached"
            else:
                va, tp, curve = run_fold(seed, a, b)
                cache[kv], cache[kt] = va, tp
                np.savez(CACHE, **cache)
                note = "val RMSE by epoch " + " ".join(f"{c:.3f}" for c in curve)
            print(f"  seed {seed:>4} fold {k}  {note}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
            oof_s[b] = va
            te_s += tp / len(folds)
        print(f"  seed {seed}: OOF RMSE {rmse(y, np.clip(oof_s, *CLIP)):.4f}")
        oof += oof_s / len(SEEDS)
        te_pred += te_s / len(SEEDS)
    print(f"{COL} (grouped folds) standalone OOF RMSE: {rmse(y, np.clip(oof, *CLIP)):.4f}")
    result[COL] = list(oof) + list(te_pred)
    pd.DataFrame(result).to_csv(OUT / "deberta_grouped.csv", index=False)
print("done")
