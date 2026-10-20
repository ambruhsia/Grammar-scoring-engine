# %% [markdown]
# # Verbatim transcripts and the text members on them (approved idea #3, 2026-10-14)
# Whisper repairs grammar while transcribing, which removes the errors the score is about; our "raw" side so far is a CTC model
# whose transcripts keep the errors but also misrecognise words. Here Whisper-large-v3-turbo (MIT) transcribes every clip with a
# prompt full of fillers, restarts and grammar mistakes, so it keeps what the speaker actually said (other participants' public
# write-ups report the same prompt trick; this code is written from scratch). Three stages, each saved as it finishes:
# - A `asr_v.jsonl`: verbatim transcripts (text, segments, words with times), same layout as the cached Whisper file.
# - B `qwen_hidden_verbatim.npz`, `qwen_judge_verbatim.csv`: Qwen2.5-7B-Instruct (4-bit) rubric-prompt hidden states on the
#   verbatim text, exactly the recipe of our Qwen heads (last token + mean over transcript tokens, layers 16/20/24).
# - C `deberta_vpair.csv`: DeBERTa-v3-base fine-tuned on the pair (verbatim, Whisper clean), exactly the recipe of
#   `deberta_pair_pred` (speaker-grouped seed-42 folds, seeds 42/7/2024, 6 epochs), only the raw side of the pair changes.
# The audio and transcripts stay on Kaggle; all models run inside this session. No labels are used before stage C, and stage C
# uses them only out of fold.

# %%
import importlib.util
import subprocess
import sys

for pkg, mod in (("faster-whisper", "faster_whisper"), ("bitsandbytes", "bitsandbytes")):
    if importlib.util.find_spec(mod) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

# %%
import gc
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

OUT = Path("/kaggle/working")
SR = 16000
ASR_MODEL, ASR_REV = "dropbox-dash/faster-whisper-large-v3-turbo", "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
PROMPT = ("Um, so, uh, I- I think the, the school was, er, very big. He go to the market yesterday and, uh, buyed some "
          "vegetable. Like, my friend she don't, um... she doesn't like it.")
assert torch.cuda.is_available(), "no CUDA device: the accelerator must be a GPU"
torch.manual_seed(42)


def find(name):
    hits = [Path(d) / name for d, _, files in os.walk("/kaggle/input", followlinks=True) if name in files]
    assert hits, f"{name} not under /kaggle/input: attach the competition data and the artefacts dataset"
    return min(hits, key=lambda p: len(p.parts))


def load_jsonl(p):
    out = {}
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            out[(r["split"], r["filename"])] = r
        except json.JSONDecodeError:
            pass
    return out


found = Counter(Path(d) for d, _, files in os.walk("/kaggle/input", followlinks=True) for f in files if f.lower().endswith(".wav"))
pick = lambda name: max(((n, d) for d, n in found.items() if d.name.lower() == name), default=(0, None))[1]
AUDIO = {"train": pick("train"), "test": pick("test")}
assert None not in AUDIO.values(), "competition audio not found: add the competition data"
train, test = pd.read_csv(find("train.csv")), pd.read_csv(find("test.csv"))
KEYS = [("train", f) for f in train.filename[train.label > 0]] + [("test", f) for f in test.filename]
W = load_jsonl(find("asr_c.jsonl"))
print(f"{len(KEYS)} clips; GPU {torch.cuda.get_device_name(0)}")


def load16k(path):
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(1)
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(y, dtype=np.float32)


# %% [markdown]
# ## Stage A: verbatim transcription

# %%
from faster_whisper import WhisperModel

t0 = time.time()
from huggingface_hub import snapshot_download

asr = WhisperModel(snapshot_download(ASR_MODEL, revision=ASR_REV), device="cuda", compute_type="float16")
V = {}
with open(OUT / "asr_v.jsonl", "w", encoding="utf-8") as fh:
    for i, (split, fn) in enumerate(KEYS):
        audio = load16k(AUDIO[split] / fn)
        for wt in (True, False):        # faster-whisper's word alignment can crash on an odd clip: redo it without word times
            try:
                segs, info = asr.transcribe(audio, language="en", beam_size=5, word_timestamps=wt,
                                            condition_on_previous_text=True, initial_prompt=PROMPT, vad_filter=False)
                S, Wd = [], []
                for s in segs:
                    S.append(dict(start=round(s.start, 2), end=round(s.end, 2), text=s.text, avg_logprob=round(s.avg_logprob, 4),
                                  no_speech_prob=round(s.no_speech_prob, 4), compression_ratio=round(s.compression_ratio, 3)))
                    Wd += [[w.word, round(w.start, 2), round(w.end, 2), round(w.probability, 4)] for w in (s.words or [])]
                break
            except IndexError as exc:
                print(f"  {split}/{fn}: word alignment failed ({exc}); transcribing without word times", flush=True)
        r = dict(split=split, filename=fn, duration=round(info.duration, 2), text="".join(x["text"] for x in S).strip(),
                 segments=S, words=Wd, word_times=wt)
        V[(split, fn)] = r
        fh.write(json.dumps(r) + "\n")
        if (i + 1) % 100 == 0 or i == 0:
            el = time.time() - t0
            print(f"  asr {i + 1}/{len(KEYS)}  {el / (i + 1):.2f} s/clip  ETA {el / (i + 1) * (len(KEYS) - i - 1) / 60:.1f} min", flush=True)
del asr
gc.collect(); torch.cuda.empty_cache()
nv = np.array([len(V[k]["text"].split()) for k in KEYS]); nc = np.array([len((W[k]["text"] or "").split()) for k in KEYS])
leak = np.mean(["buyed" in V[k]["text"] for k in KEYS])
print(f"stage A done {(time.time() - t0) / 60:.1f} min; words per clip verbatim {np.median(nv):.0f} vs clean {np.median(nc):.0f}; "
      f"empty {np.mean(nv == 0):.1%}; prompt words leaked into the text {leak:.1%}")
print("example:\n  clean   :", W[KEYS[0]]["text"][:200], "\n  verbatim:", V[KEYS[0]]["text"][:200])

# %% [markdown]
# ## Stage B: Qwen2.5-7B rubric-prompt hidden states on the verbatim text (the recipe of our Qwen heads)

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

QMODEL, LAYERS, MAX_TEXT_TOKENS, BATCH_FWD = "Qwen/Qwen2.5-7B-Instruct", (16, 20, 24), 600, 8
RUBRIC = """1: The person's speech struggles with proper sentence structure and syntax, displaying limited control over simple grammatical structures and memorized sentence patterns.
2: The person has a limited understanding of sentence structure and syntax. Although they use simple structures, they consistently make basic sentence structure and grammatical mistakes. They might leave sentences incomplete.
3: The person demonstrates a decent grasp of sentence structure but makes errors in grammatical structure, or they show a decent grasp of grammatical structure but make errors in sentence syntax and structure.
4: The person displays a strong understanding of sentence structure and syntax. They consistently show good control of grammar. While occasional errors may occur, they are generally minor and do not lead to misunderstandings; the person can correct most of them.
5: Overall, the person showcases high grammatical accuracy and adept control of complex grammar. They use grammar accurately and effectively, seldom making noticeable mistakes. Additionally, they handle complex language structures well and correct themselves when necessary."""
t0 = time.time()
tok = AutoTokenizer.from_pretrained(QMODEL)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    QMODEL, device_map={"": 0}, torch_dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                                           bnb_4bit_use_double_quant=True)).eval()
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
DIGITS = [tok.convert_tokens_to_ids(str(d)) for d in range(1, 6)]


def chat_pieces(system, before, text, after):
    marker = "⁣TRANSCRIPT⁣"
    full = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": before + marker + after}],
                                   tokenize=False, add_generation_prompt=True)
    pre, post = full.split(marker)
    a = tok(pre, add_special_tokens=False).input_ids
    b = tok(text, add_special_tokens=False).input_ids[:MAX_TEXT_TOKENS]
    return a + b + tok(post, add_special_tokens=False).input_ids, (len(a), len(a) + len(b))


def left_pad(seqs):
    n = max(len(s) for s in seqs)
    ids, att = torch.full((len(seqs), n), PAD, dtype=torch.long), torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, n - len(s):] = torch.tensor(s); att[i, n - len(s):] = 1
    return ids.cuda(), att.cuda(), [n - len(s) for s in seqs]


BEFORE = f"Grammar rubric (score levels 1 to 5):\n{RUBRIC}\n\nAutomatic transcript of a spoken response:\n\"\"\""
AFTER = "\"\"\"\n\nRate the grammar of this response from 1 to 5 using the rubric. Answer with a single digit."
texts = [V[k]["text"] or "" for k in KEYS]
built = [chat_pieces("You are an expert assessor of spoken English grammar.", BEFORE, t, AFTER) for t in texts]
order = np.argsort([len(s) for s, _ in built])
last = np.zeros((len(texts), len(LAYERS), model.config.hidden_size), np.float16)
mean = np.zeros_like(last)
probs = np.zeros((len(texts), 5), np.float32)
with torch.inference_mode():
    for b0 in range(0, len(order), BATCH_FWD):
        idx = order[b0:b0 + BATCH_FWD]
        ids, att, pads = left_pad([built[i][0] for i in idx])
        out = model.model(input_ids=ids, attention_mask=att, output_hidden_states=True)
        p = torch.softmax(model.lm_head(out.last_hidden_state[:, -1, :]).float()[:, DIGITS], dim=-1).cpu().numpy()
        for j, i in enumerate(idx):
            lo, hi = built[i][1][0] + pads[j], built[i][1][1] + pads[j]
            for li, layer in enumerate(LAYERS):
                h = out.hidden_states[layer][j]
                last[i, li] = h[-1].float().cpu().numpy()
                mean[i, li] = (h[lo:hi].float().mean(0) if hi > lo else h[-1].float()).cpu().numpy()
            probs[i] = p[j]
        del out
keys = {"split": np.array([k[0] for k in KEYS]), "filename": np.array([k[1] for k in KEYS])}
np.savez_compressed(OUT / "qwen_hidden_verbatim.npz", layers=np.array(LAYERS), verbatim_last=last, verbatim_mean=mean, **keys)
pd.DataFrame({**{k: list(v) for k, v in keys.items()}, **{f"verbatim_p{d + 1}": probs[:, d] for d in range(5)}}).to_csv(
    OUT / "qwen_judge_verbatim.csv", index=False)
del model
gc.collect(); torch.cuda.empty_cache()
print(f"stage B done {(time.time() - t0) / 60:.1f} min")

# %% [markdown]
# ## Stage C: DeBERTa-v3-base on the (verbatim, Whisper clean) pair (the recipe of `deberta_pair_pred`)

# %%
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, DataCollatorWithPadding, get_cosine_schedule_with_warmup

DMODEL, N_EPOCHS, LR, BS, WD, WARMUP, SEEDS, CLIP = "microsoft/deberta-v3-base", 6, 2e-5, 8, 0.01, 0.10, (42, 7, 2024), (1.5, 5.0)
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))
FG = pd.read_csv(find("folds_grouped.csv"))
FG = FG[FG.seed == 42]
T = train[train.label > 0].merge(FG[["filename", "fold"]], on="filename").reset_index(drop=True)
assert len(T) == 732
TE = test[["filename"]].reset_index(drop=True)
y = T.label.values.astype(np.float64)
folds = [(np.where(T.fold.values != k)[0], np.where(T.fold.values == k)[0]) for k in range(5)]
dtok = AutoTokenizer.from_pretrained(DMODEL)
collate = DataCollatorWithPadding(dtok)
tr_txt = [(V[("train", f)]["text"], W[("train", f)]["text"]) for f in T.filename]
te_txt = [(V[("test", f)]["text"], W[("test", f)]["text"]) for f in TE.filename]
MAX_LEN = int(min(512, np.percentile([len(dtok(a, b)["input_ids"]) for a, b in tr_txt], 99) + 16))


class DS(Dataset):
    def __init__(self, pairs, yv=None):
        self.pairs, self.yv = pairs, yv

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        a, b = self.pairs[i]
        e = dict(dtok(a, b, truncation=True, max_length=MAX_LEN))
        e.pop("token_type_ids", None)
        if self.yv is not None:
            e["labels"] = float(self.yv[i])
        return e


class Reg(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = AutoConfig.from_pretrained(DMODEL)
        cfg.hidden_dropout_prob = cfg.attention_probs_dropout_prob = 0.0
        self.bb = AutoModel.from_pretrained(DMODEL, config=cfg)
        self.head = nn.Linear(cfg.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        h = self.bb(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1e-6)).squeeze(-1)


def predict(model, pairs):
    model.eval()
    out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for bt in DataLoader(DS(pairs), batch_size=32, collate_fn=collate):
            out.append(model(bt["input_ids"].cuda(), bt["attention_mask"].cuda()).float().cpu().numpy())
    return np.concatenate(out)


t0 = time.time()
oof, te_pred = np.zeros(len(y)), np.zeros(len(TE))
for seed in SEEDS:
    for k, (a, b) in enumerate(folds):
        torch.manual_seed(seed); np.random.seed(seed)
        mu, sd = y[a].mean(), y[a].std()
        g = torch.Generator(); g.manual_seed(seed)
        tr_dl = DataLoader(DS([tr_txt[i] for i in a], (y[a] - mu) / sd), batch_size=BS, shuffle=True, collate_fn=collate, generator=g)
        model = Reg().cuda().float()
        no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
        opt = torch.optim.AdamW([
            {"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)], "weight_decay": WD},
            {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)], "weight_decay": 0.0}], lr=LR)
        sch = get_cosine_schedule_with_warmup(opt, int(WARMUP * N_EPOCHS * len(tr_dl)), N_EPOCHS * len(tr_dl))
        scaler = torch.amp.GradScaler("cuda")
        for _ in range(N_EPOCHS):
            model.train()
            for bt in tr_dl:
                with torch.autocast("cuda", dtype=torch.float16):
                    pred = model(bt["input_ids"].cuda(), bt["attention_mask"].cuda())
                loss = nn.functional.mse_loss(pred.float(), bt["labels"].float().cuda())
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sch.step()
        va = predict(model, [tr_txt[i] for i in b]) * sd + mu
        oof[b] += va / len(SEEDS)
        te_pred += (predict(model, te_txt) * sd + mu) / (len(SEEDS) * len(folds))
        print(f"  deberta seed {seed} fold {k}: val RMSE {rmse(y[b], np.clip(va, *CLIP)):.3f}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
        del model, opt, sch, scaler
        gc.collect(); torch.cuda.empty_cache()
pd.DataFrame({"split": ["train"] * len(T) + ["test"] * len(TE), "filename": list(T.filename) + list(TE.filename),
              "deberta_vpair_pred": np.r_[oof, te_pred]}).to_csv(OUT / "deberta_vpair.csv", index=False)
print(f"stage C done {(time.time() - t0) / 60:.1f} min; deberta_vpair_pred alone OOF RMSE {rmse(y, np.clip(oof, *CLIP)):.4f}")
