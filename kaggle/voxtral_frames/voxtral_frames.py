# %% [markdown]
# # Voxtral-Mini-3B frame states and question-first views (Phase I, 2026-10-15)
# Two additions to the earlier Voxtral job (amritarajput54/shl-voxtral-features), both from other participants' public write-ups
# (Techy-A: attention pooling over Voxtral's frames on random test-length crops; lijoraju: question-first views); code written
# from scratch.
# Model: mistralai/Voxtral-Mini-3B-2507 (Apache-2.0), pinned revision, fp16 on one T4. Reads the competition audio; the model runs
# inside this session, so the audio stays on Kaggle. No labels are used.
# Writes /kaggle/working:
# - `voxtral_frames.npz`: view A (audio first, then the grammar question, exactly as the earlier job), the per-frame state over
#   the audio tokens that cover real audio (12.5 per second), averaged over LM layers 12-14, projected to 256 dimensions by a PCA
#   fitted on training frames only. `F` [sum of frames, 256] float16, `offsets`, `mu`, `V`, `evr`.
# - `voxtral_views.npz`: views B and C (a question BEFORE the audio, so the audio states depend on it): the mean over real audio
#   tokens of LM layers 6-18, `B` and `C` [clips, 13, 3072] float16 (same layout as voxtral_lm.npz `A`).
# - `voxtral_frames_info.csv`: per clip duration, token counts, finiteness.

# %%
import importlib.metadata
import subprocess
import sys


def version_tuple(name):
    try:
        return tuple(int(x) for x in importlib.metadata.version(name).split(".")[:2])
    except Exception:
        return (0, 0)


# mistral-common first, and nothing may import transformers before it is installed (see the earlier job).
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "mistral-common[audio]>=1.8.1"], check=True)
if version_tuple("transformers") < (4, 56):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.56,<6"], check=True)

# %%
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import transformers
from transformers import AutoProcessor, VoxtralForConditionalGeneration

OUT = Path("/kaggle/working")
MODEL, REVISION = "mistralai/Voxtral-Mini-3B-2507", "3060fe34b35ba5d44202ce9ff3c097642914f8f3"
SR, CHUNK_S, TOK_PER_S = 16000, 30.0, 12.5
PROMPT_A = "Listen to this spoken answer from an English speaking test. How accurate and complex is the speaker's grammar?"
PROMPT_B = ("You will hear a spoken answer from an English speaking test. Grade the speaker's grammar on this rubric. "
            "1: struggles with sentence structure, limited control of simple structures. "
            "2: simple structures with consistent basic errors, incomplete sentences. "
            "3: a decent grasp of either grammar or sentence structure, with errors in the other. "
            "4: strong control, occasional minor errors that do not impede understanding. "
            "5: high accuracy, complex structures handled well, self-corrections.")
PROMPT_C = ("Listen carefully for grammatical mistakes in this spoken answer: wrong verb forms or tenses, missing or wrong "
            "articles, subject-verb agreement errors, wrong prepositions, and sentences left incomplete.")
FRAME_LAYERS = [12, 13, 14]
VIEW_LAYERS = list(range(6, 19))
PCA_DIM, PCA_FIT_FRAMES = 256, 150_000
assert torch.cuda.is_available(), "no CUDA device: the accelerator must be a GPU"
dev = "cuda:0"
torch.manual_seed(42)
rng = np.random.default_rng(42)


def find_csv(name):
    hits = [Path(d) / name for d, _, files in os.walk("/kaggle/input", followlinks=True) if name in files]
    assert hits, f"{name} not found: add the competition data"
    return min(hits, key=lambda p: len(p.parts))


found = Counter(Path(d) for d, _, files in os.walk("/kaggle/input", followlinks=True) for f in files if f.lower().endswith(".wav"))
pick = lambda name: max(((n, d) for d, n in found.items() if d.name.lower() == name), default=(0, None))[1]
AUDIO = {"train": pick("train"), "test": pick("test")}
assert None not in AUDIO.values(), "competition audio not found: add the competition data"
train, test = pd.read_csv(find_csv("train.csv")), pd.read_csv(find_csv("test.csv"))
KEYS = [("train", f) for f in train.filename[train.label > 0]] + [("test", f) for f in test.filename]
print(f"{len(KEYS)} clips (labelled train + test); transformers {transformers.__version__}; GPU {torch.cuda.get_device_name(0)}")

# %%
proc = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
if type(proc.tokenizer).__name__ != "MistralCommonBackend":
    from transformers.tokenization_mistral_common import MistralCommonBackend
    proc.tokenizer = MistralCommonBackend.from_pretrained(MODEL, revision=REVISION)
assert type(proc.tokenizer).__name__ == "MistralCommonBackend", f"tokenizer is {type(proc.tokenizer).__name__}"
try:
    model = VoxtralForConditionalGeneration.from_pretrained(MODEL, revision=REVISION, dtype=torch.float16)
except TypeError:
    model = VoxtralForConditionalGeneration.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float16)
model = model.to(dev).eval()
audio_id = model.config.audio_token_id
HID = model.config.text_config.hidden_size
print(f"language model {model.config.text_config.num_hidden_layers} layers, hidden {HID}; audio token id {audio_id}")


def load(split, fn):
    y, sr = sf.read(str(AUDIO[split] / fn), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(1)
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    dur = len(y) / SR
    if dur > CHUNK_S and dur % CHUNK_S < 2.0:          # a near-empty last 30 s chunk would be almost all padding
        y = y[: int((dur - dur % CHUNK_S) * SR)]
        dur = len(y) / SR
    return y, dur


def run_view(path, dur, prompt, audio_first, layers):
    """-> (hidden states of `layers` at the real audio tokens [len(layers), T, HID] float32, index of the first audio token)."""
    content = [{"type": "audio", "path": str(path)}, {"type": "text", "text": prompt}]
    if not audio_first:
        content = content[::-1]
    inputs = proc.apply_chat_template([{"role": "user", "content": content}]).to(dev, dtype=torch.float16)
    hs = model(**inputs, output_hidden_states=True).hidden_states       # embeddings + one entry per layer
    ids = inputs["input_ids"][0]
    pos = torch.nonzero(ids == audio_id).squeeze(-1)
    j = torch.arange(len(pos), device=pos.device)
    chunk, within = j // int(CHUNK_S * TOK_PER_S), j % int(CHUNK_S * TOK_PER_S)
    keep = pos[within < torch.ceil(torch.clamp(dur - CHUNK_S * chunk, 0, CHUNK_S) * TOK_PER_S)]
    H = torch.stack([hs[l][0, keep] for l in layers]).float()
    first = int(pos[0])
    del hs
    return H, first, len(pos), len(keep)


# %%
n = len(KEYS)
frames = [None] * n
VB = np.full((n, len(VIEW_LAYERS), HID), np.nan, np.float16)
VC = np.full_like(VB, np.nan)
info = []
tmp = OUT / "_clip.wav"
t0 = time.time()
with torch.inference_mode():
    for i, (split, fn) in enumerate(KEYS):
        y, dur = load(split, fn)
        sf.write(tmp, y, SR)
        H, first_a, n_tok, n_real = run_view(tmp, dur, PROMPT_A, True, FRAME_LAYERS)
        fa = bool(torch.isfinite(H).all())
        frames[i] = H.mean(0).half().cpu().numpy() if fa else None
        row = {"split": split, "filename": fn, "duration": dur, "audio_tokens": n_tok, "real_tokens": n_real,
               "first_audio_A": first_a, "finite_A": fa}
        for name, prompt, store in (("B", PROMPT_B, VB), ("C", PROMPT_C, VC)):
            H, first, _, _ = run_view(tmp, dur, prompt, False, VIEW_LAYERS)
            ok = bool(torch.isfinite(H).all())
            if ok:
                store[i] = H.mean(1).half().cpu().numpy()
            row[f"first_audio_{name}"], row[f"finite_{name}"] = first, ok
            del H
        info.append(row)
        if i == 0:
            print(f"  first audio token: view A {row['first_audio_A']}, B {row['first_audio_B']}, C {row['first_audio_C']} "
                  f"(B and C must be well after A: the question comes first)")
            assert row["first_audio_B"] > row["first_audio_A"] + 20 and row["first_audio_C"] > row["first_audio_A"] + 20
        if (i + 1) % 100 == 0 or i == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{n}  {el / (i + 1):.2f} s/clip  ETA {el / (i + 1) * (n - i - 1) / 60:.1f} min", flush=True)
tmp.unlink(missing_ok=True)
I = pd.DataFrame(info)
print(f"forward passes done {(time.time() - t0) / 60:.1f} min; finite A {I.finite_A.mean():.1%} B {I.finite_B.mean():.1%} "
      f"C {I.finite_C.mean():.1%}; real tokens median {I.real_tokens.median():.0f}")

# %%
# PCA on training frames only (label-free), then every clip's frames projected.
del model
torch.cuda.empty_cache()
tr_idx = [i for i, (s, _) in enumerate(KEYS) if s == "train" and frames[i] is not None]
pool = np.concatenate([frames[i] for i in tr_idx]).astype(np.float32)
sel = rng.choice(len(pool), size=min(PCA_FIT_FRAMES, len(pool)), replace=False)
X = torch.from_numpy(pool[sel]).to(dev)
mu = X.mean(0, keepdim=True)
C = ((X - mu).T @ (X - mu)) / (len(X) - 1)
evals, evecs = torch.linalg.eigh(C.double())
order = torch.argsort(evals, descending=True)
V = evecs[:, order[:PCA_DIM]].float()
evr = (evals[order[:PCA_DIM]] / evals.sum()).cpu().numpy()
print(f"PCA on {len(X)} of {len(pool)} training frames: {PCA_DIM} components explain {evr.sum():.1%}")
del pool, X, C
lens = [0 if f is None else len(f) for f in frames]
offsets = np.concatenate([[0], np.cumsum(lens)])
F = np.zeros((offsets[-1], PCA_DIM), np.float16)
for i, f in enumerate(frames):
    if f is not None:
        F[offsets[i]:offsets[i + 1]] = ((torch.from_numpy(f.astype(np.float32)).to(dev) - mu) @ V).half().cpu().numpy()
keys = {"split": np.array([k[0] for k in KEYS]), "filename": np.array([k[1] for k in KEYS])}
np.savez_compressed(OUT / "voxtral_frames.npz", F=F, offsets=offsets, mu=mu.cpu().numpy().astype(np.float32),
                    V=V.cpu().numpy().astype(np.float16), evr=evr, layers=np.array(FRAME_LAYERS), **keys)
np.savez_compressed(OUT / "voxtral_views.npz", B=VB, C=VC, layers=np.array(VIEW_LAYERS), **keys)
I.to_csv(OUT / "voxtral_frames_info.csv", index=False)
print(f"done {(time.time() - t0) / 60:.1f} min; frames {F.shape}; files: "
      + ", ".join(f"{p.name} {p.stat().st_size / 1e6:.0f} MB" for p in sorted(OUT.glob('voxtral_*'))))
