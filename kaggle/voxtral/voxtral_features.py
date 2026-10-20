# %% [markdown]
# # Voxtral-Mini-3B audio-LLM features for SHL grammar scoring (approved 2026-10-14)
# An audio language model "listens" to each clip together with a grammar question; its hidden states over the audio tokens
# become features for ridge heads fitted locally on speaker-grouped folds. The idea of using Voxtral this way comes from other
# participants' public write-ups (AMITABHSHUKLA, Techy-A, PRINCE2-AI); this code is written from scratch.
# Model: mistralai/Voxtral-Mini-3B-2507 (Apache-2.0), pinned revision, fp16 on one T4. Reads the competition audio; the model
# runs inside this session, so the audio stays on Kaggle. No labels are used.
# Writes /kaggle/working:
# - `voxtral_lm.npz`: language-model layers 6-18 (the lower-middle band, 20-60% of 30 layers): the mean over the audio tokens
#   that cover real audio (`A`), and the last prompt token's state (`last`), float16.
# - `voxtral_enc.npz`: the audio encoder (a Whisper-large-v3 encoder) layers 16-32: mean and std over real frames, float16.

# %%
import importlib.metadata
import importlib.util
import subprocess
import sys


def version_tuple(name):
    try:
        return tuple(int(x) for x in importlib.metadata.version(name).split(".")[:2])
    except Exception:
        return (0, 0)


# mistral-common first, and nothing may import transformers before it is installed: transformers decides at import time
# whether Voxtral gets Mistral's tokenizer backend (with the chat template) or a generic one without it.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "mistral-common[audio]>=1.8.1"], check=True)  # with audio extras
if version_tuple("transformers") < (4, 56):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.56,<6"], check=True)

# %%
import math
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
SR, CHUNK_S, TOK_PER_S, FRAMES_PER_S = 16000, 30.0, 12.5, 50
PROMPT = "Listen to this spoken answer from an English speaking test. How accurate and complex is the speaker's grammar?"
ENC_LAYERS = list(range(16, 33))           # audio encoder layers (1-based outputs of layers 16..32)
assert torch.cuda.is_available(), "no CUDA device: the accelerator must be a GPU"
dev = "cuda:0"
torch.manual_seed(42)


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
L = model.config.text_config.num_hidden_layers
LM_LAYERS = list(range(int(0.2 * L), int(0.6 * L) + 1))
encoder = next(m for m in model.modules() if type(m).__name__ == "VoxtralEncoder")
print(f"language model {L} layers, using {LM_LAYERS[0]}-{LM_LAYERS[-1]}; audio encoder {len(encoder.layers)} layers, "
      f"using {ENC_LAYERS[0]}-{ENC_LAYERS[-1]}; audio token id {audio_id}")

enc_out = {}


def make_hook(i):
    def hook(module, args, output):
        enc_out[i] = (output[0] if isinstance(output, tuple) else output).detach()
    return hook


handles = [encoder.layers[i - 1].register_forward_hook(make_hook(i)) for i in ENC_LAYERS]

# %%
n = len(KEYS)
A = np.full((n, len(LM_LAYERS), model.config.text_config.hidden_size), np.nan, np.float16)
LAST = np.full_like(A, np.nan)
ENC = np.full((n, len(ENC_LAYERS), 2 * encoder.config.hidden_size), np.nan, np.float16)
info = []
tmp = OUT / "_clip.wav"
t0 = time.time()
with torch.inference_mode():
    for i, (split, fn) in enumerate(KEYS):
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
        sf.write(tmp, y, SR)
        conv = [{"role": "user", "content": [{"type": "audio", "path": str(tmp)}, {"type": "text", "text": PROMPT}]}]
        inputs = proc.apply_chat_template(conv).to(dev, dtype=torch.float16)
        enc_out.clear()
        hs = model(**inputs, output_hidden_states=True).hidden_states       # embeddings + one entry per layer
        ids = inputs["input_ids"][0]
        pos = torch.nonzero(ids == audio_id).squeeze(-1)
        j = torch.arange(len(pos), device=pos.device)
        chunk, within = j // int(CHUNK_S * TOK_PER_S), j % int(CHUNK_S * TOK_PER_S)
        keep = pos[within < torch.ceil(torch.clamp(dur - CHUNK_S * chunk, 0, CHUNK_S) * TOK_PER_S)]
        H = torch.stack([hs[l][0] for l in LM_LAYERS]).float()               # [layers, T, hidden]
        finite = bool(torch.isfinite(H).all())
        if finite:
            A[i] = H[:, keep].mean(1).half().cpu().numpy()
            LAST[i] = H[:, -1].half().cpu().numpy()
        enc_ok = len(enc_out) == len(ENC_LAYERS)
        if enc_ok:
            n_chunks = enc_out[ENC_LAYERS[0]].shape[0]
            real = [min(enc_out[ENC_LAYERS[0]].shape[1], math.ceil(max(0.0, min(dur - CHUNK_S * c, CHUNK_S)) * FRAMES_PER_S))
                    for c in range(n_chunks)]
            for li, l in enumerate(ENC_LAYERS):
                E = torch.cat([enc_out[l][c, :real[c]] for c in range(n_chunks) if real[c] > 0]).float()
                if torch.isfinite(E).all():
                    ENC[i, li] = torch.cat([E.mean(0), E.std(0)]).half().cpu().numpy()
        info.append({"split": split, "filename": fn, "duration": dur, "audio_tokens": len(pos), "real_tokens": len(keep),
                     "lm_finite": finite, "enc_ok": enc_ok})
        del hs, H
        if (i + 1) % 100 == 0 or i == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{n}  {el / (i + 1):.2f} s/clip  ETA {el / (i + 1) * (n - i - 1) / 60:.1f} min  "
                  f"non-finite so far {sum(not r['lm_finite'] for r in info)}", flush=True)
for h in handles:
    h.remove()
tmp.unlink(missing_ok=True)

keys = {"split": np.array([k[0] for k in KEYS]), "filename": np.array([k[1] for k in KEYS])}
np.savez_compressed(OUT / "voxtral_lm.npz", layers=np.array(LM_LAYERS), A=A, last=LAST, **keys)
np.savez_compressed(OUT / "voxtral_enc.npz", layers=np.array(ENC_LAYERS), H=ENC, **keys)
pd.DataFrame(info).to_csv(OUT / "voxtral_info.csv", index=False)
I = pd.DataFrame(info)
print(f"done {(time.time() - t0) / 60:.1f} min; LM states finite {I.lm_finite.mean():.1%}; encoder states {I.enc_ok.mean():.1%}; "
      f"real audio tokens median {I.real_tokens.median():.0f}")
