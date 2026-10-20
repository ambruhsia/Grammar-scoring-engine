# %% [markdown]
# # First-30-s crop embeddings (PLAN_RUBRIC C4 / G0)
# Every clip's first 30 s (or the whole clip if shorter), so both recording batches are seen at the same length.
# - WavLM-large (CC BY-SA 3.0): all 25 hidden layers, mean + std over time -> `crop30_wavlm_large.npz`
# - Whisper-large-v3 encoder (Apache-2.0): all 33 hidden layers, mean + std over the valid (non-padded) frames
#   -> `crop30_whisper_large_v3.npz`
# float16, keyed by (split, filename). Ridge heads are fitted locally on speaker-grouped folds.

# %%
import math, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

OUT = Path("/kaggle/working")
SR, CROP_S = 16_000, 30.0
WAVLM, WHISPER = "microsoft/wavlm-large", "openai/whisper-large-v3"
dev = "cuda"
assert torch.cuda.is_available(), "no CUDA device: the accelerator must be a GPU"


def find(name):
    hits = [Path(d) / name for d, _, files in os.walk("/kaggle/input", followlinks=True) if name in files]
    if not hits:
        raise FileNotFoundError(name)
    return min(hits, key=lambda p: len(p.parts))


train = pd.read_csv(find("train.csv")); train.columns = [c.strip().lower() for c in train.columns]
test = pd.read_csv(find("test.csv")); test.columns = [c.strip().lower() for c in test.columns]
KEYS = [("train", f) for f in train[train.label > 0].filename] + [("test", f) for f in test.filename]
AUDIO = {}
for d, _, files in os.walk("/kaggle/input", followlinks=True):
    if any(f.endswith(".wav") for f in files) and Path(d).name.lower() in ("train", "test"):
        AUDIO[Path(d).name.lower()] = Path(d)
print(len(KEYS), "clips;", {k: str(v) for k, v in AUDIO.items()}, torch.cuda.get_device_name(0))


def load16k(path):
    import soundfile as sf, librosa
    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        y, sr = librosa.load(str(path), sr=None, mono=False)
        y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1 if y.shape[0] > y.shape[1] else 0)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(y[: int(CROP_S * SR)], dtype=np.float32)


def pool(hs, n_valid):
    h = torch.stack(hs)[:, 0, :n_valid].float()          # (layers, frames, dims)
    return torch.cat([h.mean(1), h.std(1)], -1).cpu().numpy().astype(np.float16)


def save(name, H):
    np.savez(OUT / name, split=np.array([k[0] for k in KEYS]), filename=np.array([k[1] for k in KEYS]), H=np.stack(H))
    print("saved", name, np.stack(H).shape)


# %%
from transformers import AutoFeatureExtractor, AutoModel

try:
    fe = AutoFeatureExtractor.from_pretrained(WAVLM)
except Exception:
    from transformers import Wav2Vec2FeatureExtractor
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SR, padding_value=0.0, do_normalize=True,
                                  return_attention_mask=True)
model = AutoModel.from_pretrained(WAVLM).to(dev).eval()
H, t0 = [], time.time()
for i, (s, f) in enumerate(KEYS, 1):
    a = load16k(AUDIO[s] / f)
    x = fe(a, sampling_rate=SR, return_tensors="pt").input_values.to(dev)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        hs = model(x, output_hidden_states=True).hidden_states
    H.append(pool(hs, hs[0].shape[1]))
    if i % 200 == 0:
        print(f"  wavlm {i}/{len(KEYS)} {(time.time() - t0) / 60:.1f} min", flush=True)
save("crop30_wavlm_large.npz", H)
del model; torch.cuda.empty_cache()

# %%
from transformers import WhisperFeatureExtractor, WhisperModel

fe = WhisperFeatureExtractor.from_pretrained(WHISPER)
enc = WhisperModel.from_pretrained(WHISPER, torch_dtype=torch.float16).get_encoder().to(dev).eval()
H, t0 = [], time.time()
for i, (s, f) in enumerate(KEYS, 1):
    a = load16k(AUDIO[s] / f)
    x = fe(a, sampling_rate=SR, return_tensors="pt").input_features.to(dev, torch.float16)
    n_valid = max(1, min(1500, math.ceil(len(a) / SR * 50)))
    with torch.inference_mode():
        hs = enc(x, output_hidden_states=True).hidden_states
    H.append(pool(hs, n_valid))
    if i % 200 == 0:
        print(f"  whisper {i}/{len(KEYS)} {(time.time() - t0) / 60:.1f} min", flush=True)
save("crop30_whisper_large_v3.npz", H)
print("done")
