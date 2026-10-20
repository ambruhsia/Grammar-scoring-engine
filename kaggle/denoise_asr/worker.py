"""Worker for shl-denoise-asr: each clip transcribed twice by the same Whisper, from the original and from the denoised audio.
usage (inside the Kaggle notebook): worker.py <part> <n parts>"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

PART, NPART = int(sys.argv[1]), int(sys.argv[2])
OUT = Path("/kaggle/working")
CLIPS = json.loads((OUT / "clips.json").read_text())[PART::NPART]
files = {}
for d, _, fs in os.walk("/kaggle/input", followlinks=True):
    for f in fs:
        if f.lower().endswith(".wav"):
            files[("test" if "test" in d.lower() else "train", f)] = Path(d) / f
dev = "cuda"
from speechbrain.inference.separation import SepformerSeparation  # noqa: E402

enh = SepformerSeparation.from_hparams(source="speechbrain/sepformer-dns4-16k-enhancement", savedir=str(OUT / f"sb_dns4_{PART}"),
                                       run_opts={"device": dev})
from transformers import pipeline  # noqa: E402

asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", torch_dtype=torch.float16, device=0)
SR = 16000


def snr_db(x):
    n, h = int(0.025 * SR), int(0.010 * SR)
    k = 1 + max(len(x) - n, 0) // h
    F = np.stack([x[i * h:i * h + n] for i in range(k) if len(x[i * h:i * h + n]) == n])
    db = 10 * np.log10(np.mean(F ** 2, 1) + 1e-12)
    return float(np.percentile(db, 95) - np.percentile(db, 10)), float(np.percentile(db, 10))


def denoise(x):
    """SepFormer DNS-4 enhancement on 8 s windows (7 s hop), 1 s linear cross-fade; peak matched to the original."""
    W, H = 8 * SR, 7 * SR
    if len(x) <= W:
        starts = [0]
    else:
        starts = list(range(0, len(x) - W + 1, H))
        if starts[-1] + W < len(x):
            starts.append(len(x) - W)
    out, wsum = np.zeros(len(x)), np.zeros(len(x))
    ramp = np.linspace(0, 1, SR)
    for s in starts:
        seg = x[s:s + W]
        with torch.inference_mode():
            y = enh.separate_batch(torch.from_numpy(seg).float()[None].to(dev))[0, :, 0].float().cpu().numpy()[:len(seg)]
        f = np.ones(len(seg))
        if s > 0:
            f[:SR] = ramp[:min(SR, len(seg))]
        if s + W < len(x):
            f[-SR:] = ramp[::-1][-min(SR, len(seg)):]
        out[s:s + len(seg)] += y * f
        wsum[s:s + len(seg)] += f
    y = out / np.maximum(wsum, 1e-8)
    return (y / (np.abs(y).max() + 1e-9) * np.abs(x).max()).astype(np.float32)


def transcribe(x):
    r = asr({"raw": x, "sampling_rate": SR}, chunk_length_s=30, batch_size=4,
            generate_kwargs={"language": "english", "task": "transcribe", "num_beams": 1, "do_sample": False})
    return r["text"].strip()


rows, t0 = [], time.time()
for i, (split, fn, b3) in enumerate(CLIPS):
    x, sr = sf.read(files[(split, fn)], dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    assert sr == SR, (fn, sr)
    y = denoise(x)
    s0, n0 = snr_db(x)
    s1, n1 = snr_db(y)
    rows.append({"split": split, "filename": fn, "b3": b3, "duration": len(x) / SR, "snr_orig": s0, "noise_orig": n0,
                 "snr_enh": s1, "noise_enh": n1, "text_orig": transcribe(x), "text_enh": transcribe(y)})
    if i == 0 or (i + 1) % 20 == 0:
        el = time.time() - t0
        print(f"  part {PART}: {i + 1}/{len(CLIPS)} {el / (i + 1):.1f} s/clip ETA {el / (i + 1) * (len(CLIPS) - i - 1) / 60:.1f} min"
              f" | snr {s0:.1f} -> {s1:.1f} dB", flush=True)
    if (i + 1) % 50 == 0:
        pd.DataFrame(rows).to_csv(OUT / f"denoise_part{PART}.csv", index=False)
pd.DataFrame(rows).to_csv(OUT / f"denoise_part{PART}.csv", index=False)
print(f"part {PART} done {(time.time() - t0) / 60:.1f} min", flush=True)
