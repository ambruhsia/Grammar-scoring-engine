# %% [markdown]
# # Recording-quality features per clip (CPU only, 2026-10-19; backlog: "an SNR estimate per clip")
# A noisy or band-limited recording degrades the transcripts and the speech embeddings, which can read as weak grammar. These
# label-free measures let the stack tell the recording from the speaker. Frame energy on 25 ms frames (10 ms hop):
#   aq_snr_db        95th minus 10th percentile of frame level (dB): speech level over the noise floor
#   aq_noise_db      10th percentile frame level (dBFS): the noise floor
#   aq_speech_db     95th percentile frame level (dBFS)
#   aq_active        share of frames more than 10 dB above the noise floor
#   aq_clip          share of samples at |x| >= 0.99 (clipping)
#   aq_flat_noise    spectral flatness of the quietest 10% of frames (white-ish noise -> 1, hum / tonal -> 0)
#   aq_rolloff_khz   95% spectral roll-off of the loud frames (recording bandwidth)
#   aq_centroid_khz  spectral centroid of the loud frames
#   aq_sr_khz        sample rate
# Reads the competition audio; nothing is learned and no labels are used. Writes /kaggle/working/audio_quality.csv.

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

OUT = Path("/kaggle/working")
files = [Path(d) / f for d, _, fs in os.walk("/kaggle/input", followlinks=True) for f in fs if f.lower().endswith(".wav")]
print(len(files), "wav files; folders:", sorted({str(p.parent) for p in files}))


def split_of(p: Path) -> str:
    parts = [s.lower() for s in p.parts]
    if any("test" in s for s in parts):
        return "test"
    if any("train" in s for s in parts):
        return "train"
    raise ValueError(p)


def frames(x, n, h):
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    k = 1 + (len(x) - n) // h
    return np.lib.stride_tricks.as_strided(x, (k, n), (x.strides[0] * h, x.strides[0]))


rows = []
for p in sorted(files):
    x, sr = sf.read(p, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    n, h = int(0.025 * sr), int(0.010 * sr)
    F = frames(x - x.mean(), n, h) * np.hanning(n)
    db = 10 * np.log10(np.mean(F ** 2, 1) + 1e-12)
    lo, hi = np.percentile(db, 10), np.percentile(db, 95)
    S = np.abs(np.fft.rfft(F, axis=1)) ** 2 + 1e-12
    f = np.fft.rfftfreq(n, 1 / sr)
    quiet, loud = db <= lo, db >= np.percentile(db, 70)
    flat = np.exp(np.mean(np.log(S[quiet]), 1)) / np.mean(S[quiet], 1)
    Sl = S[loud].mean(0)
    cum = np.cumsum(Sl) / Sl.sum()
    rows.append({"split": split_of(p), "filename": p.name, "aq_snr_db": hi - lo, "aq_noise_db": lo, "aq_speech_db": hi,
                 "aq_active": float(np.mean(db > lo + 10)), "aq_clip": float(np.mean(np.abs(x) >= 0.99)),
                 "aq_flat_noise": float(np.median(flat)), "aq_rolloff_khz": float(f[np.searchsorted(cum, 0.95)] / 1000),
                 "aq_centroid_khz": float((f * Sl).sum() / Sl.sum() / 1000), "aq_sr_khz": sr / 1000})
A = pd.DataFrame(rows)
assert not A.duplicated(["split", "filename"]).any(), "a (split, filename) occurs twice"
A.to_csv(OUT / "audio_quality.csv", index=False)
print(A.groupby("split").size().to_dict())
print(A.drop(columns=["filename"]).groupby("split").describe().T.round(3).to_string())
