"""Zero-label guard (PLAN_RUBRIC A7): do any test clips look like the 37 static-noise sentinels?

The sentinels (train IDs 5037-5073, label 0) were dropped before transcription, so no cached signal
exists for them. This reads the raw audio instead: the sentinels, a random sample of normal train clips
and every test clip (downloaded to a local folder; competition data stays on this machine).

Per clip, from 1024-sample Hann frames (hop 512):
    flatness   median spectral flatness (geometric / arithmetic mean of the power spectrum); white-ish
               noise sits near 0.5, speech far lower
    rms_db     loudness in dBFS
    zcr        zero-crossing rate
    speech_band  share of power in 300-3400 Hz
The detector is a one-feature threshold on whichever feature separates sentinels from normal clips
best (midpoint of the gap), cross-checked with a nearest-centroid rule on all four z-scored features.
Test clips are also checked for empty transcripts (Whisper word count from cache/asr_c.jsonl).

    python src/sentinel_guard.py <audio_dir>      # audio_dir/{train,test}/*.wav + manifest.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
FRAME, HOP = 1024, 512
FEATS = ["flatness", "rms_db", "zcr", "speech_band"]


def signal_features(path: Path) -> dict:
    sr, raw = wavfile.read(path)
    x = raw.astype(np.float64)
    if np.issubdtype(raw.dtype, np.integer):
        x /= float(np.iinfo(raw.dtype).max) + 1.0
    if x.ndim > 1:
        x = x.mean(1)
    n = 1 + max(0, (len(x) - FRAME) // HOP)
    fr = np.stack([x[i * HOP:i * HOP + FRAME] for i in range(n)]) * np.hanning(FRAME)
    P = np.abs(np.fft.rfft(fr, axis=1)) ** 2 + 1e-12
    flat = np.exp(np.log(P).mean(1)) / P.mean(1)
    f = np.fft.rfftfreq(FRAME, 1 / sr)
    band = (f >= 300) & (f <= 3400)
    return {"sr": sr, "seconds": len(x) / sr, "flatness": float(np.median(flat)),
            "rms_db": float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)),
            "zcr": float(np.mean(np.abs(np.diff(np.sign(x))) > 0)),
            "speech_band": float(P[:, band].sum() / P.sum())}


def main(audio_dir: Path):
    man = pd.read_csv(audio_dir / "manifest.csv")
    rows = []
    for r in man.itertuples():
        p = audio_dir / r.split / r.filename
        rows.append({"split": r.split, "filename": r.filename, "kind": r.kind,
                     **(signal_features(p) if p.exists() else {k: np.nan for k in FEATS})})
    D = pd.DataFrame(rows)
    W = {}
    for line in (ROOT / "cache" / "asr_c.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        W[(rec["split"], rec["filename"])] = len((rec.get("text") or "").split())
    D["whisper_words"] = [W.get((s, f), np.nan) for s, f in zip(D.split, D.filename)]

    sen, nor, tst = D[D.kind == "sentinel"], D[D.kind == "normal"], D[D.kind == "test"]
    print(f"clips read: sentinel {sen.flatness.notna().sum()}/{len(sen)}  normal {nor.flatness.notna().sum()}/{len(nor)}  "
          f"test {tst.flatness.notna().sum()}/{len(tst)}")
    print(D.groupby("kind")[FEATS + ["seconds"]].describe().T.loc[(slice(None), ["min", "50%", "max"]), :].round(4))

    best, best_gap = None, -np.inf
    for c in FEATS:
        lo_s, hi_s, lo_n, hi_n = sen[c].min(), sen[c].max(), nor[c].min(), nor[c].max()
        gap, direction = (lo_s - hi_n, ">") if sen[c].median() > nor[c].median() else (lo_n - hi_s, "<")
        scale = np.nanstd(np.r_[sen[c], nor[c]])
        if gap / scale > best_gap:
            best, best_gap, thr = (c, direction), gap / scale, (lo_s + hi_n) / 2 if direction == ">" else (hi_s + lo_n) / 2
            best_thr = thr
    c, direction = best
    print(f"\nbest single feature: {c} {direction} {best_thr:.4f}   (gap between classes = {best_gap:.2f} pooled std)")
    flag = tst[c] > best_thr if direction == ">" else tst[c] < best_thr

    mu, sd = D[FEATS].mean(), D[FEATS].std()
    z = (D[FEATS] - mu) / sd
    cs, cn = z[D.kind == "sentinel"].mean(), z[D.kind == "normal"].mean()
    near_sent = ((z - cs) ** 2).sum(axis=1) < ((z - cn) ** 2).sum(axis=1)
    D["flag_threshold"] = False
    D.loc[tst.index, "flag_threshold"] = flag.values
    D["flag_centroid"] = near_sent
    D["dist_to_threshold_std"] = (D[c] - best_thr) / D[c].std() * (1 if direction == ">" else -1)

    print(f"normal train clips misflagged: threshold {int((nor[c] > best_thr).sum() if direction == '>' else (nor[c] < best_thr).sum())}"
          f"  centroid {int(near_sent[nor.index].sum())}   sentinels missed by centroid {int((~near_sent[sen.index]).sum())}")
    print(f"test clips flagged: threshold {int(flag.sum())}   centroid {int(near_sent[tst.index].sum())}")
    print(f"test transcripts: min Whisper words {int(tst.whisper_words.min())}, clips under 10 words "
          f"{int((tst.whisper_words < 10).sum())}")
    print("closest test clips to the sentinel side:")
    print(D.loc[tst.index].sort_values("dist_to_threshold_std", ascending=False)
          [["filename", c, "rms_db", "seconds", "whisper_words", "dist_to_threshold_std"]].head(5).to_string(index=False))
    out = ROOT / "reports" / "phaseA"
    out.mkdir(parents=True, exist_ok=True)
    D.to_csv(out / "sentinel_guard.csv", index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
