"""Builds denoise_asr.py (the Kaggle notebook source) from worker.py and clips.json, and kernel-metadata.json.
usage: python kaggle/denoise_asr/build.py   (then jupytext -> .ipynb, kernelspec, kaggle kernels push)"""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
clips = (HERE / "clips.json").read_text()
worker = (HERE / "worker.py").read_text(encoding="utf-8")
assert "'''" not in worker and '"""' not in clips
HEAD = '''# %% [markdown]
# # Does recording noise change the transcripts? Denoise-then-transcribe diagnostic (2026-10-20)
# Test short clips are noisier recordings than the training short clips (SNR 37 vs 42 dB). If noise makes the speech recogniser
# produce errors, the transcript features read them as weak grammar on exactly the clips we score worst. Each selected clip is
# transcribed twice by the same Whisper (openai/whisper-large-v3-turbo, MIT; greedy, English): from the original audio and from
# the audio denoised by SpeechBrain's SepFormer DNS-4 enhancement model (speechbrain/sepformer-dns4-16k-enhancement,
# Apache-2.0; 8 s windows, 1 s cross-fade). The difference between the two transcripts is caused by the noise alone.
# Clips: every short clip (train 174, test 147) and, as a control, the 69 long test clips plus 69 random long training clips.
# No labels are used; nothing leaves Kaggle. Writes /kaggle/working/denoise_asr.csv (transcripts, SNR before / after).

# %%
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
print("torch", torch.__version__, "| python", sys.version.split()[0], "| GPUs", torch.cuda.device_count())
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hyperpyyaml"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "speechbrain==1.1.2"], check=True)
try:
    import speechbrain.inference.separation  # noqa: F401
except Exception as e:
    print("speechbrain 1.1.2 import failed:", repr(e), "-> trying 1.0.3")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "speechbrain==1.0.3"], check=True)
    import speechbrain.inference.separation  # noqa: F401
import speechbrain
print("speechbrain", speechbrain.__version__)
OUT = Path("/kaggle/working")
'''
LAUNCH = '''
# %%
from huggingface_hub import snapshot_download
snapshot_download("openai/whisper-large-v3-turbo")
snapshot_download("speechbrain/sepformer-dns4-16k-enhancement")
n_gpu = torch.cuda.device_count()
assert n_gpu >= 1, "no CUDA device: the accelerator must be a GPU"
procs = []
for g in range(n_gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    log = open(OUT / f"denoise_gpu{g}.log", "w")
    procs.append((subprocess.Popen([sys.executable, str(WORKER), str(g), str(n_gpu)], env=env, stdout=log, stderr=subprocess.STDOUT), log, g))
t0 = time.time()
while any(p.poll() is None for p, _, _ in procs):
    time.sleep(60)
    print(f"[{(time.time() - t0) / 60:.0f} min] " + " | ".join(
        ((OUT / f"denoise_gpu{g}.log").read_text().strip().splitlines() or ["-"])[-1][:110] for _, _, g in procs), flush=True)
for p, log, g in procs:
    log.close()
    print(f"--- GPU {g} exit {p.returncode}")
    print((OUT / f"denoise_gpu{g}.log").read_text()[-3000:])
assert all(p.returncode == 0 for p, _, _ in procs), "a worker failed"

# %%
import pandas as pd
D = pd.concat([pd.read_csv(OUT / f"denoise_part{g}.csv") for g in range(n_gpu)], ignore_index=True)
D.to_csv(OUT / "denoise_asr.csv", index=False)
print(len(D), "clips")
print(D.groupby(["split", "b3"])[["snr_orig", "snr_enh", "noise_orig", "noise_enh"]].mean().round(1).to_string())
print("done")
'''
nb = (HEAD + "(OUT / \"clips.json\").write_text(\"\"\"" + clips + "\"\"\")\n" + "WORKER = OUT / \"worker.py\"\n"
      + "WORKER.write_text(r'''" + worker + "''')\n" + LAUNCH)
ast.parse(nb)
(HERE / "denoise_asr.py").write_text(nb, encoding="utf-8")
meta = {"id": "amritarajput54/shl-denoise-asr", "title": "shl-denoise-asr", "code_file": "denoise_asr.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "keywords": [], "dataset_sources": [], "kernel_sources": [], "competition_sources": ["shl-hiring-assessment-2026"],
        "model_sources": []}
(HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built", len(nb), "chars")
