# %% [markdown]
# # Qwen3.5-9B pairwise examiner, anchored by batch (2026-10-17)
# The absolute rubric grade from an LLM is noisy and compressed. Comparative judgements are easier for a model and for raters
# alike, so here the examiner compares each answer's GRAMMAR with 8 graded training answers ("anchors") from the same recording
# batch (exact 45.06 s / other short / long; grades 2.0-5.0, 5.0 twice), asking "which shows better grammar, A or B?". Every pair
# is asked in both orders (the clip as A, then as B), so position bias cancels in the average.
# Anchors (kaggle/qwen35_pairwise/make_anchors.py): training clips of single-clip speakers (speaker groups never cross train and
# test, so no clip is ever compared with its own speaker), >= 50 verbatim words, the one per grade that the out-of-fold model
# agrees with most. An anchor is not compared with itself. The anchors' labels are used only locally, after this job.
# Transcript: the verbatim Whisper transcript (keeps the speaker's own words, fillers and repetitions).
# Model: Qwen/Qwen3.5-9B (Apache-2.0), NF4 on each T4, thinking off; one forward pass per (clip, anchor, order); the next-token
# probabilities of "A" and "B". No generation. Nothing leaves Kaggle.
# Writes /kaggle/working/qwen35_pairwise.csv: split, filename, batch, anchor, anchor_grade, order (0 = clip is A), p_a, top.

# %%
import json
import os
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes>=0.45"], check=True)
OUT = Path("/kaggle/working")
ANCHORS = json.loads('__ANCHORS__')
BATCH = json.loads('__BATCH__')
(OUT / "anchors.json").write_text(json.dumps(ANCHORS))
(OUT / "batch_map.json").write_text(json.dumps(BATCH))
SCRIPT = OUT / "pairwise.py"
SCRIPT.write_text(r"""
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

PART, NPART = int(sys.argv[1]), int(sys.argv[2])
OUT = Path("/kaggle/working")
MODEL = "Qwen/Qwen3.5-9B"
ANCHORS = json.loads((OUT / "anchors.json").read_text())
BATCH = json.loads((OUT / "batch_map.json").read_text())
SYSTEM = "You are an experienced examiner of spoken English in a speaking test."
QUESTION = ("Below are transcripts of two different candidates' 45-60 second spoken answers in the same English speaking test. "
            "The transcripts were made automatically and keep the speakers' own words, fillers and repetitions.\n\n"
            "Compare ONLY their GRAMMAR: accuracy (verb forms, tense, agreement, articles, prepositions, word order) and control of "
            "sentence structure (complete sentences, range and complexity of structures). Ignore the topic, the content, vocabulary, "
            "pronunciation and the length of the answers. Fillers, hesitations and self-corrections are normal in speech and are "
            "not grammar errors by themselves.")

def find(name):
    hits = [Path(d) / name for d, _, files in os.walk("/kaggle/input", followlinks=True) if name in files]
    assert hits, name
    return min(hits, key=lambda p: len(p.parts))

TX = {}
for line in find("asr_v.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        r = json.loads(line); TX[(r["split"], r["filename"])] = r.get("text") or ""
train, test = pd.read_csv(find("train.csv")), pd.read_csv(find("test.csv"))
KEYS = [("train", f) for f in train.filename[train.label > 0]] + [("test", f) for f in test.filename]
assert len(KEYS) == 948 and all(f"{s}/{f}" in BATCH for s, f in KEYS) and all(k in TX for k in KEYS)
KEYS = KEYS[PART::NPART]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, device_map={"": 0}, dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16))
model.eval()
ids = [tok.encode(x, add_special_tokens=False)[0] for x in ("A", "B")]

def shorten(t, n=220):
    w = t.split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")

def build(a, b):
    body = (f"{QUESTION}\n\nANSWER A:\n{shorten(a)}\n\nANSWER B:\n{shorten(b)}\n\n"
            "Which candidate shows better grammar? Answer with one letter: A or B.")
    msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]
    try:
        return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

rows, t0 = [], time.time()
with torch.inference_mode():
    for i, k in enumerate(KEYS):
        bt = BATCH[f"{k[0]}/{k[1]}"]
        for af, ag in ANCHORS[bt]:
            if k == ("train", af):
                continue                                   # an anchor is not compared with itself
            for order in (0, 1):
                a, b = (TX[k], TX[("train", af)]) if order == 0 else (TX[("train", af)], TX[k])
                enc = tok(build(a, b), return_tensors="pt").to(0)
                logits = model(**enc).logits[0, -1].float()
                p = torch.softmax(logits[ids], 0).cpu().numpy()
                top = int(torch.argmax(logits))
                if i == 0 and order == 0 and af == ANCHORS[bt][0][0]:
                    print("  prompt tokens", enc["input_ids"].shape[1], "| top-5 next tokens:",
                          [tok.decode([t]) for t in torch.topk(logits, 5).indices.tolist()], flush=True)
                rows.append({"split": k[0], "filename": k[1], "batch": bt, "anchor": af, "anchor_grade": ag, "order": order,
                             "p_a": float(p[0]), "top": int(top in ids)})
        if i == 0 or (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  part {PART}: {i + 1}/{len(KEYS)} {el / (i + 1):.2f} s/clip ETA {el / (i + 1) * (len(KEYS) - i - 1) / 60:.1f} min", flush=True)
        if (i + 1) % 100 == 0:
            pd.DataFrame(rows).to_csv(OUT / f"pairwise_part{PART}.csv", index=False)      # checkpoint
pd.DataFrame(rows).to_csv(OUT / f"pairwise_part{PART}.csv", index=False)
print(f"part {PART} done {(time.time() - t0) / 60:.1f} min", flush=True)
""")

# %%
import torch
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.5-9B")
n_gpu = torch.cuda.device_count()
assert n_gpu >= 1, "no CUDA device: the accelerator must be a GPU"
procs = []
for g in range(n_gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    log = open(OUT / f"pairwise_gpu{g}.log", "w")
    procs.append((subprocess.Popen([sys.executable, str(SCRIPT), str(g), str(n_gpu)], env=env, stdout=log, stderr=subprocess.STDOUT), log, g))
t0 = time.time()
while any(p.poll() is None for p, _, _ in procs):
    time.sleep(60)
    print(f"[{(time.time() - t0) / 60:.0f} min] " + " | ".join(
        ((OUT / f"pairwise_gpu{g}.log").read_text().strip().splitlines() or ["-"])[-1][:100] for _, _, g in procs), flush=True)
for p, log, g in procs:
    log.close()
    print(f"--- GPU {g} exit {p.returncode}\n" + (OUT / f"pairwise_gpu{g}.log").read_text()[-2500:])
assert all(p.returncode == 0 for p, _, _ in procs), "a pairwise process failed"

# %%
import pandas as pd
P = pd.concat([pd.read_csv(OUT / f"pairwise_part{g}.csv") for g in range(n_gpu)], ignore_index=True)
P.to_csv(OUT / "qwen35_pairwise.csv", index=False)
print(f"{len(P)} comparisons, {P[['split', 'filename']].drop_duplicates().shape[0]} clips; top token is A/B: {P.top.mean():.1%}")
pa0, pa1 = P[P.order == 0].p_a.mean(), P[P.order == 1].p_a.mean()
print(f"mean P(A): clip first {pa0:.3f}, anchor first {pa1:.3f} (position bias = their mean - 0.5: {(pa0 + pa1) / 2 - 0.5:+.3f})")
print("done")
