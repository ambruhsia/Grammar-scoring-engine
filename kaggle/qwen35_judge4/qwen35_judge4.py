# %% [markdown]
# # Qwen3.5-9B examiner, round 4: self-consistency + few-shot anchors (Phase I, 2026-10-17)
# The examiner's rubric grade is the only component proven by a matched public A/B (0.3280 with it vs 0.3335 without). This job
# makes that grade less noisy instead of adding new signals:
# - self-consistency: 5 paraphrases of the rubric question (band wording, rubric in reverse order, two-rater framing, "grammar
#   only", condensed scale) on the Whisper and verbatim transcripts; with round 1's original wording that is 6 grades per clip
#   and transcript, averaged locally (their spread is an uncertainty feature);
# - few-shot: 6 graded training answers (grades 2.0-5.0, all from the training topic "playground") shown before the Whisper
#   transcript. A training clip in seed-42 fold k only sees anchors from the other folds of the speaker-grouped split (never its
#   own fold, its own speaker or itself); test clips see one fixed set. The anchor table is embedded below.
# Model: Qwen/Qwen3.5-9B (Apache-2.0), NF4 on each T4, thinking off; one forward pass per (clip, prompt); the next-token
# distribution over the digits 1-5. No generation. Nothing leaves Kaggle.
# Writes /kaggle/working/qwen35_judge4.csv: split, filename, {view}_{prompt}_p1..p5, {view}_{prompt}_top.

# %%
import json
import os
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes>=0.45"], check=True)
OUT = Path("/kaggle/working")
ANCHORS = json.loads('{}')  # anchor clips and their grades (training labels) removed for the public copy: chosen per fold from the training set
(OUT / "anchors.json").write_text(json.dumps(ANCHORS))
SCRIPT = OUT / "judge.py"
SCRIPT.write_text(r"""
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

PART, NPART = int(sys.argv[1]), int(sys.argv[2])
OUT = Path("/kaggle/working")
MODEL = "Qwen/Qwen3.5-9B"
ANCHORS = json.loads((OUT / "anchors.json").read_text())
SYSTEM = "You are an experienced examiner of spoken English in a speaking test."
VIEWS = {"verbatim": "an automatic transcript that keeps the speaker's own words, fillers and repetitions",
         "whisper": "an automatic transcript that may have smoothed over some of the speaker's errors"}
R = {1: "struggles with sentence structure; limited control of simple grammatical structures.",
     2: "uses simple structures, with consistent basic errors; sentences are often incomplete.",
     3: "a decent grasp of either grammar or sentence structure, with errors in the other.",
     4: "strong control of grammar and structure; occasional minor errors that do not impede understanding.",
     5: "high grammatical accuracy; complex structures handled well; self-corrections when needed."}
RUBRIC = "\n".join(f"{k}: {v}" for k, v in R.items())
PROMPTS = {
    "para_bands": ("Which band best describes the candidate's GRAMMAR?\n"
                   "Band 1: very limited control; even simple sentences break down.\n"
                   "Band 2: relies on simple sentences; frequent basic errors; many sentences left unfinished.\n"
                   "Band 3: reasonable control of either grammatical accuracy or sentence structure, but noticeable errors in the other.\n"
                   "Band 4: good control of grammar and sentence structure; only occasional small slips that do not affect meaning.\n"
                   "Band 5: consistently accurate; handles complex sentences with ease; corrects own slips.\n"
                   "Answer with the band number, one digit from 1 to 5."),
    "para_reversed": ("Grade the speaker's GRAMMAR on this rubric (highest band first):\n"
                      + "\n".join(f"{k}: {R[k]}" for k in (5, 4, 3, 2, 1)) + "\nAnswer with one digit from 1 to 5."),
    "para_raters": ("Two trained raters will grade this candidate's grammar on a 1-5 scale (1 = struggles with basic sentence "
                    "structure, 3 = a decent grasp with clear errors, 5 = accurate and complex). Which grade would they most likely "
                    "agree on? Answer with one digit from 1 to 5."),
    "para_only": ("Judge only GRAMMAR (accuracy and range of structures), not pronunciation, fluency, vocabulary or the content of "
                  "the answer. Fillers and repetitions are normal in speech and should not lower the grade by themselves. On a scale "
                  "from 1 (very weak grammar) to 5 (excellent grammar), what grade does this speaker deserve? Answer with one digit "
                  "from 1 to 5."),
    "para_short": ("Rate the speaker's grammatical control: 1 = limited, 2 = basic with many errors, 3 = adequate with some errors, "
                   "4 = strong with minor slips, 5 = near-native accuracy and complexity. Answer with one digit from 1 to 5."),
}
PLAN = [(v, p) for v in ("whisper", "verbatim") for p in PROMPTS] + [("whisper", "fewshot")]

def find(name):
    hits = [Path(d) / name for d, _, files in os.walk("/kaggle/input", followlinks=True) if name in files]
    assert hits, name
    return min(hits, key=lambda p: len(p.parts))

def jl(p, field):
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line); out[(r["split"], r["filename"])] = r.get(field) or ""
    return out

TX = {"verbatim": jl(find("asr_v.jsonl"), "text"), "whisper": jl(find("asr_c.jsonl"), "text")}
train, test = pd.read_csv(find("train.csv")), pd.read_csv(find("test.csv"))
FG = pd.read_csv(find("folds_grouped.csv")); FOLD = FG[FG.seed == 42].set_index("filename").fold.to_dict()
KEYS = [("train", f) for f in train.filename[train.label > 0]] + [("test", f) for f in test.filename]
KEYS = KEYS[PART::NPART]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, device_map={"": 0}, dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16))
model.eval()
digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 6)]

def shorten(t, n=110):
    w = t.split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")

def fewshot_block(k):
    rows = ANCHORS[k]
    ex = "\n\n".join(f"Example {i + 1} (grade {g:g}):\n{shorten(TX['whisper'][('train', f)])}" for i, (f, g) in enumerate(rows))
    return ("Here are graded answers from the same test, to show how the scale is used:\n\n" + ex + "\n\n"
            "Now grade the candidate's GRAMMAR on this rubric:\n" + RUBRIC + "\nAnswer with one digit from 1 to 5.")

def build(view, prompt, key):
    text = TX[view][key][:4000]
    if prompt == "fewshot":
        k = "test" if key[0] == "test" else str(FOLD[key[1]])
        assert key[0] == "test" or all(FOLD[f] != FOLD[key[1]] for f, _ in ANCHORS[k]), key
        body = f"{fewshot_block(k)}\n\nThe candidate's answer ({VIEWS[view]}):\n{text}\n\nGrade:"
    else:
        body = f"Below is {VIEWS[view]} of a candidate's 45-60 second spoken answer.\n\nTRANSCRIPT:\n{text}\n\n{PROMPTS[prompt]}"
    msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]
    try:
        return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

rows, t0 = [], time.time()
with torch.inference_mode():
    for i, k in enumerate(KEYS):
        row = {"split": k[0], "filename": k[1]}
        for view, prompt in PLAN:
            enc = tok(build(view, prompt, k), return_tensors="pt").to(0)
            logits = model(**enc).logits[0, -1].float()
            p = torch.softmax(logits[digit_ids], 0).cpu().numpy()
            for d in range(5):
                row[f"{view}_{prompt}_p{d + 1}"] = float(p[d])
            row[f"{view}_{prompt}_top"] = int(int(torch.argmax(logits)) in digit_ids)
        rows.append(row)
        if i == 0 or (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  part {PART}: {i + 1}/{len(KEYS)} {el / (i + 1):.2f} s/clip ETA {el / (i + 1) * (len(KEYS) - i - 1) / 60:.1f} min", flush=True)
pd.DataFrame(rows).to_csv(OUT / f"judge_part{PART}.csv", index=False)
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
    log = open(OUT / f"judge_gpu{g}.log", "w")
    procs.append((subprocess.Popen([sys.executable, str(SCRIPT), str(g), str(n_gpu)], env=env, stdout=log, stderr=subprocess.STDOUT), log, g))
t0 = time.time()
while any(p.poll() is None for p, _, _ in procs):
    time.sleep(60)
    print(f"[{(time.time() - t0) / 60:.0f} min] " + " | ".join(
        ((OUT / f"judge_gpu{g}.log").read_text().strip().splitlines() or ["-"])[-1][:100] for _, _, g in procs), flush=True)
for p, log, g in procs:
    log.close()
    print(f"--- GPU {g} exit {p.returncode}\n" + (OUT / f"judge_gpu{g}.log").read_text()[-2500:])
assert all(p.returncode == 0 for p, _, _ in procs), "a judge process failed"

# %%
import pandas as pd
J = pd.concat([pd.read_csv(OUT / f"judge_part{g}.csv") for g in range(n_gpu)], ignore_index=True)
J.to_csv(OUT / "qwen35_judge4.csv", index=False)
tops = [c for c in J.columns if c.endswith("_top")]
print(f"{len(J)} clips; share where the most likely next token is a digit: " + ", ".join(f"{c[:-4]} {J[c].mean():.1%}" for c in tops))
print("done")
