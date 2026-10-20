# %% [markdown]
# # Qwen3.5-9B zero-shot grammar judge, round 3: an error profile (Phase I, 2026-10-16)
# The judge's numbers carry information the stack misses almost only on SHORT clips (partial r with the grade given the stack's
# prediction: short +0.08..+0.17, 45.06 s batch +0.11..+0.31, long ~0). Round 3 asks for counts of each error type (agreement,
# verb form, articles, prepositions, word order, fragments), of self-corrections and of the range of complex structures.
# Round 1 (shl-qwen35-judge: rubric + accuracy) correlates +0.73 with the grade and +0.14 with the honest stack's residuals.
# Round 2 asks about complexity, completeness and the CEFR level, and asks the rubric question with two transcripts side by
# side (verbatim + CTC raw), so the model can tell recognition noise from the speaker's own errors.
# The #1 public write-up (AMITABHSHUKLA) scores each transcript with Qwen3.5-9B and the official rubric, taking the
# probability-weighted grade; our Qwen2.5-7B judge correlates +0.11..+0.13 with the honest stack's residuals, so a stronger judge
# may carry more. Code written from scratch.
# Model: Qwen/Qwen3.5-9B (Apache-2.0), 4-bit NF4 (bitsandbytes) on each T4, thinking mode off. No generation: one forward pass per
# (clip, transcript, prompt); the next-token distribution over the digits 1-5 is the judge's grade distribution. The state at the
# answer position (LM layers at 50% / 65% / 80% depth) is kept for ridge heads. No labels are used; nothing leaves Kaggle.
# Transcripts: Whisper clean (asr_c.jsonl) and CTC raw (ctc.jsonl) from the caches dataset, verbatim Whisper-large-v3-turbo
# (asr_v.jsonl) from the shl-verbatim-text job's output.
# Writes /kaggle/working: qwen35_judge3.csv (split, filename, {view}_{prompt}_q0..q5 = P(count = k), {view}_{prompt}_top),
# qwen35_hidden3.npz.

# %%
import os
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes>=0.45"], check=True)
OUT = Path("/kaggle/working")
SCRIPT = OUT / "judge.py"
SCRIPT.write_text(r'''
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

PART, NPART = int(sys.argv[1]), int(sys.argv[2])
OUT = Path("/kaggle/working")
MODEL = "Qwen/Qwen3.5-9B"
RUBRIC = ("1: struggles with sentence structure; limited control of simple grammatical structures.\n"
          "2: uses simple structures, with consistent basic errors; sentences are often incomplete.\n"
          "3: a decent grasp of either grammar or sentence structure, with errors in the other.\n"
          "4: strong control of grammar and structure; occasional minor errors that do not impede understanding.\n"
          "5: high grammatical accuracy; complex structures handled well; self-corrections when needed.")
VIEWS = {"verbatim": "an automatic transcript that keeps the speaker's own words, fillers and repetitions",
         "ctc": "a raw letter-by-letter automatic transcript: no punctuation, many misspelled words, but the speaker's grammar is kept",
         "whisper": "an automatic transcript that may have smoothed over some of the speaker's errors",
         "both": "two automatic transcripts of the same answer (A keeps the speaker's words and fillers; B is a raw letter-by-letter "
                 "transcript with misspellings; where they disagree, trust the grammar both share)"}
PROMPTS = {
    "agreement": ("How many subject-verb AGREEMENT errors does the speaker make (for example 'he go', 'they was', 'the people is')? "
                  "Answer with one digit from 0 to 5, where 5 means five or more."),
    "verbform": ("How many errors in VERB TENSE or VERB FORM does the speaker make (for example 'yesterday I go', 'I have went', "
                 "'he can goes', 'I am agree')? Answer with one digit from 0 to 5, where 5 means five or more."),
    "articles": ("How many missing or wrong ARTICLES or determiners does the speaker produce (a, an, the, some, much, many)? "
                 "Answer with one digit from 0 to 5, where 5 means five or more."),
    "prepositions": ("How many wrong or missing PREPOSITIONS does the speaker produce? Answer with one digit from 0 to 5, where 5 "
                     "means five or more."),
    "wordorder": ("How many sentences have wrong WORD ORDER or a broken structure? Answer with one digit from 0 to 5, where 5 means "
                  "five or more."),
    "fragments": ("How many sentences are left INCOMPLETE or abandoned (not counting deliberate self-corrections)? Answer with one "
                  "digit from 0 to 5, where 5 means five or more."),
    "selfcorr": ("How many times does the speaker notice and CORRECT their own grammatical mistake (for example 'he go... he goes')? "
                 "Answer with one digit from 0 to 5, where 5 means five or more."),
    "range": ("How many DIFFERENT kinds of complex structures does the speaker use correctly (relative clauses, conditionals, passive "
              "voice, perfect tenses, subordinate clauses, comparatives)? Answer with one digit from 0 to 5, where 5 means five or more."),
}
PLAN = [("verbatim", p) for p in ("agreement", "verbform", "articles", "prepositions", "wordorder", "fragments", "selfcorr", "range")]
PLAN += [("both", "agreement"), ("both", "verbform"), ("both", "articles")]
N_DIGITS = {}

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

TX = {"verbatim": jl(find("asr_v.jsonl"), "text"), "ctc": jl(find("ctc.jsonl"), "raw_text"), "whisper": jl(find("asr_c.jsonl"), "text")}
TX["both"] = {k: "A: " + TX["verbatim"][k][:2500] + "\n\nB: " + TX["ctc"].get(k, "")[:2500] for k in TX["verbatim"]}
train, test = pd.read_csv(find("train.csv")), pd.read_csv(find("test.csv"))
KEYS = [("train", f) for f in train.filename[train.label > 0]] + [("test", f) for f in test.filename]
KEYS = KEYS[PART::NPART]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, device_map={"": 0}, dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16))
model.eval()
digits = [tok.encode(str(d), add_special_tokens=False) for d in range(0, 6)]
assert all(len(d) == 1 for d in digits), digits
digit_ids = [d[0] for d in digits]
n_layers = model.config.text_config.num_hidden_layers if hasattr(model.config, "text_config") else model.config.num_hidden_layers
LAYERS = sorted({int(round(f * n_layers)) for f in (0.5, 0.65, 0.8)})
print(f"part {PART}/{NPART}: {len(KEYS)} clips; {n_layers} layers, keeping {LAYERS}; digit ids {digit_ids}", flush=True)

def build(view, prompt, text):
    msg = [{"role": "system", "content": "You are an experienced examiner of spoken English in a speaking test."},
           {"role": "user", "content": f"Below is {VIEWS[view]} of a candidate's 45-60 second spoken answer.\n\n"
                                       f"TRANSCRIPT:\n{text[:4000]}\n\n{PROMPTS[prompt]}"}]
    try:
        s = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        s = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return s

rows, H = [], np.zeros((len(KEYS), len(PLAN), len(LAYERS), model.config.text_config.hidden_size
                        if hasattr(model.config, "text_config") else model.config.hidden_size), np.float16)
t0 = time.time()
with torch.inference_mode():
    for i, k in enumerate(KEYS):
        row = {"split": k[0], "filename": k[1]}
        for ji, (view, prompt) in enumerate(PLAN):
            nd = 6
            enc = tok(build(view, prompt, TX[view][k]), return_tensors="pt").to(0)
            out = model(**enc, output_hidden_states=True)
            logits = out.logits[0, -1].float()
            p = torch.softmax(logits[digit_ids[:nd]], 0).cpu().numpy()
            for d in range(nd):
                row[f"{view}_{prompt}_q{d}"] = float(p[d])
            row[f"{view}_{prompt}_top"] = int(int(torch.argmax(logits)) in digit_ids[:nd])
            hs = out.hidden_states
            for li, l in enumerate(LAYERS):
                H[i, ji, li] = hs[l][0, -1].float().cpu().numpy().astype(np.float16)
        rows.append(row)
        if i == 0 or (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  part {PART}: {i + 1}/{len(KEYS)} {el / (i + 1):.2f} s/clip ETA {el / (i + 1) * (len(KEYS) - i - 1) / 60:.1f} min", flush=True)
pd.DataFrame(rows).to_csv(OUT / f"judge_part{PART}.csv", index=False)
np.savez_compressed(OUT / f"hidden_part{PART}.npz", H=H, split=np.array([k[0] for k in KEYS]), filename=np.array([k[1] for k in KEYS]),
                    layers=np.array(LAYERS), plan=np.array([f"{v}_{p}" for v, p in PLAN]))
print(f"part {PART} done {(time.time() - t0) / 60:.1f} min", flush=True)
''')

# %%
import torch
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.5-9B")            # once, before two processes would race on the cache
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
import numpy as np
import pandas as pd
J = pd.concat([pd.read_csv(OUT / f"judge_part{g}.csv") for g in range(n_gpu)], ignore_index=True)
J.to_csv(OUT / "qwen35_judge3.csv", index=False)
parts = [np.load(OUT / f"hidden_part{g}.npz") for g in range(n_gpu)]
np.savez_compressed(OUT / "qwen35_hidden3.npz", H=np.concatenate([z["H"] for z in parts]),
                    split=np.concatenate([z["split"] for z in parts]), filename=np.concatenate([z["filename"] for z in parts]),
                    layers=parts[0]["layers"], plan=parts[0]["plan"])
tops = [c for c in J.columns if c.endswith("_top")]
print(f"{len(J)} clips; share where the most likely next token is a digit: " + ", ".join(f"{c[:-4]} {J[c].mean():.1%}" for c in tops))
print("done")
