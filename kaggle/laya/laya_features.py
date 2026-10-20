# %% [markdown]
# # Laya rubric questions for SHL grammar scoring (a new member, approved 2026-10-14)
# Laya (convaiinnovations/laya, Apache-2.0; ModernBERT-large encoder, 421M) answers typed questions about a text with calibrated
# probabilities. Each transcript (CTC raw, Whisper clean) is asked rubric-aligned questions; no labels are used here.
# Reads only the cached transcripts from the attached caches dataset; Laya runs inside this session, so the text stays on Kaggle.
# Writes to /kaggle/working:
# - `laya_judge.csv`: per clip and text, each question's level probabilities and expected level.
# - `laya_hidden.npz`: the encoder's mean-pooled last hidden state while answering the rubric question (1024 per text), for
#   ridge heads fitted locally on speaker-grouped folds.

# %%
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("laya") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "laya==0.4.7"], check=True)

# %%
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import laya

OUT = Path("/kaggle/working")
MODEL, REVISION = "convaiinnovations/laya", "7b928d828b7b0e022f929d9bd2e44165aa270148"
MAX_LEN, BATCH = 512, 16
assert torch.cuda.is_available(), "no CUDA device: the accelerator must be a GPU"
torch.manual_seed(42)


def find(name):
    for d, _, files in os.walk("/kaggle/input", followlinks=True):
        if name in files:
            return Path(d) / name
    raise FileNotFoundError(f"{name} not found under /kaggle/input: attach the caches dataset")


def load_jsonl(p):
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            out[(r["split"], r["filename"])] = r
        except json.JSONDecodeError:
            pass
    return out


W, C = load_jsonl(find("asr_c.jsonl")), load_jsonl(find("ctc.jsonl"))
KEYS = sorted(set(W) & set(C))
STATES = {"ctc": ["Automatic transcript of a spoken English answer (no punctuation; some words may be misrecognised): "
                  + (C[k]["raw_text"] or "") for k in KEYS],
          "whisper": ["Automatic transcript of a spoken English answer: " + (W[k]["text"] or "") for k in KEYS]}
print(f"{len(KEYS)} clips, GPU {torch.cuda.get_device_name(0)}, laya {laya.__version__}")

# %%
# Rubric-aligned questions. "score" levels run from worst to best; fillers, pauses and recognition errors are not grammar.
IGNORE = " Ignore fillers, pauses, repetitions and transcription errors."
QUESTIONS = {
    "grammar": {"type": "score", "instructions": "Rate the speaker's grammar on this rubric." + IGNORE, "criteria": [
        "struggles with sentence structure; only memorised or very simple patterns",
        "limited control; simple structures with consistent basic mistakes; may leave sentences incomplete",
        "a decent grasp of sentence structure but errors in grammar or syntax",
        "strong control of grammar; occasional minor errors that do not cause misunderstanding",
        "high grammatical accuracy and control of complex structures; seldom makes mistakes"]},
    "accuracy": {"type": "score", "instructions": "How accurate is the speaker's grammar?" + IGNORE, "criteria": [
        "frequent basic grammatical mistakes", "noticeable grammatical errors", "occasional minor errors",
        "almost no grammatical errors"]},
    "complexity": {"type": "score", "instructions": "How complex are the speaker's sentence structures?", "criteria": [
        "memorised phrases and very simple sentences", "simple sentences", "some complex sentences",
        "varied and complex sentence structures"]},
    "incomplete": {"type": "noul", "instructions": "Does the speaker leave sentences unfinished or incomplete?" + IGNORE},
    "selfcorrect": {"type": "noul", "instructions": "Does the speaker notice and correct their own grammatical mistakes?"},
}

agent = laya.load(MODEL, device="cuda", revision=REVISION)
agent.model.eval()

# mean-pooled encoder output while answering the rubric question, captured by a forward hook (one row per state)
captured = []


def hook(module, args, kwargs, output):
    ids = kwargs.get("input_ids", args[0] if args else None)
    h = output.last_hidden_state.float()
    m = (ids != agent.tok.pad_token_id).float()[:, :, None]
    captured.append(((h * m).sum(1) / m.sum(1).clamp(min=1.0)).cpu().numpy().astype(np.float16))


judge = {"split": [k[0] for k in KEYS], "filename": [k[1] for k in KEYS]}
hidden = {"split": np.array(judge["split"]), "filename": np.array(judge["filename"])}
t0 = time.time()
for text, states in STATES.items():
    for qid, q in QUESTIONS.items():
        handle = agent.model.encoder.register_forward_hook(hook, with_kwargs=True) if qid == "grammar" else None
        captured.clear()
        res = agent.predict_batch(states, {qid: q}, batch_size=BATCH, max_len=MAX_LEN)
        if handle is not None:
            handle.remove()
            H = np.concatenate(captured)
            assert len(H) == len(states), f"captured {len(H)} rows for {len(states)} states"
            hidden[f"{text}_mean"] = H
        assert len(res) == len(states)
        a = [r["answers"][qid] for r in res]
        if q["type"] == "score":
            k = len(q["criteria"])
            for i in range(k):
                judge[f"{text}_{qid}_p{i}"] = [x["probabilities"][str(i)] for x in a]
            judge[f"{text}_{qid}_score"] = [x["score"] for x in a]
        else:
            judge[f"{text}_{qid}_p"] = [x["noul"] for x in a]
        print(f"  {text} {qid}: done  {(time.time() - t0) / 60:.1f} min", flush=True)

pd.DataFrame(judge).to_csv(OUT / "laya_judge.csv", index=False)
np.savez_compressed(OUT / "laya_hidden.npz", **hidden)
print(f"done {(time.time() - t0) / 60:.1f} min; columns {len(judge) - 2}")
print(pd.DataFrame(judge).filter(like="score").describe().round(3).to_string())
