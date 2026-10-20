# %% [markdown]
# # Laya fine-tuned the way it is designed: through its decision head, on typed rubric questions (approved idea #5)
# Our first Laya fine-tune replaced the decision head with a regression head, so it became one more transcript regressor. Here
# Laya keeps its decision head and is trained with its own `laya.train.finetune` (soft cross-entropy, per-type temperature
# calibration) on typed questions about each clip, fixed before running:
# - seven yes/no threshold questions, "is this speaker's grammar at level t or above?" for t = 2.0, 2.5, ..., 5.0, its strongest
#   question type; and one score question over the five rubric levels;
# - soft targets: half the clip's own label, half its speaker group's labels (the denoising the other members use);
# - speaker-grouped seed-42 folds: 5 fine-tunes, each predicting its held-out fold and the test clips.
# Prediction: expected grade = 1.5 + 0.5 * sum_t P(grade >= t) (made monotone in t), plus the score question's expected level and
# entropy (a calibrated uncertainty for the confidence-dependent blend). A fold "collapses" when its predictions barely vary
# (std < 0.1) or its validation RMSE is >= 0.97 x predicting the training mean; it is then retried once with another seed.
# Backbone: convaiinnovations/laya (Apache-2.0), pinned revision; transcripts stay on Kaggle.

# %%
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("laya") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "laya==0.4.7"], check=True)

# %%
import gc
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download

import laya
from laya.train import TrainConfig, finetune

OUT = Path("/kaggle/working")
MODEL, REVISION = "convaiinnovations/laya", "7b928d828b7b0e022f929d9bd2e44165aa270148"
THRESHOLDS = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
LEVELS = ["struggles with sentence structure; only memorised or very simple patterns",
          "limited control; simple structures with consistent basic mistakes; may leave sentences incomplete",
          "a decent grasp of sentence structure but errors in grammar or syntax",
          "strong control of grammar; occasional minor errors that do not cause misunderstanding",
          "high grammatical accuracy and control of complex structures; seldom makes mistakes"]
IGNORE = " Ignore fillers, pauses, repeated words and misrecognised words."
QUESTIONS = {"level": {"type": "score", "instructions": "Rate the speaker's grammar on this rubric." + IGNORE, "criteria": LEVELS}}
for t in THRESHOLDS:
    QUESTIONS[f"ge_{t:g}"] = {"type": "noul", "instructions": f"On a 1-5 grammar rubric (1 = {LEVELS[0]}; 5 = {LEVELS[4]}), is this "
                                                              f"speaker's grammar at level {t:g} or above?" + IGNORE}
CFG = dict(epochs=4, micro_batch=8, grad_accum=2, encoder_lr=2.5e-5, head_lr=1e-4, loss="soft-ce", max_len=512)
CLIP, LAM = (1.5, 5.0), 0.5
assert torch.cuda.is_available(), "no CUDA device: the accelerator must be a GPU"
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def find(name):
    hits = [Path(d) / name for d, _, files in os.walk("/kaggle/input", followlinks=True) if name in files]
    assert hits, f"{name} not under /kaggle/input: attach the artefacts dataset"
    return min(hits, key=lambda p: len(p.parts))


def load_jsonl(p):
    out = {}
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            out[(r["split"], r["filename"])] = r
        except json.JSONDecodeError:
            pass
    return out


W, C = load_jsonl(find("asr_c.jsonl")), load_jsonl(find("ctc.jsonl"))
train, test = pd.read_csv(find("train.csv")), pd.read_csv(find("test.csv"))
FG = pd.read_csv(find("folds_grouped.csv"))
FG = FG[FG.seed == 42]
SG = pd.read_csv(find("speaker_groups.csv"))
T = (train[train.label > 0].merge(FG[["filename", "fold"]], on="filename")
     .merge(SG[SG.split == "train"][["filename", "group"]], on="filename").reset_index(drop=True))
assert len(T) == 732
TE = test[["filename"]].reset_index(drop=True)
y = T.label.values.astype(float)
state = lambda s, f: (f"Raw (the speaker's exact words; no punctuation, some words may be misrecognised): "
                      f"{C[(s, f)]['raw_text'] or '(not recognised)'}\nClean (the same words as a cleaned transcript): {W[(s, f)]['text']}")
tr_states = [state("train", f) for f in T.filename]
te_states = [state("test", f) for f in TE.filename]


def level_dist(v):
    """A grade on the 1-5 scale -> probabilities over the 5 levels, split between the two neighbours."""
    x = float(np.clip(v, 1.0, 5.0)) - 1.0
    lo = int(np.floor(x))
    p = np.zeros(5)
    if lo >= 4:
        p[4] = 1.0
    else:
        p[lo], p[lo + 1] = 1.0 - (x - lo), x - lo
    return p


def gold_rows(idx):
    """Soft targets for training clips idx: (1 - LAM) x the clip's own label + LAM x its speaker group's labels (within idx)."""
    rows, sub = [], T.iloc[idx]
    by_group = sub.groupby("group").label.apply(list).to_dict()
    for i in idx:
        g = by_group[T.group[i]]
        lev = (1 - LAM) * level_dist(y[i]) + LAM * np.mean([level_dist(v) for v in g], 0)
        gold = {"level": {"probabilities": {str(k): float(lev[k]) for k in range(5)}}}
        for t in THRESHOLDS:
            p = (1 - LAM) * float(y[i] >= t) + LAM * float(np.mean([v >= t for v in g]))
            gold[f"ge_{t:g}"] = {"probabilities": {"false": 1.0 - p, "true": p}}
        rows.append({"state": tr_states[i], "questions": QUESTIONS, "gold": gold})
    return rows


def predict(agent, states):
    res = agent.predict_batch(states, QUESTIONS, batch_size=16, max_len=CFG["max_len"])
    ge = np.array([[r["answers"][f"ge_{t:g}"]["noul"] for t in THRESHOLDS] for r in res])
    ge = np.minimum.accumulate(ge, axis=1)                      # P(grade >= t) can only fall as t rises
    lev = np.array([[r["answers"]["level"]["probabilities"][str(k)] for k in range(5)] for r in res])
    return (1.5 + 0.5 * ge.sum(1), lev @ np.arange(1, 6), -(lev * np.log(np.clip(lev, 1e-12, 1))).sum(1))


base = snapshot_download(MODEL, revision=REVISION, ignore_patterns=["multilingual/*", "typed-decisions/*", "assets/*", "eval/*"])
print(f"train {len(T)}, test {len(TE)}; laya {laya.__version__}; GPU {torch.cuda.get_device_name(0)}; base checkpoint {base}")

# %%
oof = {k: np.zeros(len(T)) for k in ("thr", "score", "entropy")}
te = {k: np.zeros(len(TE)) for k in ("thr", "score", "entropy")}
log, t0 = [], time.time()
for k in range(5):
    a, b = np.where(T.fold.values != k)[0], np.where(T.fold.values == k)[0]
    data = OUT / f"_train_fold{k}.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in gold_rows(a)), encoding="utf-8")
    for attempt, seed in enumerate((42 + k, 1042 + k)):
        out_dir = OUT / f"_ckpt_fold{k}"
        summary = finetune(str(data), base, str(out_dir), TrainConfig(seed=seed, **CFG))
        agent = laya.load(str(out_dir), device="cuda")
        thr_b, score_b, ent_b = predict(agent, [tr_states[i] for i in b])
        base_rmse = rmse(y[b], np.full(len(b), y[a].mean()))
        r = rmse(y[b], np.clip(thr_b, *CLIP))
        collapsed = bool(np.std(thr_b) < 0.1 or r >= 0.97 * base_rmse)
        log.append({"fold": k, "attempt": attempt, "seed": seed, "val_rmse_thr": r, "val_rmse_score": rmse(y[b], np.clip(score_b, *CLIP)),
                    "mean_baseline": base_rmse, "pred_std": float(np.std(thr_b)), "collapsed": collapsed,
                    "minutes": (time.time() - t0) / 60, "train_items": summary.get("train_items"),
                    "calibration_items": summary.get("calibration_items")})
        print(f"  fold {k} attempt {attempt}: val RMSE thresholds {r:.3f}, score {log[-1]['val_rmse_score']:.3f} "
              f"(mean baseline {base_rmse:.3f}){'  COLLAPSED' if collapsed else ''}  [{log[-1]['minutes']:.0f} min]", flush=True)
        (OUT / "laya_dh_folds.json").write_text(json.dumps(log, indent=1))
        if not collapsed or attempt == 1:
            oof["thr"][b], oof["score"][b], oof["entropy"][b] = thr_b, score_b, ent_b
            thr_t, score_t, ent_t = predict(agent, te_states)
            te["thr"] += thr_t / 5; te["score"] += score_t / 5; te["entropy"] += ent_t / 5
            del agent
            shutil.rmtree(out_dir, ignore_errors=True)
            gc.collect(); torch.cuda.empty_cache()
            break
        del agent
        shutil.rmtree(out_dir, ignore_errors=True)
        gc.collect(); torch.cuda.empty_cache()
    data.unlink(missing_ok=True)

pd.DataFrame({"split": ["train"] * len(T) + ["test"] * len(TE), "filename": list(T.filename) + list(TE.filename),
              "laya_dh_thr_pred": np.r_[oof["thr"], te["thr"]], "laya_dh_score_pred": np.r_[oof["score"], te["score"]],
              "laya_dh_entropy": np.r_[oof["entropy"], te["entropy"]]}).to_csv(OUT / "laya_dh.csv", index=False)
print(f"done {(time.time() - t0) / 60:.0f} min; OOF RMSE thresholds {rmse(y, np.clip(oof['thr'], *CLIP)):.4f}, "
      f"score {rmse(y, np.clip(oof['score'], *CLIP)):.4f}; collapsed folds kept {sum(e['collapsed'] for e in log if e['attempt'] == 1)}")
