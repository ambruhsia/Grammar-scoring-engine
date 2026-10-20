# %% [markdown]
# # Laya rubric checklist (Amrita's method, 2026-10-14): typed rubric questions per sentence, zero-shot
# Laya (convaiinnovations/laya, Apache-2.0, pinned revision) is a "System One" decision model: given a state and typed questions
# it returns calibrated probabilities over the options it was given. Here it is used that way, not as a regressor.
# - **Sentence units.** Whisper's own segments end mid-sentence 60% of the time, which would make "is this an incomplete
#   sentence?" fire on sentences that finish in the next segment. Units are therefore sentences rebuilt from Whisper's word-level
#   punctuation (a unit ends after a word ending in . ! ?); clips with no terminal punctuation fall back to Whisper's segments;
#   a unit over 60 words is split at Whisper segment boundaries. Each unit carries the CTC words spoken in its time span
#   (fillers removed), so the state shows the raw words next to Whisper's cleaned version.
# - **Checklist (yes/no, per unit):** six questions, one per rubric descriptor, written as concrete criteria, plus two
#   inference questions comparing the raw and clean versions (same message? only grammatical differences?), which separate
#   grammar repairs from CTC misrecognitions.
# - **Rubric score (per clip):** one score question with the five rubric levels verbatim, on the CTC text and on the Whisper text.
# No labels are used. Reads the cached transcripts from the artefacts dataset; Laya runs inside this session, so the text stays
# on Kaggle. Writes /kaggle/working/laya_units.csv (one row per unit: texts and the six probabilities) and laya_rubric_score.csv
# (per clip and text: the five level probabilities, expected level, entropy, tokens dropped).

# %%
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("laya") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "laya==0.4.7"], check=True)

# %%
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import laya

OUT = Path("/kaggle/working")
MODEL, REVISION = "convaiinnovations/laya", "7b928d828b7b0e022f929d9bd2e44165aa270148"
MAX_LEN, BATCH, MAX_UNIT_WORDS = 512, 32, 60
NON_LEXICAL = {"uh", "uhh", "um", "umm", "erm", "er", "ah", "ahh", "eh", "mm", "hmm", "hm", "uhm"}
_W = re.compile(r"[a-z']+")
assert torch.cuda.is_available(), "no CUDA device: the accelerator must be a GPU"
torch.manual_seed(42)


def find(name):
    hits = [Path(d) / name for d, _, files in os.walk("/kaggle/input", followlinks=True) if name in files]
    if not hits:
        raise FileNotFoundError(f"{name} not under /kaggle/input: attach the artefacts dataset")
    return min(hits, key=lambda p: len(p.parts))


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
n_words = lambda t: len([w for w in _W.findall((t or "").lower()) if w not in NON_LEXICAL])


def segment_of(t, segs):
    """Index of the Whisper segment containing time t (the nearest one if none does)."""
    inside = [i for i, s in enumerate(segs) if s["start"] - 0.01 <= t <= s["end"] + 0.01]
    return inside[0] if inside else int(np.argmin([min(abs(t - s["start"]), abs(t - s["end"])) for s in segs]))


def units(w, c):
    """Sentence units of one clip -> list of dicts (whisper text, ctc text, start, end)."""
    segs, words = w["segments"], w["words"]
    if not segs:
        return []
    groups = []
    if any(str(x[0]).strip().endswith((".", "!", "?")) for x in words):
        cur = []
        for x in words:
            cur.append(x)
            if str(x[0]).strip().endswith((".", "!", "?")):
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
    else:                                                    # no terminal punctuation anywhere: Whisper's segments
        by_seg = {}
        for x in words:
            by_seg.setdefault(segment_of((x[1] + x[2]) / 2, segs), []).append(x)
        groups = [by_seg[i] for i in sorted(by_seg)]
    split = []
    for g in groups:                                         # over-long units: split at Whisper segment boundaries
        if n_words(" ".join(str(x[0]) for x in g)) <= MAX_UNIT_WORDS:
            split.append(g)
            continue
        cur, cur_seg = [], None
        for x in g:
            s = segment_of((x[1] + x[2]) / 2, segs)
            if cur and s != cur_seg:
                split.append(cur)
                cur = []
            cur.append(x)
            cur_seg = s
        if cur:
            split.append(cur)
    out = [{"whisper": "".join(str(x[0]) for x in g).strip(), "start": float(g[0][1]), "end": float(g[-1][2]), "ctc": []}
           for g in split if n_words(" ".join(str(x[0]) for x in g)) > 0]
    if not out:
        return []
    for wd, st, en, *_ in c["words"]:                        # CTC words by midpoint (nearest unit if between units)
        wd = (wd or "").lower()
        if not _W.fullmatch(wd) or wd in NON_LEXICAL:
            continue
        m = (float(st) + float(en)) / 2
        inside = [i for i, u in enumerate(out) if u["start"] <= m <= u["end"]]
        j = inside[0] if inside else int(np.argmin([min(abs(m - u["start"]), abs(m - u["end"])) for u in out]))
        out[j]["ctc"].append(wd)
    for u in out:
        u["ctc"] = " ".join(u["ctc"])
    return out


U = []
for k in KEYS:
    for i, u in enumerate(units(W[k], C[k])):
        U.append({"split": k[0], "filename": k[1], "unit": i, "n_words": n_words(u["whisper"]), "start": u["start"],
                  "end": u["end"], "whisper_text": u["whisper"], "ctc_text": u["ctc"]})
U = pd.DataFrame(U)
print(f"{len(KEYS)} clips -> {len(U)} sentence units (per clip median {U.groupby(['split', 'filename']).size().median():.0f}); "
      f"words per unit median {U.n_words.median():.0f}, max {U.n_words.max()}; units with no CTC words {(U.ctc_text == '').mean():.1%}; "
      f"GPU {torch.cuda.get_device_name(0)}, laya {laya.__version__}")

# %%
IGNORE = " Ignore fillers, pauses, repeated words and misrecognised words."
CHECKLIST = {
    "error": {"type": "noul", "criteria": {"false": "grammatically correct as spoken", "true": "at least one grammatical error as spoken"},
              "instructions": "As the speaker said it (the raw words), does this sentence contain a grammatical error?" + IGNORE},
    "basic_error": {"type": "noul", "criteria": {"false": "no basic error", "true": "a basic agreement, tense, article or plural error"},
                    "instructions": "Does the speaker make a basic grammatical error here: wrong subject-verb agreement, wrong verb "
                                    "tense or form, a missing or wrong article, or a wrong plural?" + IGNORE},
    "incomplete": {"type": "noul", "criteria": {"false": "a complete sentence", "true": "incomplete or broken off"},
                   "instructions": "Is this an incomplete sentence: it breaks off before the thought is finished, or it has no "
                                   "main verb?" + IGNORE},
    "complex_ok": {"type": "noul", "criteria": {"false": "only simple structures, or complex ones used wrongly",
                                                "true": "a complex structure used correctly"},
                   "instructions": "Does the speaker correctly use a subordinate clause or another complex structure here (for "
                                   "example a because, if, when or although clause, a relative clause with who, which or that, or a "
                                   "passive)?" + IGNORE},
    "formulaic": {"type": "noul", "criteria": {"false": "the speaker's own construction", "true": "mostly a memorised or formulaic phrase"},
                  "instructions": "Is this sentence mostly a memorised or formulaic phrase rather than the speaker's own construction "
                                  "(for example 'in my opinion', 'I would like to talk about', 'that's all')?"},
    "selfcorrect": {"type": "noul", "criteria": {"false": "no self-correction", "true": "the speaker corrects their own grammatical mistake"},
                    "instructions": "Does the speaker correct a grammatical mistake themselves: they say a wrong form, then restate it "
                                    "correctly (for example 'he go, he goes')? Repeating the same words is not a correction."},
    # raw vs clean (inference questions, Laya's strength): a grammar repair keeps the message, a misrecognition changes it
    "same_meaning": {"type": "noul", "criteria": {"false": "a different message: misheard, missing or extra content words",
                                                  "true": "the same message, even if the grammar or wording differs slightly"},
                     "instructions": "Do the raw words and the clean transcript express the same message?"},
    "grammar_only": {"type": "noul", "criteria": {"false": "some differences are misheard or different content words",
                                                  "true": "every difference is grammatical, or there is none"},
                     "instructions": "Are all the differences between the raw words and the clean transcript grammatical (verb "
                                     "forms, agreement, articles, plurals, word order), with no misheard or different content words?"},
}
RUBRIC = [
    "The person's speech struggles with proper sentence structure and syntax, displaying limited control over simple grammatical structures and memorized sentence patterns.",
    "The person has a limited understanding of sentence structure and syntax. Although they use simple structures, they consistently make basic sentence structure and grammatical mistakes. They might leave sentences incomplete.",
    "The person demonstrates a decent grasp of sentence structure but makes errors in grammatical structure, or they show a decent grasp of grammatical structure but make errors in sentence syntax and structure.",
    "The person displays a strong understanding of sentence structure and syntax. They consistently show good control of grammar. While occasional errors may occur, they are generally minor and do not lead to misunderstandings; the person can correct most of them.",
    "Overall, the person showcases high grammatical accuracy and adept control of complex grammar. They use grammar accurately and effectively, seldom making noticeable mistakes. Additionally, they handle complex language structures well and correct themselves when necessary.",
]
SCORE = {"rubric": {"type": "score", "instructions": "Rate the speaker's grammar on this rubric." + IGNORE, "criteria": RUBRIC}}

agent = laya.load(MODEL, device="cuda", revision=REVISION)
t0 = time.time()
states = [f"Raw (the speaker's exact words; no punctuation, some words may be misrecognised): {r.ctc_text or '(not recognised)'}\n"
          f"Clean (the same words as a cleaned transcript): {r.whisper_text}" for r in U.itertuples()]
res = agent.predict_batch(states, CHECKLIST, batch_size=BATCH, max_len=MAX_LEN)
assert len(res) == len(U)
for q in CHECKLIST:
    U[f"p_{q}"] = [r["answers"][q]["noul"] for r in res]
U["tokens_dropped"] = [r["usage"]["state_tokens_dropped"] for r in res]
U.to_csv(OUT / "laya_units.csv", index=False)
print(f"checklist: {len(U)} units x {len(CHECKLIST)} questions in {(time.time() - t0) / 60:.1f} min; units truncated "
      f"{(U.tokens_dropped > 0).mean():.1%}", flush=True)
print(U[[f"p_{q}" for q in CHECKLIST]].describe().round(3).to_string())

# %%
t0 = time.time()
S = {"split": [k[0] for k in KEYS], "filename": [k[1] for k in KEYS]}
texts = {"ctc": ["Automatic transcript of a spoken English answer (no punctuation; some words may be misrecognised): "
                 + (C[k]["raw_text"] or "") for k in KEYS],
         "whisper": ["Automatic transcript of a spoken English answer: " + (W[k]["text"] or "") for k in KEYS]}
for text, st in texts.items():
    # the five verbatim rubric levels take ~230 tokens: a 320-token option budget keeps them whole, and max_len 1024 leaves
    # room for the transcript (ModernBERT's positions go to 8192; Laya's English checkpoint was trained at 512)
    res = agent.predict_batch(st, SCORE, batch_size=8, max_len=1024, head_max_len=320)
    P = np.array([[r["answers"]["rubric"]["probabilities"][str(i)] for i in range(5)] for r in res])
    for i in range(5):
        S[f"{text}_p{i}"] = P[:, i]
    S[f"{text}_ev"] = P @ np.arange(1, 6)                    # expected level on the 1-5 scale
    S[f"{text}_entropy"] = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)
    S[f"{text}_tokens_dropped"] = [r["usage"]["state_tokens_dropped"] for r in res]
    print(f"rubric score on {text}: {(time.time() - t0) / 60:.1f} min; clips truncated "
          f"{np.mean(np.array(S[f'{text}_tokens_dropped']) > 0):.1%}", flush=True)
pd.DataFrame(S).to_csv(OUT / "laya_rubric_score.csv", index=False)
print("done")
