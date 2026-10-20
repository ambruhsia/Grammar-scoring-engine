# %% [markdown]
# # Grammar scoring of spoken English: speaker-honest validation and an LLM examiner
# **SHL Hiring Assessment 2026** · Amrita (Kaggle: ambruhsia)
#
# **TL;DR.**
# 1. Two stacked pipelines (speech encoders, an audio LLM, transcript fine-tunes and LLM heads, rubric features) are blended per
#    recording batch, corrected by a zero-shot LLM examiner, bagged over 4 fold-seed sets and combined per batch.
# 2. Validated on unseen speakers at the test's batch mix: RMSE 0.501, Pearson 0.875; TRAINING RMSE 0.480; public leaderboard
#    0.3279 (a second pick with a test-set offset: 0.3271, ~23rd of 448).
# 3. What moved the score most: fixing validation (a leaky member), averaging two pipelines on short clips, and the examiner
#    correction. Short clips and unseen prompts remain the hardest part (section 9).
#
# ## 0. Report
# **Problem.** Predict a 1–5 grammar score (0.5 steps) for 45–60 s spoken-English clips, from the audio. The leaderboard metric,
# decoded from probe submissions, is **0.6 × RMSE + 0.4 × (1 − Pearson)**; the private board (≈86 test clips) decides the ranking,
# so the work optimises validation that looks like the test set rather than the public board (≈130 clips, ≈0.003 per clip).
#
# **Data and preprocessing** (sections 2–4).
# - *Audio:* read as 16 kHz mono (appendix A0). 37 training clips (IDs 5037–5073) are static noise labelled 0; a spectral-flatness
#   detector built from them flags no test clip, so they are removed (732 clips remain).
# - *Transcription:* three systems with different error profiles: Whisper (clean, punctuated text), a CTC model (raw text that keeps
#   the speaker's errors, no language model) and Whisper-large-v3-turbo with a verbatim prompt (errors and fillers kept, spelled
#   correctly). Handcrafted fluency and text features and the rubric layer (section 4) are computed from them.
# - *Speakers:* 53% of training clips share a voice with another training clip (x-vector cosine > 0.988); twins' grades differ by
#   0.24 vs 1.15 for random pairs. Every test speaker is new, so folds are **grouped by speaker**.
# - *Recording batches:* a ~45 s and a ~60 s batch (cut at 47.7 s): 24% of train but 68% of test are short. Each batch gets a
#   **test-mix weight** (test share / train share) in validation and blending.
# - *Prompts:* test answers prompts the training set barely covers ("floods" is 10% of test and absent from train), so a
#   topic-held-out check is part of validation.
#
# **Pipeline.**
# ```
#  audio ──► Whisper (clean) / CTC (raw) / Whisper-large-v3-turbo (verbatim) transcripts
#             ├─► rubric layer + Laya checklist questions (interpretable features)
#             ├─► DeBERTa-v3 fine-tuned on (raw, clean) transcript pairs
#             └─► Qwen2.5-7B / Qwen3.5-9B rubric-prompt hidden states → ridge heads
#  audio ──► WavLM-base / WavLM-large / Whisper-medium encoders → ridge and SVR heads
#  audio ──► Voxtral-Mini-3B audio LLM: states over the audio tokens (grammar question after or before the audio)
#  Qwen + speech-encoder embeddings ─► Rank-N-Contrast embedding → ridge
#       every learned column out-of-fold on SPEAKER-GROUPED folds
#       → 9 model families → NNLS blend per recording batch → examiner correction (Qwen3.5-9B zero-shot rubric judgements)
#       → same-speaker pooling → base + combination pipelines × 4 fold-seed sets, combined per batch
# ```
#
# **Validation.** StratifiedGroupKFold on speaker groups; primary metric `cv_testmix` (RMSE with each batch weighted to its test
# share). Every learned column is out-of-fold on the grouped folds, and blend weights and the correction are refit inside each
# fold. A change was kept only if it also held on **fresh fold seeds** and on the **topic-held-out** check (computed after the
# correction stage too); the public board was the last, noisy check.
#
# **Results** (section 7 recomputes them):
#
# | TRAINING RMSE | cv_grouped | **cv_testmix** | cv_short | cv_long | Pearson | public |
# |---|---|---|---|---|---|---|
# | 0.480 | 0.4917 | **0.5013** | 0.5082 | 0.4865 | 0.875 | 0.3279 |
#
# The public score is consistent with the CV: 130-clip resamples of the out-of-fold predictions, scored with the leaderboard's
# metric at the test batch mix, give 0.306–0.420 (95%). The second pick (section 8) subtracts a −0.041 test-set offset measured on
# the public board (0.3271); it is the only step tuned on the public board and is not part of the model or its CV.
#
# **How this notebook runs.** On a CPU in about a minute. *Computed here:* batches, speaker groups, folds, handcrafted and rubric
# features, rubric tests, the stacking of all 8 runs (blends, examiner correction, pooling, combination), every metric
# and figure; each replayed run is asserted equal to the run's own files, and the submission byte-identical to the submitted one.
# *Read from cache:* the transcripts, the speech embeddings and the learned members' out-of-fold predictions (each run's
# `cands_<run>.npz`), because they need the audio and a GPU. Appendix A holds the code for those stages (`RECOMPUTE_GPU`) and
# lists the GPU notebooks that produced the rest.
#
# **Key findings** (details and numbers in section 9).
# 1. *Validation, not model capacity, was the main lever:* a leaky member had inflated CV by 0.02; removing it was the largest
#    single public gain.
# 2. *Short clips:* new members improved only long clips on fresh folds; averaging two independent pipelines was the one short-clip
#    gain that held.
# 3. *The examiner sees what the stack misses on short clips* (partial r +0.10 given the final prediction), but 174 short training
#    clips are too few to learn more of it without fitting their prompts.
# 4. *Fairness:* when the CTC transcriber fails (more often on accented speech), good speakers are under-scored by 0.69.
# 5. *Raters' behaviour:* restarts go with lower grades (they are disfluency, not self-correction), and the model rewards the amount
#    spoken on short clips while the raters don't.
#
# **Limitations.** Short clips stay compressed toward the middle (true 5.0 → predicted 4.3; RMSE 0.76 there). The transcript
# fine-tunes are out-of-fold on the grouped seed-42 folds only (slightly optimistic on the other fold seeds). The examiner correction
# is close to neutral under prompt shift (0.5143 → 0.5149); it is kept because a matched public A/B favours it (0.3280 vs 0.3335).
# The CoLA classifier's model card states no licence; every other model allows commercial use.
#
# **Models and licences** (checked on the model cards):
#
# | Stage | Model | Licence |
# |---|---|---|
# | ASR, clean text (appendix A1) | Whisper small.en, faster-whisper conversion | MIT |
# | ASR, raw text (A2) | facebook/wav2vec2-large-robust-ft-swbd-300h | Apache-2.0 |
# | ASR, verbatim text (job shl-verbatim-text) | Whisper-large-v3-turbo, faster-whisper conversion | MIT |
# | Raw-text fluency (A3) | GPT-2 | MIT |
# | Sentence acceptability (A3) | textattack/roberta-base-CoLA | not stated on the model card |
# | Sentence embeddings (A3) | sentence-transformers/all-mpnet-base-v2 | Apache-2.0 |
# | Speech encoders (A4) | microsoft/wavlm-base-plus, microsoft/wavlm-large | CC BY-SA 3.0 (UniSpeech repo) |
# | Speech encoder (A4) | openai/whisper-medium, encoder only | Apache-2.0 |
# | Speaker x-vectors (A5) | microsoft/wavlm-base-plus-sv | CC BY-SA 3.0 (UniSpeech repo) |
# | Audio LLM (jobs shl-voxtral-*) | mistralai/Voxtral-Mini-3B-2507 | Apache-2.0 |
# | Transcript regressors (A7, shl-deberta-large-pair) | microsoft/deberta-v3-base, microsoft/deberta-v3-large | MIT |
# | Rubric-prompt features (A8) | Qwen/Qwen2.5-7B-Instruct, 4-bit | Apache-2.0 |
# | Examiner (shl-qwen35-judge, -judge2) | Qwen/Qwen3.5-9B, NF4 | Apache-2.0 |
# | Checklist questions (shl-laya-checklist) | convaiinnovations/laya | Apache-2.0 |
# | Parsing (section 4) | spaCy en_core_web_sm 3.8.0 | MIT |
# | Grammar checker (cached) | LanguageTool 6.5, local server | LGPL-2.1 |
# | Modelling | scikit-learn, LightGBM, XGBoost, PyTorch (Rank-N-Contrast) | BSD-3, MIT, Apache-2.0, BSD-3 |
#
# **Credits.** Ideas from other participants' public write-ups (the code here is written from scratch): speaker twins, label
# denoising toward the speaker mean, and Qwen2.5-7B hidden states on the CTC transcript (starkhushi); the two recording batches and
# per-batch blend weights (PJ2001-IND); 30-second audio crops (devyanshbatra); an audio LLM's hidden states (Voxtral), the decoded
# leaderboard metric, bias probes and the averaging identity (Techy-A); a Qwen3.5-9B rubric judge (AMITABHSHUKLA); question-first
# Voxtral views and the DeBERTa-v3-large recipe (lijoraju); same-speaker pooling (nakulgrover118, Anurag Nayak); a verbatim
# Whisper prompt (Techy-A, PRINCE2-AI).

# %% [markdown]
# ## 1. Setup and data loading
# All imports and constants are in the next cell. Only the appendix's audio and GPU libraries (torch, transformers,
# faster-whisper, librosa) are imported inside its functions, because this notebook runs on a CPU without them. Inputs: the
# competition data, the final pick's artefacts (`shl-grammar-final-artefacts`: each of the 8 runs' candidate matrices and files,
# the examiner's judge files, the folds) and the transcript / embedding caches (`shl-grammar-caches-v18`).
# - **Default:** everything that runs on a CPU is computed here: batches, speaker groups, handcrafted and rubric features, rubric
#   tests, the stacking of all 8 runs (blends, examiner correction, pooling, the combination) and every metric. The learned
#   members' out-of-fold predictions are read from the runs' candidate files.
# - `RECOMPUTE_GPU = True` also runs the appendix's ported stages (transcription, speech encoders, DeBERTa-v3-base, Qwen2.5);
#   the Phase I GPU jobs (Voxtral, Qwen3.5-9B examiner, DeBERTa-v3-large, verbatim transcripts) are listed there with their
#   notebooks. Files land in `/kaggle/working/recomputed/` under the names the sections read.

# %%
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.io import wavfile
from scipy.optimize import nnls
from scipy.stats import gaussian_kde, mannwhitneyu, pearsonr, rankdata, spearmanr
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import adjusted_rand_score, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# ---- run and paths
FINAL_PICK = "hybrid11"            # public 0.3279: short clips = mean of both bags, long clips = combination bag
TEST_OFFSET = -0.041               # second pick (pubmix_c, public 0.3271): hybrid11 minus a test-set offset from the public board
BASE_RUNS = ("1015-070513", "1015-114531", "1015-165432", "1015-172136")     # base pipeline on 4 fold-seed sets
COMBO_RUNS = ("1016-100139", "1016-140135", "1016-140601", "1016-141007")    # combination pipeline on the same 4 sets
SHRINK_LAMBDA, POOL_S = 0.25, 0.3  # combination: long clips' weight in the short-clip blend; same-speaker pooling strength
RECOMPUTE_GPU = False              # run the appendix's GPU stages (GPU + internet, hours); default: cached outputs
FLOAT_FORMAT = "%.10f"             # submission format: floating-point order differs across machines at the 1e-15 level
# ---- data and validation
SEED = 42
SEEDS = (42, 1337, 2024)
CLIP = (1.5, 5.0)
LEVELS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
BATCH_MODES = (45.0, 60.0)         # the two recording-batch duration modes; the cut is the density minimum between them
TWIN_PCT = 99                      # percentile of test->train nearest x-vector similarity used as the same-voice threshold
DENOISE_LAMBDA = 0.5               # members fit on (1 - l) * y + l * speaker-group mean; always scored on y
# ---- transcripts and rubric layer
NON_LEXICAL = {"uh", "uhh", "um", "umm", "erm", "er", "ah", "ahh", "eh", "mm", "hmm", "hm", "uhm"}
FINITE = {"VBD", "VBP", "VBZ", "MD"}
SUB_DEPS = {"advcl", "ccomp", "csubj", "relcl"}
MIN_PAUSE_S, PAUSE_CAP_S, BOUNDARY_TOL_S, THOUGHT_GRACE = 0.25, 3.00, 0.20, 1
DUR_MAX, PARTIAL_MIN, MAX_INVERSIONS = 0.2, 0.05, 1     # rubric-layer selection checks 3, 4 and 2
ALPHA = 0.05                       # rubric conformance tests
CTC_FAIL_CUT = 0.6                 # repair_wer_nofiller above this = the CTC transcript failed
K_TOPICS = 10                      # prompt clusters (k-means on sentence embeddings, no labels)
# ---- public-board comparison
PUBLIC_SCORE, N_PUBLIC, N_RESAMPLE = 0.3279, 130, 2000   # public score of hybrid11 (before the offset); ~130 public clips
# ---- appendix (audio / GPU stages)
RAW = Path("/kaggle/working/recomputed")
SR = 16_000

np.random.seed(SEED)
RUNTIME = {}
_t0 = time.time()
_base = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path(".")      # local runs: the project root
LOCAL_DIRS = [_base / p for p in ("kaggle/final/artefacts", "cache", "cache/qwen")]
_W = re.compile(r"[a-z']+")


def find(name):
    """Path of a cached artefact or competition file (under /kaggle/input on Kaggle, the local cache otherwise)."""
    roots = [Path("/kaggle/input")] if Path("/kaggle/input").exists() else LOCAL_DIRS
    for root in roots:
        hits = [Path(d) / name for d, _, files in os.walk(root, followlinks=True) if name in files]
        if hits:
            return min(hits, key=lambda p: len(p.parts))
    raise FileNotFoundError(f"{name}: attach the artefacts datasets")


def load_jsonl(path):
    """jsonl -> {(split, filename): record}."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[(r["split"], r["filename"])] = r
    return out


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def wrmse(a, b, w):
    a, b, w = (np.asarray(v, float) for v in (a, b, w))
    return float(np.sqrt(np.sum(w * (a - b) ** 2) / np.sum(w)))


def max_abs_diff(a, b):
    """Largest |a - b|, treating matching NaNs as equal (a NaN on one side only counts as infinite)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if (np.isnan(a) != np.isnan(b)).any():
        return float("inf")
    ok = ~np.isnan(a)
    return float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else 0.0


train_all = pd.read_csv(find("train.csv"))
test = pd.read_csv(find("test.csv"))
train = train_all[train_all.label > 0].reset_index(drop=True)
F = pd.read_csv(find("features.csv"))
G = pd.read_csv(find("speaker_groups.csv"))
C = np.load(find(f"cands_{BASE_RUNS[0]}.npz"), allow_pickle=True)    # row order, labels, batches (shared by all 8 runs)
W, Cc = load_jsonl(find("asr_c.jsonl")), load_jsonl(find("ctc.jsonl"))
T0 = F[F.split == "train"].merge(train, on="filename").reset_index(drop=True)     # the run's row order
TE0 = F[F.split == "test"].reset_index(drop=True)
assert list(T0.filename) == list(C["filename"]) and list(TE0.filename) == list(C["test_filename"]), "row order differs from the run"
keys = list(zip(F.split, F.filename))
print(f"train {len(train_all)} rows -> {len(train)} after removing {len(train_all) - len(train)} sentinel clips; test {len(test)}")
RUNTIME["1 setup"] = time.time() - _t0

# %% [markdown]
# ## 2. Preprocessing
# **Computed here**, from the clip durations and the cached x-vectors, and checked against the run's own copies: the
# recording-batch cut (the minimum of the duration density between the 45 s and 60 s modes), each clip's batch, the test-mix
# weights (test share / train share of the clip's batch), the speaker groups (average-linkage clustering of the training x-vectors at
# cosine distance 1 − t, t = the 99th percentile of test→train nearest similarity) and the speaker-grouped folds.
# **Read from cache**, because they need the audio: the durations (Whisper, appendix A1), the x-vectors (appendix A5) and the
# spectral flatness of the sentinel check (appendix A0). Audio is read as 16 kHz mono (appendix A0).

# %%
_t = time.time()
dur_all = np.r_[T0.duration.values, TE0.duration.values]
xs = np.linspace(BATCH_MODES[0] + 1, BATCH_MODES[1] - 1, 1401)
batch_cut = float(xs[np.argmin(gaussian_kde(dur_all, bw_method=0.05)(xs))])
batch_tr = np.where(T0.duration < batch_cut, "short", "long")
batch_te = np.where(TE0.duration < batch_cut, "short", "long")
p_tr, p_te = pd.Series(batch_tr).value_counts(normalize=True), pd.Series(batch_te).value_counts(normalize=True)
w_live = np.array([p_te[b] / p_tr[b] for b in batch_tr])
w_live /= w_live.mean()
assert (batch_tr == C["batch"]).all() and (batch_te == C["batch_test"]).all() and np.allclose(w_live, C["w"]), "batches differ"
print(f"batch cut {batch_cut:.2f} s; short share train {p_tr['short']:.1%}, test {p_te['short']:.1%}; "
      f"test-mix weights short {w_live[batch_tr == 'short'][0]:.3f}, long {w_live[batch_tr == 'long'][0]:.3f}")

zx = np.load(find("spk_xvec.npz"))
ix_x = {k_: i for i, k_ in enumerate(zip(zx["split"], zx["filename"]))}
E = zx["E"].astype(np.float64)
E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
Xv_tr, Xv_te = E[[ix_x[("train", f)] for f in T0.filename]], E[[ix_x[("test", f)] for f in TE0.filename]]
nn_test = (Xv_te @ Xv_tr.T).max(1)
twin_t = float(np.percentile(nn_test, TWIN_PCT))
groups = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                 distance_threshold=1.0 - twin_t).fit_predict(Xv_tr)
groups_run = G[G.split == "train"].set_index("filename").loc[T0.filename, "group"].values
assert adjusted_rand_score(groups_run, groups) == 1.0, "speaker groups differ from the run"
St = Xv_tr @ Xv_tr.T
np.fill_diagonal(St, -1)
y0 = T0.label.values.astype(float)
iu = np.triu_indices(len(y0), 1)
twin_pair = St[iu] > twin_t
sizes = pd.Series(groups).value_counts()
print(f"twin threshold t = {twin_t:.3f}; speaker groups {sizes.size}, train clips with a twin "
      f"{(pd.Series(groups).map(sizes) >= 2).mean():.1%}, largest {sizes.max()}; label gap: twin pairs "
      f"{np.abs(y0[iu[0]] - y0[iu[1]])[twin_pair].mean():.3f} ({twin_pair.sum()} pairs) vs random pairs "
      f"{np.abs(y0[iu[0]] - y0[iu[1]]).mean():.3f}")

strat = T0.label.replace({1.0: 1.5}).astype(str)
FD = pd.read_csv(find("folds_all.csv"))       # the runs' speaker-grouped folds for all 12 fold seeds of the final pick
FOLDS, same_folds = {}, True
for s in sorted(FD.seed.unique()):
    fs = FD[FD.seed == s].set_index("filename").loc[T0.filename, "fold"].values
    FOLDS[int(s)] = [(np.where(fs != k_)[0], np.where(fs == k_)[0]) for k_ in range(5)]
    with warnings.catch_warnings():               # four clips below 2.0 form a class smaller than the 5 folds (as in the run)
        warnings.simplefilter("ignore", UserWarning)
        live = list(StratifiedGroupKFold(5, shuffle=True, random_state=int(s)).split(T0, strat, groups_run))
    same_folds &= all(np.array_equal(np.sort(b), np.sort(lb)) for (_, b), (_, lb) in zip(FOLDS[int(s)], live))
import sklearn
print(f"speaker-grouped folds rebuilt here (scikit-learn {sklearn.__version__}; the run used 1.9.1) match the run's folds: "
      f"{same_folds}. Every number below uses the run's folds.")

S = pd.read_csv(find("sentinel_guard.csv"))
fig, axes = plt.subplots(1, 3, figsize=(16, 3.6))
for kind, c in (("normal", "#2e86c1"), ("sentinel", "#c0392b"), ("test", "#7f8c8d")):
    axes[0].hist(S.loc[S.kind == kind, "flatness"], bins=40, alpha=0.6, label=kind, color=c)
axes[0].set(title="Sentinel detector: spectral flatness", xlabel="median spectral flatness"); axes[0].legend(frameon=False)
for d, lab, c in ((T0.duration, "train", "#2e86c1"), (TE0.duration, "test", "#e67e22")):
    axes[1].hist(d, bins=np.arange(0, 63), alpha=0.6, density=True, label=lab, color=c)
axes[1].axvline(batch_cut, color="k", ls="--", lw=1); axes[1].set(title="Two recording batches", xlabel="duration (s)")
axes[1].legend(frameon=False)
axes[2].hist(St.max(1), bins=50, alpha=0.6, density=True, label="train → nearest other train", color="#2e86c1")
axes[2].hist(nn_test, bins=50, alpha=0.6, density=True, label="test → nearest train", color="#e67e22")
axes[2].axvline(twin_t, color="k", ls="--", lw=1); axes[2].set(title="Speaker twins (x-vector cosine)", xlabel="cosine similarity")
axes[2].legend(frameon=False, fontsize=8)
fig.tight_layout(); plt.show()
flat_cut = (S.loc[S.kind == "sentinel", "flatness"].min() + S.loc[S.kind == "normal", "flatness"].max()) / 2
print(f"sentinel check (cached flatness): cut {flat_cut:.3f}; test clips above it: {int((S.loc[S.kind == 'test', 'flatness'] > flat_cut).sum())}")
RUNTIME["2 preprocessing"] = time.time() - _t

# %% [markdown]
# ## 3. Transcription and handcrafted features
# Two ASR passes, read from cache (appendix A1, A2): Whisper gives clean, punctuated text (it repairs grammar); a CTC model gives
# raw text that keeps the speaker's errors. **Computed here** from the two transcripts: the handcrafted fluency and text features
# and the Whisper–CTC disagreement `repair_wer_nofiller` (fillers removed), the strongest single handcrafted accuracy signal. Pauses
# count only mid-clause, and the longest one per clip is forgiven (thinking time is not a grammar error). Each column is checked
# against the run's `features.csv`. **Read from cache:** the GPT-2 and CoLA columns (GPU, appendix A3).

# %%
_t = time.time()
norm = lambda t: _W.findall((t or "").lower())
SUBORDINATORS = {"because", "although", "though", "while", "whereas", "since", "unless", "until", "whether", "if", "when",
                 "after", "before", "that", "which", "who", "whom", "whose"}
CONNECTIVES = {"however", "therefore", "moreover", "furthermore", "nevertheless", "consequently", "meanwhile", "otherwise",
               "instead", "thus"}
MODALS = {"would", "could", "should", "might", "must", "may", "shall", "ought"}


def lev(a, b):
    """Word-level Levenshtein distance."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def pause_feats(words_, bounds, dur):
    """Pauses >= 0.25 s count only mid-clause (not at a sentence boundary); the longest mid-clause pause is forgiven."""
    out = dict(midclause_pause_ratio=0.0, midclause_pause_per_min=0.0, boundary_share=1.0, articulation_rate=0.0,
               phonation_ratio=0.0)
    if len(words_) < 3 or dur <= 0:
        return out
    st, en = np.array([w[1] for w in words_], float), np.array([w[2] for w in words_], float)
    span = max(en[-1] - st[0], 1e-6)
    phon = float(np.clip(en - st, 0, None).sum())
    B = np.asarray(bounds, float)
    natural, mid = [], []
    for g, t in zip(st[1:] - en[:-1], en[:-1]):
        if g < MIN_PAUSE_S:
            continue
        at_b = B.size and np.min(np.abs(B - t)) <= BOUNDARY_TOL_S
        (natural if at_b else mid).append(min(float(g), PAUSE_CAP_S))
    mid.sort(reverse=True)
    counted, n_all = mid[THOUGHT_GRACE:], len(mid) + len(natural)
    out["midclause_pause_ratio"] = sum(counted) / span
    out["midclause_pause_per_min"] = len(counted) / (span / 60.0)
    out["boundary_share"] = len(natural) / n_all if n_all else 1.0
    out["articulation_rate"] = len(words_) / max(phon, 1e-6)
    out["phonation_ratio"] = phon / span
    return out


def text_feats(clean):
    toks = norm(clean)
    content = [t for t in toks if t not in NON_LEXICAL]
    n = max(len(toks), 1)
    sents = [s for s in re.split(r"[.!?]+", clean or "") if s.strip()]
    rep1 = sum(1 for a, b in zip(toks, toks[1:]) if a == b and len(a) > 1)
    rep2 = sum(1 for i in range(len(toks) - 3) if toks[i:i + 2] == toks[i + 2:i + 4])
    return {"n_words": float(len(toks)), "text_ttr": len(set(content)) / max(len(content), 1),
            "mean_sent_len": len(toks) / max(len(sents), 1),
            "std_sent_len": float(np.std([len(norm(s)) for s in sents])) if sents else 0.0,
            "subordinator_rate": sum(t in SUBORDINATORS for t in toks) / n, "connective_rate": sum(t in CONNECTIVES for t in toks) / n,
            "modal_rate": sum(t in MODALS for t in toks) / n, "repetition_rate": (rep1 + rep2) / n,
            "mean_word_len": float(np.mean([len(t) for t in content])) if content else 0.0,
            "long_word_rate": sum(len(t) >= 8 for t in content) / max(len(content), 1)}


def handcrafted(w, c):
    f = {"duration": w["duration"]}
    bounds = [s["end"] for s in w["segments"] if s["text"].strip().endswith((".", "!", "?"))]
    f.update(pause_feats(w["words"], bounds, w["duration"]))
    f.update(text_feats(w["text"]))
    wt_nf, ct_nf = ([t for t in norm(x) if t not in NON_LEXICAL] for x in (w["text"], c["raw_text"]))
    f["repair_wer_nofiller"] = min(lev(ct_nf, wt_nf) / max(len(ct_nf), 1), 2.0)
    f["asr_avg_logprob"] = float(np.mean([s["avg_logprob"] for s in w["segments"]])) if w["segments"] else 0.0
    f["no_punct"] = float(not any(s["text"].strip().endswith((".", "!", "?")) for s in w["segments"]))
    return f


HC = pd.DataFrame([handcrafted(W[k_], Cc[k_]) for k_ in keys])
hc_diff = {c: max_abs_diff(HC[c], F[c]) for c in HC.columns}
assert max(hc_diff.values()) < 1e-9, {c: d for c, d in hc_diff.items() if d >= 1e-9}
print(f"{len(HC.columns)} handcrafted columns computed from the transcripts; max |difference| from features.csv "
      f"{max(hc_diff.values()):.1e}")
k = ("train", T0.filename.iloc[0])
print("Whisper:", W[k]["text"][:220]); print("CTC    :", Cc[k]["raw_text"][:220])
print(f"repair_wer_nofiller for this clip: {HC.loc[keys.index(k), 'repair_wer_nofiller']:.3f}")
RUNTIME["3 transcription + handcrafted"] = time.time() - _t

# %% [markdown]
# ## 4. Rubric layer
# | Rubric wording | Features | Expected sign |
# |---|---|---|
# | sentence structure and syntax (all levels) | `cola_frac_bad`, `gpt2_nll_gap`, `finite_clause_ratio`, `mean_len_clause` | −, −, +, + |
# | basic mistakes (2) / errors (3) / occasional minor errors (4) / high accuracy (5) | `repair_wer_nofiller`, `lt_grammar_per100`, `llm_err_per100` | −, −, − |
# | memorized patterns (1), simple structures (2), complex structures (5) | `dep_clause_ratio`, `clauses_per_sentence`, `mean_dep_distance`, `mattr50`, `trigram_repeat_rate` | +, +, +, +, − |
# | might leave sentences incomplete (2) | `no_punct`, `fragment_ratio` | −, − |
# | can correct / correct themselves (4–5) | not measurable from transcripts (Finding 1): `corrected_error_ratio` and the restart rate `self_repair_per100` are reported, never modelled | (+) |
#
# A feature joins the rubric layer only if (1) its sign matches, (2) its mean by level is monotone with ≤ 1 inversion,
# (3) |Spearman with duration| < 0.2 in each batch, and (4) |partial Spearman with the label, controlling for repair_wer| ≥ 0.05.
#
# **Computed here** from the cached transcripts: the spaCy structure features (one parse per Whisper sentence; "finite" = tag
# VBD / VBP / VBZ / MD), lexical range and repetition (Whisper text), the restart rate (CTC words), and the error density and
# self-correction ratio from Qwen's cached minimal correction of the CTC text; then the selection table and the rubric conformance
# tests. **Read from cache:** LanguageTool grammar matches (`lt_grammar_per100`; LanguageTool 6.5 needs Java, so it ran locally),
# CoLA and GPT-2 (appendix A3), and the zero-shot rubric judge's probabilities (appendix A8).

# %%
_t = time.time()


def words(text):
    return [t for t in _W.findall((text or "").lower()) if t not in NON_LEXICAL]


def mattr(ws, win=50):
    """Moving-average type-token ratio, window 50 words."""
    if not ws:
        return np.nan
    if len(ws) <= win:
        return len(set(ws)) / len(ws)
    c = Counter(ws[:win])
    vals = [len(c)]
    for i in range(win, len(ws)):
        c[ws[i]] += 1
        c[ws[i - win]] -= 1
        if c[ws[i - win]] == 0:
            del c[ws[i - win]]
        vals.append(len(c))
    return float(np.mean(vals) / win)


def trigram_repeat(ws):
    tg = list(zip(ws, ws[1:], ws[2:]))
    if not tg:
        return np.nan
    c = Counter(tg)
    return sum(c[t] > 1 for t in tg) / len(tg)


def is_finite(tok):
    return tok.tag_ in FINITE or any(c.dep_ in ("aux", "auxpass") and c.tag_ in FINITE for c in tok.children)


def structure(text, nlp):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if words(s)]
    ws = words(text)
    out = {"mattr50": mattr(ws), "trigram_repeat_rate": trigram_repeat(ws)}
    names_ = ("finite_clause_ratio", "mean_len_clause", "dep_clause_ratio", "clauses_per_sentence", "mean_dep_distance",
              "fragment_ratio")
    if not sents:
        return {**out, **{k_: np.nan for k_ in names_}}
    n_fin = n_sub = n_sent_fin = n_frag = 0
    dists = []
    for doc in nlp.pipe(sents):
        toks = [t for t in doc if not t.is_punct and not t.is_space and t.lower_ not in NON_LEXICAL]
        fin = sum(t.tag_ in FINITE for t in toks)
        n_fin += fin
        n_sent_fin += fin > 0
        n_sub += sum(t.dep_ in SUB_DEPS or (t.dep_ == "acl" and is_finite(t)) for t in toks)
        if fin == 0 and not any(r.pos_ in ("VERB", "AUX") for r in doc if r.dep_ == "ROOT"):
            n_frag += 1
        dists += [abs(t.head.i - t.i) for t in toks if t.dep_ != "ROOT"]
    n = len(sents)
    return {**out, "finite_clause_ratio": n_sent_fin / n, "mean_len_clause": len(ws) / n_fin if n_fin else np.nan,
            "dep_clause_ratio": n_sub / n_fin if n_fin else np.nan, "clauses_per_sentence": n_fin / n,
            "mean_dep_distance": float(np.mean(dists)) if dists else np.nan, "fragment_ratio": n_frag / n}


def ctc_chunks(ctc_words, segments):
    """CTC words (fillers removed) grouped by the Whisper segment their midpoint falls in (nearest if none)."""
    segs = [(float(s["start"]), float(s["end"])) for s in segments] or [(0.0, 1e9)]
    mids = np.array([(a + b) / 2 for a, b in segs])
    chunks = [[] for _ in segs]
    for w, st, en, *_ in ctc_words:
        w = (w or "").lower()
        if not _W.fullmatch(w) or w in NON_LEXICAL:
            continue
        m = (float(st) + float(en)) / 2
        inside = [i for i, (a, b) in enumerate(segs) if a <= m <= b]
        chunks[inside[0] if inside else int(np.argmin(np.abs(mids - m)))].append(w)
    return [c for c in chunks if c]


def self_repairs(ws):
    """Retraces with a change (a word repeated 2-3 tokens later, different continuation) + partial-word restarts."""
    n = 0
    for i, t in enumerate(ws):
        for j in (i + 2, i + 3):
            if j < len(ws) and ws[j] == t and j + 1 < len(ws) and ws[j + 1] != ws[i + 1]:
                n += 1
                break
        if len(t) >= 2 and i + 1 < len(ws) and ws[i + 1] != t and ws[i + 1].startswith(t):
            n += 1
    return n


def parse_counts(s):
    m = re.search(r"\{.*?\}", s or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
        return int(d.get("corrected", 0)), int(d.get("uncorrected", 0))
    except (ValueError, TypeError):
        return None


def load_spacy():
    """en_core_web_sm 3.8.0 (the version the run used) from the artefacts dataset (a folder, or the zip if the dataset kept it
    packed), else the installed copy."""
    import tempfile
    import zipfile

    import spacy
    roots = [Path("/kaggle/input")] if Path("/kaggle/input").exists() else LOCAL_DIRS
    found = []
    for root in roots:
        for d, _, files in os.walk(root, followlinks=True):
            if "config.cfg" in files and "meta.json" in files and "en_core_web_sm" in d:
                found.append(Path(d))
            found += [Path(d) / f for f in files if f.startswith("spacy_en_core_web_sm") and f.endswith(".zip")]
    for p in found:
        if p.suffix == ".zip":
            out = Path(tempfile.mkdtemp())
            zipfile.ZipFile(p).extractall(out)
            p = next(q.parent for q in out.rglob("config.cfg"))
        try:
            return spacy.load(p, disable=["ner", "lemmatizer"]), f"{p} (spaCy {spacy.__version__})"
        except Exception as exc:
            print(f"bundled model not loadable with spaCy {spacy.__version__} ({type(exc).__name__}); trying the installed one")
    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"]), f"installed en_core_web_sm (spaCy {spacy.__version__})"


nlp, nlp_src = load_spacy()
QG = load_jsonl(find("qwen_gen.jsonl"))
J = pd.read_csv(find("qwen_judge.csv")).set_index(["split", "filename"])
LT = pd.read_csv(find("rubric_feats.csv")).set_index(["split", "filename"])        # the run's rubric features (LanguageTool here)
rows = []
for k_ in keys:
    chunks = ctc_chunks(Cc[k_]["words"], W[k_]["segments"])
    cw = [w for c in chunks for w in c]
    src, g = words(Cc[k_]["raw_text"]), QG.get(k_, {})
    cnt = parse_counts(g.get("selfcorr_json", ""))
    rows.append({**structure(W[k_]["text"], nlp), "self_repair_per100": 100 * self_repairs(cw) / max(len(cw), 1),
                 "llm_err_per100": 100 * lev(src, words(g.get("correction", ""))) / max(len(src), 1),
                 "corrected_error_ratio": np.nan if cnt is None or sum(cnt) == 0 else cnt[0] / sum(cnt)})
RB = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(keys, names=["split", "filename"]))
train_rows, test_rows = [("train", f) for f in T0.filename], [("test", f) for f in TE0.filename]
RB["corrected_error_ratio"] = RB.corrected_error_ratio.fillna(RB.loc[train_rows, "corrected_error_ratio"].median())
RB["llm_rubric_expected"] = J.loc[keys, [f"ctc_p{i}" for i in range(1, 6)]].values @ np.arange(1, 6)
RB["lt_grammar_per100"] = LT.loc[keys, "lt_grammar_per100"].values
for c in ("cola_frac_bad", "gpt2_nll_gap"):
    RB[c] = F[c].values
for c in ("repair_wer_nofiller", "no_punct", "duration", "n_words"):
    RB[c] = HC[c].values
RB["speech_time"] = [sum(max(0.0, float(w[2]) - float(w[1])) for w in W[k_]["words"]) for k_ in keys]
RB["batch"], RB["label"] = "", np.nan
RB.loc[train_rows, "batch"], RB.loc[test_rows, "batch"], RB.loc[train_rows, "label"] = batch_tr, batch_te, y0

# agreement with the run's rubric features (the spaCy columns can differ slightly if this session's spaCy version differs)
I_run = pd.read_csv(find("insight.csv")).set_index(["split", "filename"])
SPACY_COLS = ["finite_clause_ratio", "mean_len_clause", "dep_clause_ratio", "clauses_per_sentence", "mean_dep_distance",
              "fragment_ratio"]
ref = {c: (LT if c in LT.columns else I_run) for c in SPACY_COLS + ["mattr50", "trigram_repeat_rate", "self_repair_per100",
                                                                     "llm_err_per100", "corrected_error_ratio", "llm_rubric_expected"]}
agree = pd.DataFrame({c: {"max |here - run|": max_abs_diff(RB.loc[keys, c], r.loc[keys, c]),
                          "corr": np.corrcoef(RB.loc[keys, c].fillna(0), r.loc[keys, c].fillna(0))[0, 1]}
                      for c, r in ref.items()}).T
print(f"spaCy model: {nlp_src}")
print(agree.round(6).to_string())
assert (agree.drop(index=SPACY_COLS)["max |here - run|"] < 1e-9).all(), "non-spaCy rubric features differ from the run"
assert (agree.loc[SPACY_COLS, "corr"] > 0.99).all(), "spaCy features disagree with the run's"

EXPECTED = {"finite_clause_ratio": +1, "mean_len_clause": +1, "dep_clause_ratio": +1, "clauses_per_sentence": +1,
            "mean_dep_distance": +1, "mattr50": +1, "trigram_repeat_rate": -1, "fragment_ratio": -1, "lt_grammar_per100": -1,
            "self_repair_per100": 0}
EXISTING = {"cola_frac_bad": -1, "gpt2_nll_gap": -1, "repair_wer_nofiller": -1, "no_punct": -1}
QWEN = {"llm_err_per100": -1, "corrected_error_ratio": +1, "llm_rubric_expected": +1}
EXCLUDED = {"self_repair_per100": "CTC restart rate: measures disfluent restarts, not self-correction"}


def sp(x, y_):
    ok = ~(np.isnan(x) | np.isnan(y_))
    return float(spearmanr(x[ok], y_[ok]).statistic)


def partial_spearman(x, y_, z_):
    ok = ~(np.isnan(x) | np.isnan(y_) | np.isnan(z_))
    rx, ry, rz = rankdata(x[ok]), rankdata(y_[ok]), rankdata(z_[ok])
    Z = np.c_[np.ones(ok.sum()), rz]
    res_ = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return float(np.corrcoef(res_(rx), res_(ry))[0, 1])


def selection_table(T):
    y_, rw, dur, short_ = T.label.values, T.repair_wer_nofiller.values, T.duration.values, (T.batch == "short").values
    out = []
    for f, exp in {**EXPECTED, **EXISTING, **QWEN}.items():
        x = T[f].values.astype(float)
        rho = sp(x, y_)
        direction = exp if exp else np.sign(rho)
        means = [np.nanmean(x[y_ == lv]) for lv in LEVELS]
        inversions = int(sum(np.sign(b - a) == -direction for a, b in zip(means, means[1:])))
        d_short, d_long = sp(x[short_], dur[short_]), sp(x[~short_], dur[~short_])
        part = np.nan if f == "repair_wer_nofiller" else partial_spearman(x, y_.astype(float), rw)
        c1 = True if exp == 0 else bool(np.sign(rho) == exp)
        c2, c3 = inversions <= MAX_INVERSIONS, max(abs(d_short), abs(d_long)) < DUR_MAX
        c4 = True if f == "repair_wer_nofiller" else abs(part) >= PARTIAL_MIN
        out.append({"feature": f, "source": "new" if f in EXPECTED else ("qwen" if f in QWEN else "existing"),
                    "expected": {1: "+", -1: "-", 0: "data"}[exp], "spearman": rho, "1_sign": c1, "inversions": inversions,
                    "2_monotone": c2, "3_length": c3, "partial_vs_repair": part, "4_partial": c4,
                    "passes": c1 and c2 and c3 and c4 and f not in EXCLUDED})
    return pd.DataFrame(out)


RT = RB.loc[train_rows].reset_index()
sel = selection_table(RT)
sel_run = pd.read_csv(find("selection.csv")).set_index("feature")
assert (sel.set_index("feature").passes == sel_run.passes.reindex(sel.feature)).all(), "rubric-layer selection differs from the run"
print("\nRubric-layer selection (computed here; the same pass/fail as the run):")
print(sel.round(3).to_string(index=False))

show = ["mattr50", "trigram_repeat_rate", "llm_err_per100", "cola_frac_bad", "repair_wer_nofiller", "llm_rubric_expected",
        "clauses_per_sentence", "dep_clause_ratio"]
fig, axes = plt.subplots(2, 4, figsize=(16, 6))
for ax, f in zip(axes.ravel(), show):
    ax.boxplot([RT.loc[RT.label == lv, f].dropna() for lv in LEVELS], showfliers=False)
    ax.set_xticks(range(1, len(LEVELS) + 1), [f"{lv:g}" for lv in LEVELS]); ax.set_title(f, fontsize=10)
fig.supxlabel("grammar score"); fig.tight_layout(); plt.show()

# rubric conformance tests: one-sided Mann-Whitney between two label groups; r = rank-biserial effect, > 0 supports the rubric
TESTS = [
    ("L2 incomplete sentences", "fragment_ratio", lambda v: v <= 2.5, lambda v: v >= 3.0),
    ("L2 incomplete sentences", "no_punct", lambda v: v <= 2.5, lambda v: v >= 3.0),
    ("L1-2 memorized / simple", "trigram_repeat_rate", lambda v: v <= 2.5, lambda v: v >= 3.0),
    ("L1-2 memorized / simple", "mattr50", lambda v: v >= 3.0, lambda v: v <= 2.5),
    ("L3 vs L4 error density", "lt_grammar_per100", lambda v: (v >= 3.0) & (v <= 3.5), lambda v: (v >= 4.0) & (v <= 4.5)),
    ("L3 vs L4 error density", "llm_err_per100", lambda v: (v >= 3.0) & (v <= 3.5), lambda v: (v >= 4.0) & (v <= 4.5)),
    ("L5 complex structures", "dep_clause_ratio", lambda v: v == 5.0, lambda v: v == 4.0),
    ("L5 complex structures", "clauses_per_sentence", lambda v: v == 5.0, lambda v: v == 4.0),
    ("L4-5 self-correction", "corrected_error_ratio", lambda v: v >= 4.0, lambda v: v <= 3.0),
]


def mw(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    u, p = mannwhitneyu(a, b, alternative="greater")
    return float(p), float(2 * u / (len(a) * len(b)) - 1)


tests = []
for b_ in ("short", "long"):
    d = RT[RT.batch == b_]
    yb = d.label.values
    for desc, feat, ga, gb in TESTS:
        p, r = mw(d.loc[ga(yb), feat], d.loc[gb(yb), feat])
        tests.append({"batch": b_, "descriptor": desc, "test": feat, "p": p, "effect_r": r, "pass": bool(p < ALPHA)})
tests = pd.DataFrame(tests)
same = tests.set_index(["batch", "test"]).join(pd.read_csv(find("rubric_tests.csv")).set_index(["batch", "test"])[["p"]], rsuffix="_run")
same = same[~same.index.get_level_values("test").isin(SPACY_COLS)]
assert np.allclose(same.p, same.p_run, rtol=1e-6, atol=1e-12), "rubric tests differ from the run"
print("\nRubric conformance tests per batch (computed here; pass = p < 0.05; the model's own test is in section 7):")
print(tests.pivot_table(index=["descriptor", "test"], columns="batch", values=["pass", "effect_r"], aggfunc="first").round(3).to_string())
RUNTIME["4 rubric layer"] = time.time() - _t

# %% [markdown]
# ## 5. Learned members
# The final pick combines **two pipelines**, each run on **four sets of speaker-grouped fold seeds** (42/1337/2024, 7/99/555,
# 11/22/33, 44/66/88). Every member enters the stack as its out-of-fold prediction on the run's folds (the transcript fine-tunes on
# the grouped seed-42 folds), cached per run in `cands_<run>.npz` with its test predictions and its topic-held-out predictions. The
# members are built by the experiment loop (`src/`) and the GPU jobs (`kaggle/<job>/`, listed in the appendix).
# - **Base pipeline:** speech-encoder ridge heads (WavLM-base, WavLM-large, Whisper-medium encoder), sentence embeddings,
#   DeBERTa-v3-base on the Whisper text and on the (CTC raw, Whisper clean) pair, Qwen2.5-7B rubric-prompt heads (CTC, Whisper), a
#   Voxtral-Mini-3B audio-LLM head, the Laya checklist features, and 9 model families on all of them.
# - **Combination pipeline:** the base, plus a Rank-N-Contrast member, an RBF-SVR head on WavLM-large layer 20,
#   DeBERTa-v3-large (6 seeds) replacing the base pair model, DeBERTa-v3-base on the (verbatim Whisper-turbo, Whisper clean) pair,
#   Qwen3.5-9B answer-state heads on three transcripts, and Voxtral averaged over three prompt views (the grammar question after
#   the audio, or before it in two wordings) replacing the single Voxtral head.

# %%
_t = time.time()
PIPES = {"base": BASE_RUNS, "combo": COMBO_RUNS}
CANDS = {r: np.load(find(f"cands_{r}.npz"), allow_pickle=True) for rs in PIPES.values() for r in rs}
y, batch, batch_test, wts = C["y"], C["batch"], C["batch_test"], C["w"]
for r, c in CANDS.items():
    assert list(c["filename"]) == list(T0.filename) and list(c["test_filename"]) == list(TE0.filename), f"{r}: row order"
    assert np.array_equal(c["y"], y) and np.allclose(c["w"], wts) and (c["batch"] == batch).all(), f"{r}: labels or batches"
rows_m = []
for pipe, rs in PIPES.items():
    c = CANDS[rs[0]]
    CO_ = c["oof"].mean(0)
    for j, n in enumerate(c["names"]):
        rows_m.append({"pipeline": pipe, "member": str(n), "test-mix OOF RMSE": wrmse(y, CO_[:, j], wts),
                       "topic-held-out RMSE": rmse(y, c["topic"][:, j])})
members = pd.DataFrame(rows_m)
print("members of the seed-42 run of each pipeline (out-of-fold RMSE at the test batch mix; families = the 9 model families):")
print(members.pivot_table(index="member", columns="pipeline", values="test-mix OOF RMSE").round(4)
      .sort_values("combo").to_string())
RUNTIME["5 members"] = time.time() - _t

# %% [markdown]
# ## 6. Validation and stacking
# For each run, a non-negative least-squares blend is fitted **separately for short and long clips** with the test-mix weights, on
# the members' topic-held-out predictions (so the weights trust prompt-robust members); it is refit inside every fold for the
# out-of-fold blend. The combination pipeline adds three stages:
# 1. the short-clip blend also sees the long clips at 0.25 × their weight (fewer short clips, steadier weights);
# 2. an **examiner correction**: a ridge on 21 zero-shot Qwen3.5-9B rubric judgements (expected grade from the probabilities of the
#    answers 1–5, for several questions and transcripts) and the prediction predicts the residual; refit inside every fold;
# 3. **same-speaker pooling**: each prediction moves 30% toward its speaker group's mean (41 test clips sit in 17 same-voice
#    groups; grades of a speaker's clips differ by 0.24 on average).
#
# Every replayed run is checked against the run's own files. The final pick then averages the four seed sets of each pipeline
# (bagging over fold splits) and combines the pipelines per batch: **short clips get the mean of both pipelines** (on fresh folds
# the combination's gains were all on long clips; averaging the two pipelines was the one short-clip gain that held), **long
# clips get the combination**.

# %%
_t = time.time()
SHRINK_LAMBDA_ = SHRINK_LAMBDA


def blend_weights(Cm, ym, wm, bm, shrink=False):
    """Per-batch NNLS on sqrt(weight)-scaled rows -> (short weights, long weights); shrink: the short fit also sees the long rows."""
    sh, lo = np.where(bm == "short")[0], np.where(bm == "long")[0]
    s_lo = np.sqrt(wm[lo])
    w_long = nnls(Cm[lo] * s_lo[:, None], ym[lo] * s_lo)[0]
    R = np.r_[sh, lo] if shrink else sh
    s_r = np.sqrt(np.r_[wm[sh], SHRINK_LAMBDA_ * wm[lo]] if shrink else wm[sh])
    return nnls(Cm[R] * s_r[:, None], ym[R] * s_r)[0], w_long


def apply_blend(Cm, bm, wb):
    return np.where(bm == "short", Cm @ wb[0], Cm @ wb[1])


def judge_features(J1, J2, keys_):
    """The examiner correction's 21 inputs: round 1 (expected grade + log-odds of grade >= 4; verbatim / CTC / Whisper transcript x
    rubric / accuracy question) and round 2 (expected grade; complexity, completeness, CEFR, two transcripts side by side)."""
    cols = {}
    for v in ("verbatim", "ctc", "whisper"):
        for p in ("holistic", "accuracy"):
            P = J1.loc[keys_, [f"{v}_{p}_p{i}" for i in range(1, 6)]].values
            P = P / P.sum(1, keepdims=True)
            cols[f"judge35_{v}_{p}_ev"] = P @ np.arange(1, 6)
            cols[f"judge35_{v}_{p}_lo"] = np.log(P[:, 3:].sum(1) + 1e-9) - np.log(P[:, :3].sum(1) + 1e-9)
    for c in sorted({c.rsplit("_p", 1)[0] for c in J2.columns if "_p" in c and not c.endswith("_top")}):
        nd = 6 if c.endswith("cefr") else 5
        P = J2.loc[keys_, [f"{c}_p{i}" for i in range(1, nd + 1)]].values
        cols[f"j2_{c}_ev"] = (P / P.sum(1, keepdims=True)) @ np.arange(1, nd + 1)
    return pd.DataFrame(cols).values


def ridge_correction(Z, e, sw):
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    m = RidgeCV(alphas=np.logspace(-1, 4, 30)).fit((Z - mu) / sd, e, sample_weight=sw)
    return lambda Zn: m.predict((Zn - mu) / sd)


def pool(p, g, s):
    return (1 - s) * p + s * pd.Series(p).groupby(np.asarray(g)).transform("mean").values


def replay(c, *, shrink=False, correction=None, pool_s=0.0, in_sample=False):
    """One run's stacking from its candidate matrices -> (out-of-fold prediction, test prediction, in-sample prediction).
    in_sample: every post-member stage fitted on all 732 clips and applied to them (the TRAINING RMSE definition)."""
    CO_s, CT_s, TO = c["oof"], c["test"], c["topic"]
    seeds_ = [int(s) for s in c["seeds"]]
    wb_all = blend_weights(TO, y, wts, batch, shrink)
    test_p = np.clip(apply_blend(CT_s.mean(0), batch_test, wb_all), *CLIP)
    train_p = np.clip(apply_blend(CO_s.mean(0), batch, wb_all), *CLIP)
    seed_oofs = []
    for i, s in enumerate(seeds_):
        o = np.zeros(len(y))
        for a, b in FOLDS[s]:
            o[b] = apply_blend(CO_s[i][b], batch[b], blend_weights(TO[a], y[a], wts[a], batch[a], shrink))
        seed_oofs.append(np.clip(o, *CLIP))
    oof_p = np.clip(np.mean(seed_oofs, 0), *CLIP)
    if correction is not None:
        X, Xt = correction
        new = []
        for i, s in enumerate(seeds_):
            o = seed_oofs[i]; oc = o.copy()
            for a, b in FOLDS[s]:
                f = ridge_correction(np.c_[X[a], o[a]], y[a] - o[a], wts[a])
                oc[b] = o[b] + f(np.c_[X[b], o[b]])
            new.append(np.clip(oc, *CLIP))
        f = ridge_correction(np.c_[X, oof_p], y - oof_p, wts)
        test_p = np.clip(test_p + f(np.c_[Xt, test_p]), *CLIP)
        train_p = np.clip(train_p + f(np.c_[X, train_p]), *CLIP)
        seed_oofs = new
        oof_p = np.clip(np.mean(seed_oofs, 0), *CLIP)
    if pool_s > 0:
        seed_oofs = [np.clip(pool(o, groups_run, pool_s), *CLIP) for o in seed_oofs]
        oof_p = np.clip(np.mean(seed_oofs, 0), *CLIP)
        test_p = np.clip(pool(test_p, groups_test, pool_s), *CLIP)
        train_p = np.clip(pool(train_p, groups_run, pool_s), *CLIP)
    return oof_p, test_p, train_p


J1 = pd.read_csv(find("qwen35_judge.csv")).set_index(["split", "filename"])
J2 = pd.read_csv(find("qwen35_judge2.csv")).set_index(["split", "filename"])
XJ = (judge_features(J1, J2, train_rows), judge_features(J1, J2, test_rows))
groups_test = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                      distance_threshold=1.0 - twin_t).fit_predict(Xv_te)      # same-voice groups inside test
n_pooled = int((pd.Series(groups_test).map(pd.Series(groups_test).value_counts()) >= 2).sum())
print(f"test clips in same-voice groups: {n_pooled}")
STAGES = {"base": {}, "combo": dict(shrink=True, correction=XJ, pool_s=POOL_S)}
OUT = {}
for pipe, rs in PIPES.items():
    for r in rs:
        oof_r, test_r, train_r = replay(CANDS[r], **STAGES[pipe])
        O_ = pd.read_csv(find(f"oof_{r}.csv")).set_index("filename").loc[T0.filename, "oof"].values
        S_ = pd.read_csv(find(f"sub_{r}.csv")).set_index("filename").loc[TE0.filename, "label"].values
        d_o, d_t = np.abs(oof_r - O_).max(), np.abs(test_r - S_).max()
        assert d_o < 1e-9 and d_t < 1e-9, f"{r}: the replay differs from the run (OOF {d_o:.1e}, test {d_t:.1e})"
        OUT[r] = (oof_r, test_r, train_r)
        print(f"  {pipe:5s} run {r} (fold seeds {'/'.join(str(int(s)) for s in CANDS[r]['seeds'])}): replay = run's files "
              f"(max |diff| {max(d_o, d_t):.1e}); cv_testmix {wrmse(y, oof_r, wts):.4f}")
bag = {p: tuple(np.mean([OUT[r][k_] for r in rs], 0) for k_ in range(3)) for p, rs in PIPES.items()}
short, short_te = batch == "short", batch_test == "short"
combine = lambda b_, c_, m: np.clip(np.where(m, (b_ + c_) / 2, c_), *CLIP)
blend_oof = combine(bag["base"][0], bag["combo"][0], short)
blend_test = combine(bag["base"][1], bag["combo"][1], short_te)
train_fit = combine(bag["base"][2], bag["combo"][2], short)
WB = {p: blend_weights(CANDS[rs[0]]["topic"], y, wts, batch, STAGES[p].get("shrink", False)) for p, rs in PIPES.items()}
for p, rs in PIPES.items():
    nm = [str(n) for n in CANDS[rs[0]]["names"]]
    wt = pd.DataFrame({"member": nm, "short": WB[p][0], "long": WB[p][1]})
    print(f"\n{p} pipeline, blend weights (seed-42 run, fitted on all clips):")
    print(wt[(wt.short > 1e-3) | (wt.long > 1e-3)].round(3).sort_values("short", ascending=False).to_string(index=False))
RUNTIME["6 stacking"] = time.time() - _t

# %% [markdown]
# ## 7. Results

# %%
_t = time.time()
cv = {"cv_grouped": rmse(y, blend_oof), "cv_testmix": wrmse(y, blend_oof, wts), "cv_short": rmse(y[short], blend_oof[short]),
      "cv_long": rmse(y[~short], blend_oof[~short])}
train_rmse = rmse(y, train_fit)
assert np.isfinite(train_rmse), "TRAINING RMSE is not finite"
box = [f"TRAINING RMSE (compulsory): {train_rmse:.4f}",
       "definition: every stage after the members (blend weights, examiner correction, pooling, the combination) fitted on all",
       "732 training clips and scored on those same clips; the members enter as their out-of-fold predictions"]
width = max(len(line) for line in box) + 4
print("╔" + "═" * width + "╗")
for line in box:
    print("║  " + line.ljust(width - 2) + "║")
print("╚" + "═" * width + "╝")
print("honest numbers (out-of-fold, speaker-grouped):", "  ".join(f"{k_} {v:.4f}" for k_, v in cv.items()))
for p in PIPES:
    print(f"  {p} pipeline alone (4 seed sets averaged): cv_testmix {wrmse(y, bag[p][0], wts):.4f}  "
          f"short {rmse(y[short], bag[p][0][short]):.4f}  long {rmse(y[~short], bag[p][0][~short]):.4f}")

# %% [markdown]
# The TRAINING RMSE is close to the CV because the members, including the GPU fine-tunes, enter as out-of-fold predictions even
# here (refitting 20+ members in-sample would mostly measure memorisation), and the stacking stages are small and regularised. It
# measures the fit of the stacking layers, not generalisation.
#
# **Prompt-shift check of the correction stage.** The topic-held-out validation (fit outside one prompt cluster, score it) was
# first computed *before* the examiner correction, so the correction was never tested against unseen prompts. Here it is computed
# after it as well, for the seed-42 run of the combination: the correction is close to neutral under prompt shift, which is why
# the batch-specific version of it (its own slope on the 45.06 s batch, the best speaker-grouped CV of the project) was dropped:
# it lost on unseen prompts (0.5178 vs 0.5160) and on the public board (0.3311 vs 0.3280).

# %%
c42 = CANDS[COMBO_RUNS[0]]
tp = I_run.loc[train_rows, "topic"].values
topic_folds = [(np.where((tp != t_) & ~np.isin(groups_run, np.unique(groups_run[tp == t_])))[0], np.where(tp == t_)[0])
               for t_ in sorted(set(tp))]
nt = np.zeros(len(y))
for _, b_t in topic_folds:
    rows_ = np.setdiff1d(np.arange(len(y)), b_t)
    nt[b_t] = apply_blend(c42["topic"][b_t], batch[b_t], blend_weights(c42["topic"][rows_], y[rows_], wts[rows_], batch[rows_], True))
nt = np.clip(nt, *CLIP)
ntc = nt.copy()
for a_t, b_t in topic_folds:
    f = ridge_correction(np.c_[XJ[0][a_t], nt[a_t]], y[a_t] - nt[a_t], wts[a_t])
    ntc[b_t] = nt[b_t] + f(np.c_[XJ[0][b_t], nt[b_t]])
ntc = np.clip(ntc, *CLIP)
print(f"topic-held-out (unseen prompt clusters), combination seed-42 run: before the correction {wrmse(y, nt, wts):.4f} "
      f"(short {rmse(y[short], nt[short]):.4f}), after it {wrmse(y, ntc, wts):.4f} (short {rmse(y[short], ntc[short]):.4f})")


# %%
def qwk(a, b):
    """Quadratic weighted kappa on 0.5-step grades."""
    a, b = (np.round(np.asarray(v) * 2).astype(int) for v in (a, b))
    lo_, hi_ = min(a.min(), b.min()), max(a.max(), b.max()); n = hi_ - lo_ + 1
    O = np.zeros((n, n))
    for i_, j_ in zip(a - lo_, b - lo_):
        O[i_, j_] += 1
    Wq = np.array([[(i_ - j_) ** 2 for j_ in range(n)] for i_ in range(n)]) / (n - 1) ** 2
    Ex = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return 1 - (Wq * O).sum() / (Wq * Ex).sum()


def board_metric(t_, p_):
    """The public leaderboard's metric, decoded from shift/scale probes (Techy-A's write-up; our own two shift probes agree):
    0.6 x RMSE + 0.4 x (1 - Pearson)."""
    return 0.6 * rmse(t_, p_) + 0.4 * (1 - pearsonr(t_, p_)[0])


rng = np.random.default_rng(SEED)
boot = [wrmse(y[ix], blend_oof[ix], wts[ix]) for ix in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
stats = {"MAE": float(np.mean(np.abs(y - blend_oof))), "Pearson": pearsonr(y, blend_oof)[0], "Spearman": spearmanr(y, blend_oof)[0],
         "QWK (rounded to 0.5)": qwk(y, np.round(blend_oof * 2) / 2), "within 0.5": float(np.mean(np.abs(y - blend_oof) <= 0.5)),
         "within 1.0": float(np.mean(np.abs(y - blend_oof) <= 1.0)),
         "cv_testmix 95% CI": f"{np.percentile(boot, 2.5):.4f}–{np.percentile(boot, 97.5):.4f}",
         "board metric at the test batch mix (OOF)": None}
p_pick = wts / wts.sum()
rng_pub = np.random.default_rng(SEED)
pub = np.array([board_metric(y[ix], blend_oof[ix]) for ix in (rng_pub.choice(len(y), N_PUBLIC, replace=True, p=p_pick)
                                                             for _ in range(N_RESAMPLE))])
stats["board metric at the test batch mix (OOF)"] = float(np.mean(pub))
for k_, v in stats.items():
    print(f"  {k_:<42} {v if isinstance(v, str) else round(float(v), 4)}")
print("RMSE per score level:", "  ".join(f"{lv:g}: {rmse(y[y == lv], blend_oof[y == lv]):.3f}" for lv in LEVELS))
lo, hi = np.percentile(pub, [2.5, 97.5])
print(f"\nboard metric on {N_PUBLIC}-clip resamples of the out-of-fold predictions at the test batch mix: 95% interval "
      f"{lo:.3f}–{hi:.3f}; the public score {PUBLIC_SCORE} is {'below' if PUBLIC_SCORE < lo else 'inside'} it")

for b_ in ("short", "long"):
    m = batch == b_
    means = [blend_oof[m & (y == lv)].mean() for lv in LEVELS]
    bad = [f"{a:g}->{b:g}" for a, b, ma, mb in zip(LEVELS, LEVELS[1:], means, means[1:]) if not mb > ma]
    print(f"model monotone in level [{b_}]: {'PASS' if not bad else 'FAIL at ' + ', '.join(bad)}   means "
          + " ".join(f"{v:.2f}" for v in means))

# %% [markdown]
# **Rubric attribution.** A surrogate fitted on the rubric-layer features alone (the 14 features of section 4, grouped by the
# rubric criterion they measure) is trained to reproduce the final out-of-fold predictions. Fidelity is its out-of-fold R² on the
# seed-42 speaker-grouped folds; a criterion's importance is the drop in that held-out R² when all its features are permuted
# together (20 permutations per fold). Two surrogates (ridge, shallow LightGBM) check that the ranking doesn't depend on one.

# %%
CRITERIA = {"accuracy": ["repair_wer_nofiller", "lt_grammar_per100", "llm_err_per100"],
            "structure": ["cola_frac_bad", "gpt2_nll_gap", "finite_clause_ratio", "mean_len_clause"],
            "range": ["dep_clause_ratio", "clauses_per_sentence", "mean_dep_distance", "mattr50", "trigram_repeat_rate"],
            "completeness": ["no_punct", "fragment_ratio"]}
rub_cols = [c for cs in CRITERIA.values() for c in cs]
Xr = RT[rub_cols].astype(float)
Xr = Xr.fillna(Xr.median()).values
surrogates = {"ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
              "LightGBM (4 leaves)": lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=4, min_child_samples=20,
                                                       verbose=-1, random_state=SEED)}
rng_perm = np.random.default_rng(SEED)
attribution, fidelity = {}, {}
for sname, model in surrogates.items():
    pred, drops = np.zeros(len(y)), {k_: [] for k_ in CRITERIA}
    for a, b in FOLDS[SEED]:
        m = clone(model).fit(Xr[a], blend_oof[a])
        pred[b] = m.predict(Xr[b])
        base_r2 = r2_score(blend_oof[b], pred[b])
        for k_, cs in CRITERIA.items():
            jj = [rub_cols.index(c) for c in cs]
            for _ in range(20):
                Xp = Xr[b].copy()
                Xp[:, jj] = Xp[rng_perm.permutation(len(b))][:, jj]
                drops[k_].append(base_r2 - r2_score(blend_oof[b], m.predict(Xp)))
    fidelity[sname] = r2_score(blend_oof, pred)
    attribution[sname] = {k_: float(np.mean(v)) for k_, v in drops.items()}
attribution = pd.DataFrame(attribution)
print("surrogate fidelity (out-of-fold R² against the final OOF prediction):",
      "  ".join(f"{k_} {v:.3f}" for k_, v in fidelity.items()))
print("importance = drop in held-out R² when the criterion's features are permuted together:")
print(attribution.round(3).to_string())

GROUPS_M = {"transcript fine-tunes (DeBERTa)": lambda n: n.startswith("deberta"),
            "LLM heads (Qwen2.5, Qwen3.5)": lambda n: n.startswith("qwen"),
            "speech encoders + Voxtral": lambda n: n.startswith(("wavlm", "whisper", "voxtral")),
            "Rank-N-Contrast": lambda n: n.startswith("rnc"),
            "families on rubric + member features": lambda n: n in {"ridge", "krr", "svr", "knn", "rf", "et", "mlp", "lgbm", "xgb"}}
attr_w = pd.DataFrame({f"{p} {b_}": {g_: sum(WB[p][k_][j] for j, n in enumerate(str(x) for x in CANDS[PIPES[p][0]]["names"]) if f_(n))
                                     for g_, f_ in GROUPS_M.items()}
                       for p in PIPES for k_, b_ in enumerate(("short", "long"))})
fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
for b_, c in (("short", "#e67e22"), ("long", "#2e86c1")):
    m = batch == b_
    axes[0].scatter(y[m] + rng.normal(0, 0.05, m.sum()), blend_oof[m], s=8, alpha=0.5, label=b_, color=c)
axes[0].plot([1.5, 5], [1.5, 5], "k--", lw=1); axes[0].set(title="Predicted vs true (OOF)", xlabel="true", ylabel="predicted")
axes[0].legend(frameon=False)
axes[1].boxplot([(y - blend_oof)[y == lv] for lv in LEVELS], showfliers=False)
axes[1].set_xticks(range(1, len(LEVELS) + 1), [f"{lv:g}" for lv in LEVELS]); axes[1].axhline(0, color="k", lw=0.8)
axes[1].set(title="Residual (true − predicted) by level")
axes[2].bar(["short", "long"], [cv["cv_short"], cv["cv_long"]], color=["#e67e22", "#2e86c1"]); axes[2].set(title="RMSE by batch")
fig.tight_layout(); plt.show()
fig, axes = plt.subplots(1, 2, figsize=(16, 3.8))
attr_w.plot.barh(ax=axes[0]); axes[0].set(title="Blend weight by member group (seed-42 runs)")
attribution.plot.barh(ax=axes[1], color=["#7f8c8d", "#27ae60"])
axes[1].set(title=f"Rubric attribution: held-out R² lost when a criterion is permuted (surrogate fidelity R² "
                  f"{min(fidelity.values()):.2f}–{max(fidelity.values()):.2f})", xlabel="drop in R²")
fig.tight_layout(); plt.show()

# %% [markdown]
# **Error analysis**, recomputed from the final out-of-fold predictions (residual = label − prediction; positive = the speaker was
# scored too low), with the numbers behind the report's findings.

# %%
res = y - blend_oof
RT["pred"], RT["residual"] = blend_oof, res
tab = lambda d, by: d.assign(se=d.residual ** 2).groupby(by, observed=True).agg(
    n=("residual", "size"), rmse=("se", lambda s: float(np.sqrt(s.mean()))), mean_residual=("residual", "mean")).round(3)
print("by batch x level:\n", tab(RT[RT.label >= 2.0], ["batch", "label"]).unstack("batch").to_string())

ze = np.load(find("emb_mpnet.npz"))
ie = {k_: i for i, k_ in enumerate(zip(ze["split"], ze["filename"]))}
Em = ze["E"][[ie[k_] for k_ in keys]].astype(np.float64)
Em /= np.linalg.norm(Em, axis=1, keepdims=True)
topic = KMeans(K_TOPICS, n_init=10, random_state=0).fit_predict(Em)
tfidf = TfidfVectorizer(stop_words="english", min_df=3, max_df=0.5)
Xtf = tfidf.fit_transform([W[k_]["text"] for k_ in keys])
vocab = np.array(tfidf.get_feature_names_out())
is_tr = (F.split == "train").values
tr_topic = pd.Series(topic[is_tr], index=F.filename[is_tr].values).loc[T0.filename].values
topics = pd.DataFrame({"words": [" ".join(vocab[np.asarray(Xtf[topic == t].mean(0)).ravel().argsort()[::-1][:6]]) for t in range(K_TOPICS)],
                       "train_share": [np.mean(topic[is_tr] == t) for t in range(K_TOPICS)],
                       "test_share": [np.mean(topic[~is_tr] == t) for t in range(K_TOPICS)],
                       "train_rmse": [rmse(y[tr_topic == t], blend_oof[tr_topic == t]) if (tr_topic == t).any() else np.nan
                                      for t in range(K_TOPICS)]})
print("\nprompt clusters (k-means on sentence embeddings, no labels):\n", topics.round(3).to_string())

sr = sel.set_index("feature")
print("\nFinding 1: restart rate vs grade, Spearman "
      f"{sr.loc['self_repair_per100', 'spearman']:+.2f}; LLM corrected-error ratio {sr.loc['corrected_error_ratio', 'spearman']:+.2f} "
      f"({sr.loc['corrected_error_ratio', 'inversions']} inversions)")
judge = RB.groupby([RB.index.get_level_values("split"), "batch"]).llm_rubric_expected.mean().round(2)
print("Finding 2: zero-shot rubric judge, mean expected grade by split x batch:", judge.to_dict())
st_ = RT.speech_time.values
print(f"Finding 3: short clips, residual slope per 10 s of speech {10 * np.polyfit(st_[short], res[short], 1)[0]:+.3f} "
      f"(long clips {10 * np.polyfit(st_[~short], res[~short], 1)[0]:+.3f})")
fail = RT.repair_wer_nofiller.values > CTC_FAIL_CUT
fail_te = RB.loc[test_rows, "repair_wer_nofiller"].values > CTC_FAIL_CUT
print(f"Finding 4: CTC failure (repair rate > {CTC_FAIL_CUT}) in {fail.mean():.1%} of train, {fail_te.mean():.1%} of test; "
      f"mean residual of failed clips graded >= 4: {res[fail & (y >= 4)].mean():+.2f} (n={int((fail & (y >= 4)).sum())}), "
      f"other clips graded >= 4: {res[~fail & (y >= 4)].mean():+.2f}")
JX = pd.DataFrame(XJ[0], columns=[f"j{i}" for i in range(XJ[0].shape[1])])
partial = []
for b_, m in (("long", ~short), ("short", short)):
    A_ = np.c_[np.ones(m.sum()), blend_oof[m]]
    e1 = y[m] - A_ @ np.linalg.lstsq(A_, y[m], rcond=None)[0]
    for j in range(XJ[0].shape[1]):
        e2 = XJ[0][m, j] - A_ @ np.linalg.lstsq(A_, XJ[0][m, j], rcond=None)[0]
        partial.append((b_, j, np.corrcoef(e1, e2)[0, 1]))
pt = pd.DataFrame(partial, columns=["batch", "j", "r"]).groupby("batch").r.agg(["min", "median", "max"]).round(3)
print("Finding 5: the examiner's numbers vs the grade GIVEN the final prediction (partial correlation, 21 numbers):\n", pt.to_string())


# %%
def report_card(row, pred):
    """Rubric-language summary of one test clip from its rubric features (percentiles against training clips)."""
    pct = lambda f: float((RT[f] < row[f]).mean() * 100)
    nth = lambda v: f"{v:.0f}" + ("th" if 10 <= v % 100 < 20 else {1: "st", 2: "nd", 3: "rd"}.get(int(round(v)) % 10, "th"))
    err, rng_, rep, bad = pct("llm_err_per100"), pct("mattr50"), pct("trigram_repeat_rate"), pct("cola_frac_bad")
    accuracy = ("high grammatical accuracy" if err < 20 else "occasional minor errors" if err < 45
                else "noticeable errors in grammatical structure" if err < 75 else "frequent basic mistakes")
    rangew = "a varied vocabulary" if rng_ > 66 else "a moderate range of words" if rng_ > 33 else "a limited, repetitive range of words"
    lines = [f"Predicted grade {pred:.1f}  ({row['batch']} clip, {row['duration']:.0f} s)",
             f"- accuracy: {accuracy} (LLM error density {row['llm_err_per100']:.0f} per 100 words, {nth(err)} percentile of training clips)",
             f"- range: {rangew} (moving TTR {row['mattr50']:.2f}); repeated phrases {'often' if rep > 75 else 'sometimes' if rep > 40 else 'rarely'}",
             f"- structure: {row['cola_frac_bad']:.0%} of sentences judged ungrammatical by a CoLA classifier ({nth(bad)} percentile)"
             + ("; no sentence is completed with terminal punctuation (rubric level 2: may leave sentences incomplete)" if row["no_punct"] else ""),
             f"- zero-shot rubric judge (Qwen2.5-7B, no labels): expected grade {row['llm_rubric_expected']:.1f}"]
    return "\n".join(lines)


order = np.argsort(blend_test)
for label, j in (("LOW", order[0]), ("MID", order[len(order) // 2]), ("HIGH", order[-1])):
    f = TE0.filename.values[j]
    print(f"\n=== Report card ({label}): {f}\n" + report_card(RB.loc[("test", f)], blend_test[j]))
    print("  Whisper:", W[("test", f)]["text"][:200])
RUNTIME["7 results"] = time.time() - _t

# %% [markdown]
# ## 8. Submission
# **Final pick:** `submission.csv` is hybrid11, the model as validated. It is built from test.csv (216 rows, in its order),
# clipped to [1.5, 5.0], and checked against the submitted file.
#
# **Second pick, a hedge for the one question CV can't answer.** On the training data the model is unbiased (out-of-fold mean
# error −0.001), but the public board suggests the test predictions run high: two shift probes of an earlier submission measured
# its mean error on the public clips at +0.049, and the public scores of 32 submitted files, combined through the averaging identity
# (`src/pubmix.py`), put hybrid11's at +0.041. Both rest on the same ~130 public clips, about one standard error from zero, so
# whether the whole test set is biased is unknown; a constant shift leaves Pearson unchanged and is worth about ±0.001 on the
# private board. `submission_pick2_offset.csv` (pubmix_c, public 0.3271) is hybrid11 − 0.041: Kaggle scores the better of the two
# selections, so it covers the case where the bias is real. It is the only step tuned on the public board, and it is not part of
# the model or its CV (applied to the training predictions it would cost 0.0018).

# %%
_t = time.time()
final_test = blend_test
pick2 = test[["filename"]].merge(pd.DataFrame({"filename": TE0.filename, "label": np.clip(blend_test + TEST_OFFSET, *CLIP)}),
                                 on="filename", how="left")
pick2.to_csv("submission_pick2_offset.csv", index=False, float_format=FLOAT_FORMAT)
ref2 = pd.read_csv(find("pick2_submission.csv")).set_index("filename").label.reindex(pick2.filename).values
assert np.abs(np.round(pick2.label.values, 10) - np.round(ref2, 10)).max() < 1e-9, "second pick differs from pubmix_c"
print(f"submission_pick2_offset.csv equals pubmix_c at {FLOAT_FORMAT} (max |diff| {np.abs(pick2.label.values - ref2).max():.1e})")
print(f"test predictions: mean {blend_test.mean():.3f}; second pick (offset {TEST_OFFSET:+.3f}) mean {pick2.label.mean():.3f}; "
      f"out-of-fold cv_testmix {wrmse(y, blend_oof, wts):.4f}, with the offset {wrmse(y, np.clip(blend_oof + TEST_OFFSET, *CLIP), wts):.4f}")
sub = test[["filename"]].merge(pd.DataFrame({"filename": TE0.filename, "label": final_test}), on="filename", how="left")
assert len(sub) == 216 and sub.filename.is_unique and sub.label.notna().all() and sub.label.between(1.0, 5.0).all()
chosen = pd.read_csv(find("chosen_submission.csv"))
diff = np.abs(sub.set_index("filename").label - chosen.set_index("filename").label.reindex(sub.filename)).max()
sub.to_csv("submission.csv", index=False, float_format=FLOAT_FORMAT)
canonical = chosen.set_index("filename").reindex(sub.filename).reset_index().to_csv(index=False, float_format=FLOAT_FORMAT,
                                                                                      lineterminator="\n")
same_bytes = Path("submission.csv").read_bytes().replace(b"\r\n", b"\n") == canonical.encode()
assert same_bytes, f"submission.csv differs from {FINAL_PICK} byte for byte (max |diff| {diff})"
print(f"submission.csv identical to {FINAL_PICK} byte for byte at {FLOAT_FORMAT}: {same_bytes}  (raw max |diff| {diff:.1e})")
RUNTIME["8 submission"] = time.time() - _t

# %% [markdown]
# ## 9. Discussion: what worked, what didn't, and findings
# **What worked**, by public effect (each change submitted in isolation where possible):
# 1. *Honest validation:* speaker-grouped folds and leak-free members (public 0.3707 → 0.3441), then removing the last leaky member,
#    a fine-tuned WavLM whose random-fold out-of-fold column scored 0.46 on clips whose speaker had a twin in training and 0.65 on
#    the rest yet carried 40% of the long-clip blend (0.3414 → 0.3344).
# 2. *Averaging two pipelines on short clips* (→ 0.3309; → 0.3297 with a short-clip blend that also sees the long clips at a small
#    weight): on fresh folds the combination's new members improved only long clips.
# 3. *The examiner correction:* a matched A/B (same bagging, one change) gave 0.3280 with it vs 0.3335 without.
# 4. *Bagging over fold-seed sets* (→ 0.3283 → 0.3279; saturates after two sets). A test-set offset from the public board gives
#    0.3271 and is kept as the second pick only (section 8).
# On CV the combination's members each added 0.001–0.003: Rank-N-Contrast, DeBERTa-v3-large, Voxtral views, Qwen3.5 heads.
#
# **What didn't, and why it is instructive.**
# - *A batch-specific slope for the examiner:* the best speaker-grouped CV of the project (0.4994 on fresh folds), but worse on
#   unseen prompts (0.5178 vs 0.5160) and on the public board (0.3311 vs 0.3280). With 86 clips in a batch it learns the batch's
#   prompts, not its recordings.
# - *A less noisy examiner* (six paraphrased rubric prompts averaged, with their spread; fold-safe few-shot anchors): the
#   paraphrases agree closely, so averaging removed little noise; no variant improved both checks.
# - *An itemised error profile* from the examiner (counts of agreement, verb-form, article, preposition and word-order errors,
#   fragments, self-corrections, range of structures): each count tracks the grade alone (Spearman −0.29 to −0.47; range +0.57),
#   but almost nothing is left once the prediction is known.
# - *Speech-recognition reliability* as the cause of the short-clip gap: ruled out (short clips are recognised slightly better).
# - Crop augmentation, attention pooling over Voxtral frames, a three-way blend for the 45.06 s batch, delexicalised text models,
#   LLM judge scores as blend members, an amount-spoken correction, rescaling the predictions (the least-squares scale already
#   minimises the board metric), ordinal + EM prior adaptation, transductive pseudo-labelling, larger encoders, calibration.
#
# **Rubric findings.** Lexical range (`mattr50`) and repetition rise and fall with the grade in both batches; error density from an
# LLM's minimal correction separates levels 3 and 4. Parse-based "complex structure" measures don't rise with the grade (reversed
# at 5 vs 4: weak speakers produce long unpunctuated run-ons). LanguageTool separates levels 3 and 4 on long clips but not on short
# ones. A surrogate fitted on the rubric features alone reproduces 61–68% of the variance of the final predictions (out of fold);
# accuracy features carry most of it, structure and range less, sentence completeness almost none (section 7).
#
# **Findings (human-centred).** Fillers and thinking pauses are never graded as grammar, and the speech embeddings encode accent
# and recording batch (a limitation). Section 7 recomputes the numbers:
# 1. *Restarts aren't self-correction.* A transcript restart rate goes with *lower* grades (Spearman −0.22): it measures
#    disfluency, and the rubric credits correction. An LLM's count of corrected vs uncorrected errors doesn't track the grade
#    either (ρ +0.07, 4 inversions).
# 2. *Short clips score lower than their measurable grammar.* At the same grade, the label-free signals rate short clips the same
#    or better (a zero-shot rubric judge gives ≈3.0 in every batch), yet short training clips are graded lower. It may be a rater,
#    recording (short and long clips stay separable at equal length, AUC 0.93) or amount effect; the model follows the labels.
# 3. *Amount spoken isn't graded, but the model rewards it.* On short clips the residual falls by 0.08 per 10 s of speech.
# 4. *CTC failures penalise accented speakers.* When the CTC transcriber fails (repair rate > 0.6: 11.6% of train, 1.9% of test),
#    good speakers (≥ 4) are under-scored by 0.69.
# 5. *The examiner sees what the stack misses on short clips.* Given the final prediction, its judgements still correlate with the
#    grade on short clips (median partial r +0.10, up to +0.19) and not on long clips (+0.03).
#
# *(Amrita's reading of the ten true 5.0 clips the model under-scores most: to add.)*

# %% [markdown]
# ## Appendix A. Stages that need the audio or a GPU (`RECOMPUTE_GPU`)
# The code that produced every cached file the sections above read but could not compute on this CPU session, ported from the
# notebooks that ran it on a Kaggle T4: `notebook26509aa315` (A0–A6), `shl-deberta-grouped` (A7) and `shl-qwen-features` (A8).
# With `RECOMPUTE_GPU = False` (the default) each cell only defines its function. With `True`, attach the competition data,
# switch on a GPU and internet (the models download from Hugging Face), and each stage writes its file to
# `/kaggle/working/recomputed/` under the cached name; attach that output in place of the caches to rerun the notebook on it.
# The full set takes several hours on a T4; the fine-tunes (A6, A7) and Qwen (A8) dominate.
#
# **Phase I jobs (not ported; each is a self-contained Kaggle notebook in the repository under `kaggle/<job>/`).** They produced
# the members the base and combination pipelines add to stages A1–A8. A6 (the fine-tuned WavLM) is no longer used: its
# random-fold out-of-fold column leaked speaker twins (see the report).
#
# | Job (`kaggle/…`, Kaggle notebook `amritarajput54/shl-…`) | Produces | Used by |
# |---|---|---|
# | `verbatim` (`shl-verbatim-text`) | Whisper-large-v3-turbo verbatim transcripts; DeBERTa-v3-base on (verbatim, Whisper clean) | combination |
# | `laya_checklist` (`shl-laya-checklist`) | Laya answers to rubric checklist questions per sentence | both |
# | `voxtral` (`shl-voxtral-features`) | Voxtral-Mini-3B states over the audio tokens, question after the audio | both |
# | `voxtral_frames` (`shl-voxtral-frames`) | the same with the question before the audio (two wordings) | combination |
# | `deberta_large_pair`, `deberta_large_pair_s2` | DeBERTa-v3-large on (CTC raw, Whisper clean), seeds 42/7/2024 and 11/22/33 | combination |
# | `qwen35_judge` (`shl-qwen35-judge`) | Qwen3.5-9B rubric and accuracy judgements on 3 transcripts; answer-position states | combination |
# | `qwen35_judge2` (`shl-qwen35-judge2`) | the examiner on complexity, completeness, CEFR level and two transcripts side by side | combination |
#
# Built locally from these caches (`src/`): the Voxtral, Qwen and speech-encoder ridge heads, the RBF-SVR head (`svr_heads.py`),
# Rank-N-Contrast (`rnc.py`, PyTorch on CPU), the examiner correction and pooling (`run.py`, `pooling.py`); `experiment.py`
# holds every flag of the two pipelines. Jobs tested but not in the final pick: `qwen35_judge3` (error profile), `qwen35_judge4`
# (self-consistency and few-shot anchors), `vox_attn_gpu` (attention pooling), `deberta_large_vpair`, `crop30`, `laya_dh`, `laya_ft`.

# %%
def audio_dirs():
    """The competition's train/ and test/ wav folders under /kaggle/input."""
    found = Counter(Path(d) for d, _, files in os.walk("/kaggle/input", followlinks=True) for f in files if f.lower().endswith(".wav"))
    pick = lambda name: max(((n, d) for d, n in found.items() if d.name.lower() == name), default=(0, None))[1]
    dirs = {"train": pick("train"), "test": pick("test")}
    assert None not in dirs.values(), "competition audio not found: add the competition data as input"
    return dirs


def load16k(path):
    """Mono float32 at 16 kHz."""
    import librosa
    import soundfile as sf
    try:
        x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        x, sr = librosa.load(str(path), sr=None, mono=False)
        x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1 if x.shape[0] > x.shape[1] else 0)
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(x, dtype=np.float32)


def all_keys():
    """(split, filename) for every labelled train clip and every test clip (the sentinels are never transcribed)."""
    return [("train", f) for f in train.filename] + [("test", f) for f in test.filename]


def a0_sentinel_flatness():
    """A0 (CPU, audio): median spectral flatness per clip, 1024-sample Hann frames, hop 512 -> sentinel_flatness.csv."""
    dirs, out = audio_dirs(), []
    kinds = [("sentinel", "train", f) for f in train_all.filename[train_all.label == 0]]
    kinds += [("normal", "train", f) for f in train.filename] + [("test", "test", f) for f in test.filename]
    for kind, split, f in kinds:
        sr, raw = wavfile.read(dirs[split] / f)
        x = raw.astype(np.float64)
        if np.issubdtype(raw.dtype, np.integer):
            x /= float(np.iinfo(raw.dtype).max) + 1.0
        x = x.mean(1) if x.ndim > 1 else x
        n = 1 + max(0, (len(x) - 1024) // 512)
        P = np.abs(np.fft.rfft(np.stack([x[i * 512:i * 512 + 1024] for i in range(n)]) * np.hanning(1024), axis=1)) ** 2 + 1e-12
        out.append({"kind": kind, "split": split, "filename": f, "flatness": float(np.median(np.exp(np.log(P).mean(1)) / P.mean(1)))})
    RAW.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(RAW / "sentinel_flatness.csv", index=False)


if RECOMPUTE_GPU:
    a0_sentinel_flatness()

# %%
def a1_whisper():
    """A1: Whisper small.en (faster-whisper, fp16), beam 5, word timestamps, no conditioning on previous text, no VAD -> asr_c.jsonl."""
    from faster_whisper import WhisperModel as FasterWhisper
    dirs, model = audio_dirs(), FasterWhisper("small.en", device="cuda", compute_type="float16")
    RAW.mkdir(parents=True, exist_ok=True)
    with open(RAW / "asr_c.jsonl", "w", encoding="utf-8") as fh:
        for split, f in all_keys():
            segs, info = model.transcribe(load16k(dirs[split] / f), language="en", beam_size=5, word_timestamps=True,
                                          condition_on_previous_text=False, vad_filter=False)
            S_, Wd = [], []
            for s in segs:
                S_.append(dict(start=round(s.start, 2), end=round(s.end, 2), text=s.text, avg_logprob=round(s.avg_logprob, 4),
                               no_speech_prob=round(s.no_speech_prob, 4), compression_ratio=round(s.compression_ratio, 3),
                               temperature=s.temperature))
                Wd += [[w.word, round(w.start, 2), round(w.end, 2), round(w.probability, 4)] for w in (s.words or [])]
            fh.write(json.dumps(dict(split=split, filename=f, duration=round(info.duration, 2),
                                     text="".join(s["text"] for s in S_).strip(), segments=S_, words=Wd)) + "\n")


if RECOMPUTE_GPU:
    a1_whisper()

# %%
def a2_ctc():
    """A2: wav2vec2-large-robust-ft-swbd-300h, frozen, greedy CTC decode in 20 s chunks, word times + mean log-posterior -> ctc.jsonl.
    Switchboard conventions keep filled pauses ("uh", "um"), and a CTC model doesn't repair grammar."""
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    name, chunk_s, dev = "facebook/wav2vec2-large-robust-ft-swbd-300h", 20.0, "cuda"
    proc = Wav2Vec2Processor.from_pretrained(name)
    model = Wav2Vec2ForCTC.from_pretrained(name).to(dev).eval()
    vocab = {v: k_ for k_, v in proc.tokenizer.get_vocab().items()}
    blank, delim = model.config.pad_token_id, proc.tokenizer.word_delimiter_token

    def ctc_words(audio):
        out, step = [], int(SR * chunk_s)
        for i in range(0, len(audio), step):
            chunk = audio[i:i + step]
            if len(chunk) < SR * 0.1:
                continue
            with torch.inference_mode():
                logits = model(proc(chunk, sampling_rate=SR, return_tensors="pt", padding=True).input_values.to(dev)).logits[0]
            best = torch.log_softmax(logits.float(), dim=-1).max(dim=-1)
            ids, conf = best.indices.cpu().numpy(), best.values.cpu().numpy()
            frame_s, offset = len(chunk) / SR / len(ids), i / SR
            chars, confs, start_f, prev = [], [], None, -1

            def flush(end_f):
                if chars and start_f is not None:
                    out.append(["".join(chars).lower(), round(offset + start_f * frame_s, 3), round(offset + end_f * frame_s, 3),
                                round(float(np.mean(confs)), 4)])
                chars.clear(); confs.clear()

            for fr, tid in enumerate(ids):
                tid = int(tid)
                if tid == blank:
                    prev = tid
                    continue
                if tid == prev:                       # CTC repeat collapse
                    confs.append(float(conf[fr]))
                    continue
                prev = tid
                tok = vocab.get(tid, "")
                if tok == delim:
                    flush(fr)
                    start_f = None
                    continue
                if start_f is None:
                    start_f = fr
                chars.append(tok); confs.append(float(conf[fr]))
            flush(len(ids) - 1)
        return [w for w in out if w[0].strip()]

    dirs = audio_dirs()
    RAW.mkdir(parents=True, exist_ok=True)
    with open(RAW / "ctc.jsonl", "w", encoding="utf-8") as fh:
        for split, f in all_keys():
            audio = load16k(dirs[split] / f)
            wd = ctc_words(audio)
            fh.write(json.dumps(dict(split=split, filename=f, duration=round(len(audio) / SR, 2),
                                     raw_text=" ".join(w[0] for w in wd), words=wd)) + "\n")


if RECOMPUTE_GPU:
    a2_ctc()

# %%
def a3_text_models():
    """A3: GPT-2 loss on the raw (CTC) and clean (Whisper) text and their gap; CoLA acceptability per Whisper sentence;
    all-mpnet-base-v2 sentence embeddings of the Whisper text -> text_gpu.csv, emb_mpnet.npz."""
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda"
    Wt = load_jsonl(RAW / "asr_c.jsonl") if (RAW / "asr_c.jsonl").exists() else W
    Ct = load_jsonl(RAW / "ctc.jsonl") if (RAW / "ctc.jsonl").exists() else Cc
    tok = AutoTokenizer.from_pretrained("gpt2")
    lm = AutoModelForCausalLM.from_pretrained("gpt2").to(dev).eval()

    def nll(text):
        if not text or len(text.split()) < 3:
            return np.nan
        ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(dev)
        with torch.inference_mode():
            return float(lm(ids, labels=ids).loss)

    out = pd.DataFrame({"split": [k_[0] for k_ in keys], "filename": [k_[1] for k_ in keys]})
    out["gpt2_nll_raw"] = [nll(Ct[k_]["raw_text"]) for k_ in keys]
    out["gpt2_nll_clean"] = [nll(Wt[k_]["text"]) for k_ in keys]
    out["gpt2_nll_gap"] = out.gpt2_nll_raw - out.gpt2_nll_clean
    ctok = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
    cmod = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA").to(dev).eval()

    def p_all(sents, bs=64):
        res_ = []
        for i in range(0, len(sents), bs):
            b = ctok(sents[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
            with torch.inference_mode():
                res_.append(torch.softmax(cmod(**b).logits.float(), -1).cpu().numpy())
        return np.vstack(res_)

    probe = p_all(["The cat sat on the mat.", "The cat sit on mat the."])
    good = int(np.argmax(probe[0] - probe[1]))             # which output column means "grammatical"
    assert probe[0, good] > probe[1, good] + 0.2, "CoLA probe did not separate"
    split_sents = lambda t: ([x.strip() for x in re.split(r"(?<=[.!?])\s+", t or "") if len(x.split()) >= 3]
                             or ([t.strip()] if t and len(t.split()) >= 3 else []))
    flat, owner = [], []
    for k_ in keys:
        for s in split_sents(Wt[k_]["text"]):
            flat.append(s); owner.append(k_)
    Pc = p_all(flat)[:, good]
    by = {}
    for k_, p, s in zip(owner, Pc, flat):
        by.setdefault(k_, []).append((p, len(s.split())))
    agg = {}
    for k_ in keys:
        v = by.get(k_, [])
        if not v:
            agg[k_] = dict(cola_mean=np.nan, cola_wmean=np.nan, cola_min=np.nan, cola_frac_bad=np.nan)
            continue
        p, n = np.array([a for a, _ in v]), np.array([b for _, b in v], float)
        agg[k_] = dict(cola_mean=p.mean(), cola_wmean=(p * n).sum() / n.sum(), cola_min=p.min(), cola_frac_bad=(p < 0.5).mean())
    for c in ("cola_mean", "cola_wmean", "cola_min", "cola_frac_bad"):
        out[c] = [agg[k_][c] for k_ in keys]
    RAW.mkdir(parents=True, exist_ok=True)
    out.to_csv(RAW / "text_gpu.csv", index=False)
    st = SentenceTransformer("all-mpnet-base-v2", device=dev)
    Ee = st.encode([Wt[k_]["text"] for k_ in keys], batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    np.savez(RAW / "emb_mpnet.npz", split=np.array([k_[0] for k_ in keys]), filename=np.array([k_[1] for k_ in keys]), E=Ee)


if RECOMPUTE_GPU:
    a3_text_models()

# %%
def a4_speech_encoders():
    """A4: frozen encoders, every hidden layer, mean + std over time (wav2vec2 family: 20 s chunks; Whisper encoder: 30 s windows,
    padded frames excluded) -> wavlm_layers.npz, ssl_wavlm_large.npz, ssl_whisper_med.npz (H: clips x layers x 2*dims).
    The ridge heads on top are refit in section 5 (uniform layer mean, no layer chosen by validation)."""
    import torch
    from transformers import AutoFeatureExtractor, AutoModel, Wav2Vec2FeatureExtractor
    from transformers import WhisperModel as HFWhisper
    dev = "cuda"
    models = {"wavlm_layers.npz": "microsoft/wavlm-base-plus", "ssl_wavlm_large.npz": "microsoft/wavlm-large",
              "ssl_whisper_med.npz": "openai/whisper-medium"}
    dirs = audio_dirs()

    def accumulate(hs, n_valid, acc):
        h = hs[:, :n_valid]
        s, sq = h.sum(1), (h ** 2).sum(1)
        if acc is None:
            return [s, sq, n_valid]
        acc[0] += s; acc[1] += sq; acc[2] += n_valid
        return acc

    def finish(acc):
        mean = acc[0] / acc[2]
        return torch.cat([mean, (acc[1] / acc[2] - mean ** 2).clamp(min=0).sqrt()], -1).cpu().numpy().astype(np.float32)

    def pool_w2v(model, fe, audio):
        acc, step = None, int(SR * 20.0)
        for j in range(0, len(audio), step):
            chunk = audio[j:j + step]
            if len(chunk) < SR // 2:
                continue
            x = fe(chunk, sampling_rate=SR, return_tensors="pt").input_values.to(dev)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                hs = torch.stack(model(x, output_hidden_states=True).hidden_states)[:, 0].float()
            acc = accumulate(hs, hs.shape[1], acc)
        return finish(acc)

    def pool_whisper(enc, fe, audio):
        acc, step = None, 30 * SR
        for j in range(0, len(audio), step):
            chunk = audio[j:j + step]
            if len(chunk) < SR // 2:
                continue
            x = fe(chunk, sampling_rate=SR, return_tensors="pt").input_features.to(dev)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                hs = torch.stack(enc(x, output_hidden_states=True).hidden_states)[:, 0].float()
            acc = accumulate(hs, min(1500, math.ceil(len(chunk) / SR * 50)), acc)     # 50 encoder frames per second of audio
        return finish(acc)

    RAW.mkdir(parents=True, exist_ok=True)
    for file, name in models.items():
        if "whisper" in name:
            fe, model, pool = AutoFeatureExtractor.from_pretrained(name), HFWhisper.from_pretrained(name).encoder, pool_whisper
        else:
            try:
                fe = AutoFeatureExtractor.from_pretrained(name)
            except Exception:
                fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SR, padding_value=0.0, do_normalize=True,
                                              return_attention_mask=True)
            model, pool = AutoModel.from_pretrained(name), pool_w2v
        model = model.to(dev).float().eval()               # fp32 weights, fp16 compute under autocast
        H = np.stack([pool(model, fe, load16k(dirs[s] / f)) for s, f in keys])
        np.savez(RAW / file, split=np.array([k_[0] for k_ in keys]), filename=np.array([k_[1] for k_ in keys]), H=H)
        del model
        torch.cuda.empty_cache()


if RECOMPUTE_GPU:
    a4_speech_encoders()

# %%
def a5_xvectors():
    """A5: speaker x-vectors (microsoft/wavlm-base-plus-sv): 20 s chunks, unit-length, length-weighted mean -> spk_xvec.npz."""
    import torch
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    name, dev = "microsoft/wavlm-base-plus-sv", "cuda"
    fe = Wav2Vec2FeatureExtractor.from_pretrained(name)
    sv = WavLMForXVector.from_pretrained(name).to(dev).float().eval()
    dirs, vecs, step = audio_dirs(), [], 20 * SR
    for split, f in keys:
        audio, embs, lens = load16k(dirs[split] / f), [], []
        for j in range(0, len(audio), step):
            chunk = audio[j:j + step]
            if len(chunk) < 3 * SR and embs:          # skip short tails once there is a chunk
                continue
            with torch.inference_mode():
                e = torch.nn.functional.normalize(sv(input_values=fe(chunk, sampling_rate=SR, return_tensors="pt")
                                                     .input_values.to(dev)).embeddings, dim=-1)
            embs.append(e[0].float().cpu().numpy()); lens.append(len(chunk))
        v = np.average(embs, axis=0, weights=lens)
        vecs.append(v / np.linalg.norm(v))
    RAW.mkdir(parents=True, exist_ok=True)
    np.savez(RAW / "spk_xvec.npz", split=np.array([k_[0] for k_ in keys]), filename=np.array([k_[1] for k_ in keys]),
             E=np.stack(vecs).astype(np.float32))


if RECOMPUTE_GPU:
    a5_xvectors()

# %%
def a6_finetune_wavlm():
    """NOT USED BY THE FINAL PICK (its random-fold OOF leaked speaker twins; kept for the record). A6: WavLM-base-plus fine-tuned end to end (learned softmax over hidden layers, mean pooling, linear head; CNN front end
    frozen; 8 epochs; random 8 s crops, 2 per clip per epoch; overlapping 8 s windows, hop 4 s, averaged at prediction; target
    standardised per fold; final epoch, no early stopping), seeds 42 / 7 / 2024 on the RANDOM seed-42 StratifiedKFold folds the run
    used (hence the twin contamination noted in Limitations) -> ft_wavlm.csv (ft_wavlm_pred: OOF for train, fold mean for test)."""
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoFeatureExtractor, AutoModel, get_cosine_schedule_with_warmup
    name, dev, epochs, bs, lr, lr_head, wd, warmup = "microsoft/wavlm-base-plus", "cuda", 8, 8, 3e-5, 1e-3, 0.01, 0.10
    crop, hop = 8 * SR, 4 * SR
    dirs = audio_dirs()
    TR_W = [load16k(dirs["train"] / f) for f in T0.filename]
    TE_W = [load16k(dirs["test"] / f) for f in TE0.filename]
    yy = T0.label.values.astype(float)
    folds_rand = list(StratifiedKFold(5, shuffle=True, random_state=42).split(T0, strat))
    fe = AutoFeatureExtractor.from_pretrained(name)

    def random_crop(w, rg):
        if len(w) <= crop:
            return np.pad(w, (0, crop - len(w)))
        s = int(rg.integers(0, len(w) - crop + 1))
        return w[s:s + crop]

    def windows(w):
        if len(w) <= crop:
            return [np.pad(w, (0, crop - len(w)))]
        starts = list(range(0, len(w) - crop + 1, hop))
        if starts[-1] != len(w) - crop:
            starts.append(len(w) - crop)
        return [w[s:s + crop] for s in starts]

    class SSLReg(nn.Module):
        def __init__(self):
            super().__init__()
            cfg = AutoConfig.from_pretrained(name)
            cfg.layerdrop = 0.0
            self.bb = AutoModel.from_pretrained(name, config=cfg)
            for p in self.bb.feature_extractor.parameters():
                p.requires_grad = False
            self.lw = nn.Parameter(torch.zeros(cfg.num_hidden_layers + 1))
            self.head = nn.Linear(cfg.hidden_size, 1)

        def forward(self, x):
            hs = torch.stack(self.bb(x, output_hidden_states=True).hidden_states)
            return self.head((torch.softmax(self.lw, 0)[:, None, None, None] * hs).sum(0).mean(1)).squeeze(-1)

    def predict(model, clips):
        model.eval()
        owner, wins = [], []
        for i, w in enumerate(clips):
            ws = windows(w)
            wins += ws; owner += [i] * len(ws)
        out = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for b in range(0, len(wins), 16):
                out.append(model(fe(wins[b:b + 16], sampling_rate=SR, return_tensors="pt").input_values.to(dev)).float().cpu().numpy())
        out, owner = np.concatenate(out), np.array(owner)
        return np.array([out[owner == i].mean() for i in range(len(clips))])

    oof_, te_ = np.zeros(len(yy)), np.zeros(len(TE_W))
    for seed in (42, 7, 2024):
        for a, b in folds_rand:
            torch.manual_seed(seed)
            rg = np.random.default_rng(seed)
            mu, sd = yy[a].mean(), yy[a].std()
            model = SSLReg().to(dev).float()
            opt = torch.optim.AdamW([{"params": [p for p in model.bb.parameters() if p.requires_grad], "lr": lr, "weight_decay": wd},
                                     {"params": [model.lw, *model.head.parameters()], "lr": lr_head, "weight_decay": 0.0}])
            total = epochs * math.ceil(len(a) * 2 / bs)
            sch = get_cosine_schedule_with_warmup(opt, int(warmup * total), total)
            scaler = torch.amp.GradScaler("cuda")
            for _ in range(epochs):
                model.train()
                order_ = rg.permutation(np.repeat(a, 2))
                for i in range(0, len(order_), bs):
                    ids = order_[i:i + bs]
                    x = fe([random_crop(TR_W[j], rg) for j in ids], sampling_rate=SR, return_tensors="pt").input_values.to(dev)
                    yb = torch.tensor((yy[ids] - mu) / sd, dtype=torch.float32, device=dev)
                    with torch.autocast("cuda", dtype=torch.float16):
                        pred = model(x)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(nn.functional.mse_loss(pred.float(), yb)).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update(); sch.step()
            oof_[b] += (predict(model, [TR_W[j] for j in b]) * sd + mu) / 3
            te_ += (predict(model, TE_W) * sd + mu) / 15
            del model, opt
            torch.cuda.empty_cache()
    RAW.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"split": ["train"] * len(yy) + ["test"] * len(te_), "filename": list(T0.filename) + list(TE0.filename),
                  "ft_wavlm_pred": np.r_[oof_, te_]}).to_csv(RAW / "ft_wavlm.csv", index=False)


if RECOMPUTE_GPU:
    a6_finetune_wavlm()

# %%
def a7_deberta():
    """A7: DeBERTa-v3-base regressor (mean pooling, linear head, dropout off, 6 epochs, lr 2e-5, batch 8, cosine schedule with
    10% warm-up, target standardised per fold, final epoch) on the speaker-grouped seed-42 folds, seeds 42 / 7 / 2024; inputs:
    the Whisper text (deberta_pred) and the pair (CTC raw, Whisper clean) (deberta_pair_pred) -> deberta_grouped.csv."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoConfig, AutoModel, AutoTokenizer, DataCollatorWithPadding, get_cosine_schedule_with_warmup
    name, dev, epochs, lr, bs, wd, warmup = "microsoft/deberta-v3-base", "cuda", 6, 2e-5, 8, 0.01, 0.10
    tok = AutoTokenizer.from_pretrained(name)
    collate = DataCollatorWithPadding(tok)
    yy = T0.label.values.astype(np.float64)

    class DS(Dataset):
        def __init__(self, pairs, max_len, yv=None):
            self.pairs, self.max_len, self.yv = pairs, max_len, yv

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, i):
            a, b = self.pairs[i]
            e = dict(tok(a, b, truncation=True, max_length=self.max_len))
            e.pop("token_type_ids", None)
            if self.yv is not None:
                e["labels"] = float(self.yv[i])
            return e

    class Reg(nn.Module):
        def __init__(self):
            super().__init__()
            cfg = AutoConfig.from_pretrained(name)
            cfg.hidden_dropout_prob = cfg.attention_probs_dropout_prob = 0.0
            self.bb = AutoModel.from_pretrained(name, config=cfg)
            self.head = nn.Linear(cfg.hidden_size, 1)

        def forward(self, input_ids, attention_mask):
            h = self.bb(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            return self.head((h * m).sum(1) / m.sum(1).clamp(min=1e-6)).squeeze(-1)

    def predict(model, dl):
        model.eval()
        out = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for bt in dl:
                out.append(model(bt["input_ids"].to(dev), bt["attention_mask"].to(dev)).float().cpu().numpy())
        return np.concatenate(out)

    result = {"split": ["train"] * len(T0) + ["test"] * len(TE0), "filename": list(T0.filename) + list(TE0.filename)}
    for inp, col in (("whisper", "deberta_pred"), ("ctc_pair", "deberta_pair_pred")):
        txt = lambda s, f: (W[(s, f)]["text"], None) if inp == "whisper" else (Cc[(s, f)]["raw_text"], W[(s, f)]["text"])
        tr_txt, te_txt = [txt("train", f) for f in T0.filename], [txt("test", f) for f in TE0.filename]
        max_len = int(min(512, np.percentile([len(tok(a, b)["input_ids"]) for a, b in tr_txt], 99) + 16))
        oof_, te_ = np.zeros(len(yy)), np.zeros(len(te_txt))
        for seed in (42, 7, 2024):
            for a, b in FOLDS[42]:
                torch.manual_seed(seed); np.random.seed(seed)
                mu, sd = yy[a].mean(), yy[a].std()
                g = torch.Generator(); g.manual_seed(seed)
                tr_dl = DataLoader(DS([tr_txt[i] for i in a], max_len, (yy[a] - mu) / sd), batch_size=bs, shuffle=True,
                                   collate_fn=collate, generator=g)
                model = Reg().to(dev).float()
                no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
                opt = torch.optim.AdamW([
                    {"params": [p for n, p in model.named_parameters() if not any(k_ in n for k_ in no_decay)], "weight_decay": wd},
                    {"params": [p for n, p in model.named_parameters() if any(k_ in n for k_ in no_decay)], "weight_decay": 0.0}], lr=lr)
                sch = get_cosine_schedule_with_warmup(opt, int(warmup * epochs * len(tr_dl)), epochs * len(tr_dl))
                scaler = torch.cuda.amp.GradScaler()
                for _ in range(epochs):
                    model.train()
                    for bt in tr_dl:
                        with torch.autocast("cuda", dtype=torch.float16):
                            pred = model(bt["input_ids"].to(dev), bt["attention_mask"].to(dev))
                        loss = nn.functional.mse_loss(pred.float(), bt["labels"].float().to(dev))
                        opt.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(opt); scaler.update(); sch.step()
                va_dl = DataLoader(DS([tr_txt[i] for i in b], max_len), batch_size=32, collate_fn=collate)
                oof_[b] += (predict(model, va_dl) * sd + mu) / 3
                te_ += (predict(model, DataLoader(DS(te_txt, max_len), batch_size=32, collate_fn=collate)) * sd + mu) / 15
                del model, opt, sch, scaler
                torch.cuda.empty_cache()
        result[col] = list(oof_) + list(te_)
    RAW.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result).to_csv(RAW / "deberta_grouped.csv", index=False)


if RECOMPUTE_GPU:
    a7_deberta()

# %%
QWEN_RUBRIC = """1: The person's speech struggles with proper sentence structure and syntax, displaying limited control over simple grammatical structures and memorized sentence patterns.
2: The person has a limited understanding of sentence structure and syntax. Although they use simple structures, they consistently make basic sentence structure and grammatical mistakes. They might leave sentences incomplete.
3: The person demonstrates a decent grasp of sentence structure but makes errors in grammatical structure, or they show a decent grasp of grammatical structure but make errors in sentence syntax and structure.
4: The person displays a strong understanding of sentence structure and syntax. They consistently show good control of grammar. While occasional errors may occur, they are generally minor and do not lead to misunderstandings; the person can correct most of them.
5: Overall, the person showcases high grammatical accuracy and adept control of complex grammar. They use grammar accurately and effectively, seldom making noticeable mistakes. Additionally, they handle complex language structures well and correct themselves when necessary."""


def a8_qwen():
    """A8: Qwen2.5-7B-Instruct, 4-bit NF4. Stage 1: the rubric prompt on each transcript (CTC and Whisper): hidden states of the
    last token and the mean over the transcript's tokens at layers 16 / 20 / 24, and P(next token = 1..5) as a zero-shot judge.
    Stages 2-3 (CTC text, greedy): a minimal grammar correction and the corrected / uncorrected error counts as JSON.
    -> qwen_hidden.npz, qwen_judge.csv, qwen_gen.jsonl. The ridge heads on the hidden states are refit in section 5."""
    if importlib.util.find_spec("bitsandbytes") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    name, layers, max_text, b_fwd, b_gen = "Qwen/Qwen2.5-7B-Instruct", (16, 20, 24), 600, 8, 8
    torch.manual_seed(SEED)
    ks = sorted(set(W) & set(Cc))
    texts = {"ctc": [Cc[k_]["raw_text"] or "" for k_ in ks], "whisper": [W[k_]["text"] or "" for k_ in ks]}
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, device_map={"": 0}, torch_dtype=torch.float16, quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)).eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    digits = [tok.convert_tokens_to_ids(str(d)) for d in range(1, 6)]

    def chat_pieces(system, before, text, after):
        """Prompt token ids, split so the transcript's token span is known exactly."""
        marker = "⁣TRANSCRIPT⁣"
        full = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": before + marker + after}],
                                       tokenize=False, add_generation_prompt=True)
        pre_, post = full.split(marker)
        a = tok(pre_, add_special_tokens=False).input_ids
        b = tok(text, add_special_tokens=False).input_ids[:max_text]
        return a + b + tok(post, add_special_tokens=False).input_ids, (len(a), len(a) + len(b))

    def left_pad(seqs):
        n = max(len(s) for s in seqs)
        ids, att = torch.full((len(seqs), n), pad, dtype=torch.long), torch.zeros((len(seqs), n), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, n - len(s):] = torch.tensor(s); att[i, n - len(s):] = 1
        return ids.cuda(), att.cuda(), [n - len(s) for s in seqs]

    judge = {"split": [k_[0] for k_ in ks], "filename": [k_[1] for k_ in ks]}
    hidden = {"split": np.array(judge["split"]), "filename": np.array(judge["filename"])}
    before = f"Grammar rubric (score levels 1 to 5):\n{QWEN_RUBRIC}\n\nAutomatic transcript of a spoken response:\n\"\"\""
    after = "\"\"\"\n\nRate the grammar of this response from 1 to 5 using the rubric. Answer with a single digit."
    for tname, tx in texts.items():
        built = [chat_pieces("You are an expert assessor of spoken English grammar.", before, t, after) for t in tx]
        order_ = np.argsort([len(s) for s, _ in built])
        last = np.zeros((len(tx), len(layers), model.config.hidden_size), np.float16)
        mean = np.zeros_like(last)
        probs = np.zeros((len(tx), 5), np.float32)
        for b0 in range(0, len(order_), b_fwd):
            ix = order_[b0:b0 + b_fwd]
            ids, att, pads = left_pad([built[i][0] for i in ix])
            with torch.inference_mode():           # base model + lm_head on the last position only (no full logits)
                out = model.model(input_ids=ids, attention_mask=att, output_hidden_states=True)
                logits = model.lm_head(out.last_hidden_state[:, -1, :]).float()
            p = torch.softmax(logits[:, digits], dim=-1).cpu().numpy()
            for j, i in enumerate(ix):
                lo_, hi_ = built[i][1][0] + pads[j], built[i][1][1] + pads[j]
                for li, layer in enumerate(layers):
                    h = out.hidden_states[layer][j]
                    last[i, li] = h[-1].float().cpu().numpy()
                    mean[i, li] = (h[lo_:hi_].float().mean(0) if hi_ > lo_ else h[-1].float()).cpu().numpy()
                probs[i] = p[j]
        hidden[f"{tname}_last"], hidden[f"{tname}_mean"] = last, mean
        for d in range(5):
            judge[f"{tname}_p{d + 1}"] = probs[:, d]
    RAW.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW / "qwen_hidden.npz", layers=np.array(layers), **hidden)
    pd.DataFrame(judge).to_csv(RAW / "qwen_judge.csv", index=False)

    system = "You are a careful English grammar editor for transcripts of non-native spoken English."
    pre = "This is an automatic transcript of spoken English. It has no punctuation and some words are misrecognized.\n\nTranscript:\n\"\"\""
    corr_after = ("\"\"\"\n\nRewrite it with the MINIMUM changes needed to make it grammatical. Fix only grammar: verb forms, agreement, "
                  "articles, prepositions, word order and missing words. Do not fix spelling, do not replace words you cannot "
                  "understand, do not paraphrase, do not add punctuation. Output only the corrected transcript.")
    sc_after = ("\"\"\"\n\nCount the grammatical errors the speaker made. (a) corrected: errors the speaker corrected themselves, "
                "e.g. said a wrong form and then restated it correctly. (b) uncorrected: errors left uncorrected. Ignore fillers, "
                "pauses, repetitions without a change, and misrecognized words. Answer only with JSON: "
                "{\"corrected\": <int>, \"uncorrected\": <int>}")

    def generate(after_, max_new):
        built = [chat_pieces(system, pre, t, after_) for t in texts["ctc"]]
        order_ = np.argsort([len(s) for s, _ in built])
        outs = [""] * len(built)
        for b0 in range(0, len(order_), b_gen):
            ix = order_[b0:b0 + b_gen]
            ids, att, _ = left_pad([built[i][0] for i in ix])
            with torch.inference_mode():
                g = model.generate(input_ids=ids, attention_mask=att, do_sample=False, pad_token_id=pad,
                                   max_new_tokens=max_new(max(built[i][1][1] - built[i][1][0] for i in ix)))
            for j, i in enumerate(ix):
                outs[i] = tok.decode(g[j, ids.shape[1]:], skip_special_tokens=True).strip()
        return outs

    corrected = generate(corr_after, lambda n: min(int(1.3 * n) + 24, 800))
    counts = generate(sc_after, lambda n: 32)
    with open(RAW / "qwen_gen.jsonl", "w", encoding="utf-8") as fh:
        for k_, c, s in zip(ks, corrected, counts):
            fh.write(json.dumps({"split": k_[0], "filename": k_[1], "correction": c, "selfcorr_json": s}) + "\n")


if RECOMPUTE_GPU:
    a8_qwen()
RUNTIME["total"] = time.time() - _t0
print(pd.Series(RUNTIME).round(1).to_frame("seconds").to_string())
