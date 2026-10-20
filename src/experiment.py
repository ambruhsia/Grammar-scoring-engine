"""THE FILE THE EXPERIMENT LOOP EDITS.

Contract (run.py relies on it):
    make_candidates(ctx) -> {name: (oof_per_seed, test_per_seed)}
        oof_per_seed  : list of arrays, one per seed in ctx.seeds, each len(ctx.y), out-of-fold
        test_per_seed : list of arrays, one per seed, each len(ctx.TE)
        Every OOF array must come from ctx.folds[seed] (or the seed-42 folds the *_pred
        columns were built on). Never fit anything on rows you then score.
    blend(CO, y) or blend(CO, y, w) -> weights   (optional; default is the unweighted greedy hill-climb
        in common.py). w are the clip weights ctx.w (test-mix), passed when blend takes three arguments.

ctx fields: T, TE (train/test tables), y, FEATS, med, seeds, folds (speaker-grouped by default),
            fold_scheme, groups (speaker group per train clip), batch / batch_test ("short" / "long"),
            w (test-mix clip weights, mean 1), oof_test(model, cols, folds),
            cache (path to cache/: features.csv, asr_c.jsonl, ctc.jsonl, *.npz if pulled with --all)
Note: the *_pred columns from the notebook are OOF on its random seed-42 folds, so under grouped folds
they still carry speaker-twin information.

The starting point below reproduces the notebook's §9 (v5.1): ablation, 9 model families x seeds,
learned *_pred columns entering directly. Its CV must match the notebook's printed
"OOF CV RMSE (honest)" before any change is trusted.
"""
from __future__ import annotations

import lightgbm as lgb
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from common import CLIP, rmse

CORE = ["repair_wer_nofiller", "gpt2_nll_raw", "gpt2_nll_gap", "n_words", "text_ttr",
        "articulation_rate", "mean_word_len", "mean_sent_len", "repetition_rate", "no_punct"]
DROP_DURATION = False       # adversarial validation (AUC ~0.75) traced the train/test shift to duration
DROP_AMOUNT = False         # G9: remove inputs measuring how MUCH someone spoke (raw counts); rates stay
AMOUNT_COLS = {"duration", "n_words"}
CTC_FAIL_NEUTRAL = False    # G10: repair_wer_nofiller above CTC_FAIL_CUT measures recognition failure, not grammar ->
CTC_FAIL_CUT = 0.6          #      set to missing (median-imputed) in train and test; cut fixed from the G2 table
REPAIR_COLS = ["repair_wer_nofiller"]
# Rubric layer in the model: none for now. Phase B passers mattr50 + trigram_repeat_rate failed the gate on their
# own (1012-182525, +0.0006) and stay for interpretation; self_repair_per100 is excluded (measures disfluent
# restarts, not self-correction). C3 adds corrected_error_ratio / llm_err_per100 candidates.
RUBRIC_LAYER = []
AUDIO_HEADS = False         # C1 (off: gate rejected, see LOG): grouped-OOF frozen-encoder heads replace the random-fold audio *_pred columns
AUDIO_HEADS_ONLY = None     # diagnostic: restrict to a subset of head columns (None = all six)
QWEN_HEADS = True           # C3: qwen_ctc_pred / qwen_whisper_pred, ridge on Qwen rubric-prompt hidden states
QWEN_DENOISED = False       # Phase F check: fit the Qwen heads on the denoised target like the other members (False = raw y)
LAYA_HEADS = False          # new member (approved 2026-10-14): laya_ctc_pred / laya_whisper_pred (src/laya_feats.py)
LAYA_FT = False             # new member (approved 2026-10-14): laya_ft_pair_pred, Laya's encoder fine-tuned on the pair
SVR_HEADS = False           # idea #2 (approved 2026-10-14): RBF-SVR heads on single upper encoder layers (src/svr_heads.py)
VOXTRAL = ("voxtral_pred",)  # idea #1 (approved 2026-10-14): members from src/voxtral_feats.py, e.g. ("voxtral_pred",)
VERBATIM_QWEN = False       # idea #3: qwen_verbatim_pred, Qwen heads on the verbatim transcript (src/verbatim_feats.py)
VERBATIM_DEBERTA = False    # idea #3: deberta_vpair_pred, DeBERTa on (verbatim, Whisper clean), fixed grouped OOF
LAYA_DH = ()              # idea #5: fixed OOF columns from the Laya decision-head job (cache/laya/laya_dh.csv),
                            #   e.g. ("laya_dh_thr_pred",); OOF on the grouped seed-42 folds like the DeBERTa columns
LAYA_CHECKLIST = ["chk_same_meaning_mean", "chk_same_meaning_max", "chk_grammar_only_max", "rub_ctc_ev"]  # idea #4 (gate A): Laya checklist features that pass the selection checks (src/laya_checklist.py)
                            #   (kaggle/laya_ft job -> cache/laya/laya_ft_pair.csv; OOF on the grouped seed-42 folds)
QWEN_RUBRIC = False         # C3: llm_err_per100 + corrected_error_ratio (if they pass the Phase B checks)
QWEN_JUDGE = False          # C3: zero-shot rubric judge (llm_rubric_expected, llm_rubric_entropy)
JUDGE_PLUS = False          # Phase I: Qwen2.5-7B zero-shot judge on the CTC, Whisper and verbatim transcripts as family inputs:
                            #   log-odds of grade >= 4 (judge_*_lo); they correlate +0.11..+0.13 with the honest stack's residuals
JUDGE_MEMBER = False        # Phase I: the same six judge numbers (expected grade + log-odds, 3 views) -> RidgeCV member judge_pred
QWEN35_HEADS = ()           # Phase I: Qwen3.5-9B answer-position heads (src/qwen35_feats.py), e.g. ("qwen35_verbatim_pred",)
QWEN35_JUDGE = False        # Phase I: Qwen3.5-9B judge numbers (12) -> RidgeCV member qwen35_judge_pred
DEBERTA_LARGE = None        # Phase I: DeBERTa-v3-large on (CTC raw, Whisper clean), grouped seed-42 OOF (kaggle/deberta_large_pair,
                            #   3 seeds, 0.551 alone): None | "add" | "replace" (drops the base deberta_pair_pred)
DEBERTA_LARGE_V = None      # Phase I: DeBERTa-v3-large on (verbatim, Whisper clean) (kaggle/deberta_large_vpair): None | "add" |
                            #   "replace" (drops the base deberta_vpair_pred)
LEAK_FREE = True            # every learned column OOF on grouped folds (src/leakfree.py); leaky ones dropped
DROP_FT_WAVLM = True        # C5 rule: drop the twin-contaminated fine-tuned WavLM column
SHRINK_LAMBDA = 0.25        # Phase I, BLENDER "nnls_batch_shrink": weight of the long rows in the short-batch NNLS (offline sim on
                            #   1015-112805: 0 -> 0.5052, 0.2 -> 0.5044, 0.35 -> 0.5043; 3/3 seeds better)
BLENDER = "nnls_batch"         # D2: "hill" (unweighted hill-climb), "nnls_w" (NNLS, test-mix weights), "nnls_batch3" (Phase I),
                            #     "nnls_batch" (NNLS per recording batch, test-mix weights)
DENOISE_LAMBDA = 0.5        # D3: members fit on (1-l)*y + l*speaker-group mean; scored on y
CALIBRATE = False           # D4: True = one nested weighted linear calibration; G5(c) "batch" = a_batch + c_batch*pred
NESTED_BLEND = True         # blend weights refit inside each fold (honest blend OOF; part of the leak-free scheme)
TOPIC_OOF = True            # G11: also build topic-held-out OOF for every locally refittable member (-> cv_newtopic)
BLEND_ON_TOPIC = True       # G11: fit the blend weights on topic-held-out OOF instead of speaker-grouped OOF
RNC_HEAD = False            # new (approved 2026-10-13): Rank-N-Contrast member rnc_pred (src/rnc.py)
VOX_ATTN = False            # Phase I: attention pooling over Voxtral frames on random test-length crops (src/vox_attn.py)
VOXTRAL3 = False            # Phase I: voxtral3_pred = mean of the ridge predictions on views A (audio first), B, C (question first)
TRAIN_FIT = False           # Phase F: also refit the final pipeline on all 732 rows (in-sample) for the TRAINING RMSE
POSTPROC = None             # new (approved 2026-10-13): None | "ordinal" | "ordinal_em" (src/ordinal_em.py, run.py)
POOL_S = 0                  # Phase I: same-speaker pooling strength (src/pooling.py; 0 = off), applied after the blend
LEACE_TEXT = False          # Phase I (2026-10-17): erase the prompt topic from the text-embedding heads' inputs (src/leace.py):
                            #   mpnet sentence embeddings, Qwen2.5 and Qwen3.5 hidden states; RNC keeps its cached inputs
BATCH_NORM_AUDIO = False    # Phase I (2026-10-17): audio embeddings z-scored within each recording batch (train + test, no labels)
                            #   + a short-batch indicator in the ridge heads (src/batchnorm.py); RNC keeps its cached inputs
EXAMINER = "old"            # Phase I (2026-10-17): which rubric grade the correction sees for the Whisper + verbatim transcripts:
                            #   "old" = round 1's single prompt (expected grade + log-odds); "sc" = self-consistency: mean and
                            #   spread of 6 paraphrased rubric prompts (round 1 + judge round 4); "sc_fs" = as sc, the Whisper mean
                            #   also including the few-shot grade; "fs" = the Whisper rubric grade replaced by the few-shot grade
RESID_B45 = False           # Phase I (2026-10-16): in the residual correction, also pass every judge number multiplied by the
                            #   exact-45.06 s indicator, so the examiner scores get their own slope on that recording batch (where
                            #   their partial correlation with the grade given the stack's prediction is +0.11..+0.31, vs ~0 on long)
RESID_PROFILE = False       # Phase I (2026-10-16): add the Qwen3.5 error profile (judge round 3: expected count of each error type,
                            #   self-corrections and complex-structure range, + rates per 100 words) to the residual-correction features
RESID_CORR = False          # Phase I: after the blend, a ridge on [judge numbers, prediction] predicts the residual (nested in folds,
                            #   test-mix weighted; run.py). Judge numbers: Qwen3.5-9B rounds 1 + 2 (resid_features below)
CROP_HEADS = False          # G6: ridge heads on first-30-s crop embeddings (WavLM-large L20-22, Whisper-large-v3 L29-32)
EMB_PCS = 0                 # 2026-10-19: k principal components per frozen speech encoder (src/emb_pcs.py) as direct inputs to
                            #   the families (in ALL via add_columns; appended when the ablation picks CORE) (0 = off)
KAN_FAMILY = False          # 2026-10-20: Kolmogorov-Arnold network (src/kan_family.py, pykan) as a 10th model family
IW_FAMILIES = False         # 2026-10-19: train the 9 families with covariate-shift weights (src/shift_weights.py: how test-like each
                            #   training clip is within its recording batch, from the speech PCs; no labels)
EMB_PCS_SHIFT_Z = 0         # 2026-10-19: with EMB_PCS, keep only components whose within-batch test-minus-train mean shift is
                            #   within this many standard errors in every recording batch (0 = keep all; src/emb_pcs.py)
AUDIO_QUALITY = False       # 2026-10-19 (backlog SNR idea): label-free recording-quality features per clip (kaggle/audio_quality ->
                            #   cache/audio_quality.csv: SNR, noise floor, speech level, activity, clipping, noise flatness,
                            #   bandwidth, centroid) added to the families' inputs (in ALL; appended when the ablation picks CORE)
PERSEED_DEBERTA = False     # 2026-10-17 pilot: deberta_pred / deberta_pair_pred refitted on each of the run's fold seeds
                            #   (kaggle/deberta_perseed -> cache/deberta_perseed.csv, columns <col>__s<seed>) instead of the
                            #   seed-42 OOF columns; enters per_seed, so the families and the blend both see the per-seed values
MEMBER_WEIGHTS = False      # G5(b): test-mix sample weights in member training (where the estimator takes them)
SHORT_VIEWS = None              # G4: None | "score" (predict short views of long clips OOF) | "augment" (also train the
                            #     families on them; a view always sits in its clip's fold; scored on originals only)

# Phase I: flag overrides for batch re-tests, e.g. EXP_OVERRIDES='{"RNC_HEAD": true}' python src/run.py --desc "..."
import json as _json
import os as _os
_OVR = _json.loads(_os.environ.get("EXP_OVERRIDES", "{}"))
for _k, _v in _OVR.items():
    if _k not in globals():
        raise KeyError(f"EXP_OVERRIDES: unknown flag {_k}")
    globals()[_k] = _v
if _OVR:
    print("EXP_OVERRIDES", _OVR)


def fam_oof(ctx, model, cols, folds, y_fit, Tv=None, vsrc=None, augment=False):
    """ctx.oof_test with optional short views: views of training-fold clips join the fit (augment), and views of
    validation-fold clips are predicted by the same fold model. -> (oof, test, view_oof)."""
    from sklearn.base import clone
    yf = ctx.y if y_fit is None else np.asarray(y_fit, float)
    X, Xt = ctx.T[cols].fillna(ctx.med[cols]).values, ctx.TE[cols].fillna(ctx.med[cols]).values
    Xv = Tv[cols].fillna(ctx.med[cols]).values if Tv is not None else None
    oof, te = np.zeros(len(yf)), np.zeros(len(Xt))
    vo = np.zeros(len(vsrc)) if Tv is not None else None
    for a, b in folds:
        Xa, ya = X[a], yf[a]
        if Tv is not None:
            in_a, in_b = np.isin(vsrc, a), np.isin(vsrc, b)
            if augment:
                Xa, ya = np.vstack([Xa, Xv[in_a]]), np.concatenate([ya, yf[vsrc[in_a]]])
        m = clone(model).fit(Xa, ya)
        oof[b] = m.predict(X[b]); te += m.predict(Xt) / len(folds)
        if Tv is not None and in_b.any():
            vo[in_b] = m.predict(Xv[in_b])
    clip = lambda v: None if v is None else np.clip(v, *CLIP)
    return clip(oof), clip(te), clip(vo)


def blend(CO, y, w, batch):
    """Blend weights. hill: the notebook's convex hill-climb (unweighted). nnls_*: scipy NNLS on sqrt(w)-scaled rows."""
    from scipy.optimize import nnls
    from common import hill_climb
    if BLENDER == "hill":
        return hill_climb(CO, y)[0]
    sw = np.sqrt(w)
    fit = lambda m: nnls(CO[m] * sw[m, None], y[m] * sw[m])[0]
    if BLENDER == "nnls_w":
        return fit(np.ones(len(y), bool))
    if BLENDER == "nnls_batch":
        return {"short": fit(batch == "short"), "long": fit(batch == "long")}
    if BLENDER == "nnls_batch_shrink":              # Phase I: the short weights also see the long rows at SHRINK_LAMBDA x weight
        sh, lo = batch == "short", batch == "long"
        R = np.r_[np.where(sh)[0], np.where(lo)[0]]
        ww = np.r_[w[sh], SHRINK_LAMBDA * w[lo]]
        sww = np.sqrt(ww)
        return {"short": nnls(CO[R] * sww[:, None], y[R] * sww)[0], "long": fit(lo)}
    if BLENDER == "nnls_batch3":                    # Phase I: the exact-45.06 s recordings get their own weights
        return {k: fit(batch == k) for k in np.unique(batch)}
    raise ValueError(BLENDER)


def denoised(ctx, lam):
    """(1 - lam) * y + lam * mean label of the clip's speaker group. Groups never straddle a grouped split,
    so a training clip's group mean uses training labels only."""
    g = pd.Series(ctx.y).groupby(ctx.groups).transform("mean").values
    return (1 - lam) * ctx.y + lam * g


def add_columns(ctx, table, cols):
    """Append per-clip columns from a (split, filename)-indexed table (nothing fitted)."""
    for split, tab in (("train", ctx.T), ("test", ctx.TE)):
        vals = table.loc[[(split, f) for f in tab.filename], cols].values
        for i, c in enumerate(cols):
            tab[c] = vals[:, i]
    for c in cols:
        ctx.med[c] = ctx.T[c].median()
        if c not in ctx.FEATS:
            ctx.FEATS.append(c)


def resid_features(ctx):
    """Qwen3.5-9B judge numbers for the residual correction: round 1 (expected grade + log-odds, 6 question x transcript
    pairs) and round 2 (expected grade, 9 pairs). -> (X train, X test)."""
    import qwen35_feats
    J1 = qwen35_feats.judge_table(ctx.cache)
    P2 = pd.read_csv(ctx.cache / "qwen35" / "qwen35_judge2.csv").set_index(["split", "filename"])
    J2 = pd.DataFrame(index=P2.index)
    for c in sorted({c.rsplit("_p", 1)[0] for c in P2.columns if "_p" in c and not c.endswith("_top")}):
        nd = 6 if c.endswith("cefr") else 5
        P = P2[[f"{c}_p{i}" for i in range(1, nd + 1)]].values
        J2[f"j2_{c}_ev"] = (P / P.sum(1, keepdims=True)) @ np.arange(1, nd + 1)
    J = J1.join(J2)
    if EXAMINER != "old":
        P4 = pd.read_csv(ctx.cache / "qwen35" / "qwen35_judge4.csv").set_index(["split", "filename"]).loc[J.index]
        P1 = pd.read_csv(ctx.cache / "qwen35" / "qwen35_judge.csv").set_index(["split", "filename"]).loc[J.index]
        def ev(D, c):
            M = D[[f"{c}_p{i}" for i in range(1, 6)]].values
            return (M / M.sum(1, keepdims=True)) @ np.arange(1, 6)
        paras = ("para_bands", "para_reversed", "para_raters", "para_only", "para_short")
        if EXAMINER in ("sc", "sc_fs"):
            J = J.drop(columns=[f"judge35_{v}_holistic_{k}" for v in ("whisper", "verbatim") for k in ("ev", "lo")])
            for v in ("whisper", "verbatim"):
                E = np.c_[ev(P1, f"{v}_holistic"), np.column_stack([ev(P4, f"{v}_{q}") for q in paras])]
                if EXAMINER == "sc_fs" and v == "whisper":
                    E = np.c_[E, ev(P4, "whisper_fewshot")]
                J[f"sc_{v}_ev"], J[f"sc_{v}_sd"] = E.mean(1), E.std(1)
        elif EXAMINER == "fs":
            J = J.drop(columns=["judge35_whisper_holistic_ev", "judge35_whisper_holistic_lo"])
            J["fs_whisper_ev"] = ev(P4, "whisper_fewshot")
        else:
            raise ValueError(EXAMINER)
    if RESID_PROFILE:
        P3 = pd.read_csv(ctx.cache / "qwen35" / "qwen35_judge3.csv").set_index(["split", "filename"])
        F = pd.read_csv(ctx.cache / "features.csv").set_index(["split", "filename"])
        nw = F.loc[P3.index, "n_words"].clip(lower=10).values
        for c in sorted({c.rsplit("_q", 1)[0] for c in P3.columns if "_q" in c}):
            P = P3[[f"{c}_q{k}" for k in range(6)]].values
            ev = (P / P.sum(1, keepdims=True)) @ np.arange(6)
            P3[f"j3_{c}_ev"], P3[f"j3_{c}_rate"] = ev, 100 * ev / nw
        J = J.join(P3[[c for c in P3.columns if c.startswith("j3_")]])
    Xtr = J.loc[[("train", f) for f in ctx.T.filename]].values
    Xte = J.loc[[("test", f) for f in ctx.TE.filename]].values
    if RESID_B45:
        D = pd.read_csv(ctx.cache / "speaker_groups.csv").set_index(["split", "filename"]).duration
        ind = lambda split, tab: (np.abs(D.loc[[(split, f) for f in tab.filename]].values - 45.06) < 0.005).astype(float)[:, None]
        Xtr, Xte = np.c_[Xtr, Xtr * ind("train", ctx.T)], np.c_[Xte, Xte * ind("test", ctx.TE)]
    return Xtr, Xte


def judge_table(cache) -> pd.DataFrame:
    """Zero-shot Qwen2.5-7B rubric judge on three transcripts -> expected grade and log-odds of grade >= 4 per view."""
    J = pd.read_csv(cache / "qwen" / "qwen_judge.csv").set_index(["split", "filename"])
    V = pd.read_csv(cache / "verbatim" / "qwen_judge_verbatim.csv").set_index(["split", "filename"])
    out = pd.DataFrame(index=J.index)
    for view, D in (("ctc", J), ("whisper", J), ("verbatim", V)):
        P = D.loc[out.index, [f"{view}_p{i}" for i in range(1, 6)]].values
        P = P / P.sum(1, keepdims=True)
        out[f"judge_{view}_ev"] = P @ np.arange(1, 6)
        out[f"judge_{view}_lo"] = np.log(P[:, 3:].sum(1) + 1e-9) - np.log(P[:, :3].sum(1) + 1e-9)
    return out


def add_rubric_layer(ctx):
    """Append the rubric-layer columns from cache/rubric_feats.csv (per row, nothing fitted)."""
    from rubric_feats import features
    R = features().set_index(["split", "filename"])
    for split, tab in (("train", ctx.T), ("test", ctx.TE)):
        vals = R.loc[[(split, f) for f in tab.filename], RUBRIC_LAYER].values
        for i, c in enumerate(RUBRIC_LAYER):
            tab[c] = vals[:, i]
    for c in RUBRIC_LAYER:
        ctx.med[c] = ctx.T[c].median()
        if c not in ctx.FEATS:
            ctx.FEATS.append(c)


def lgbm(seed=None):
    kw = {} if seed is None else {"random_state": seed}
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=7,
                             min_child_samples=20, colsample_bytree=0.8,
                             reg_lambda=2.0, verbose=-1, **kw)


def zoo(seed):
    return {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
        "krr":   make_pipeline(StandardScaler(), KernelRidge(alpha=1.0, kernel="rbf", gamma=0.05)),
        "svr":   make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.1)),
        "knn":   make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=25, weights="distance")),
        "rf":    RandomForestRegressor(n_estimators=500, min_samples_leaf=4, max_features=0.6,
                                       random_state=seed, n_jobs=-1),
        "et":    ExtraTreesRegressor(n_estimators=500, min_samples_leaf=4, max_features=0.7,
                                     random_state=seed, n_jobs=-1),
        "mlp":   make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2,
                                                          max_iter=1500, random_state=seed,
                                                          early_stopping=True)),
        "lgbm":  lgbm(seed),
        "xgb":   xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=3,
                                  min_child_weight=8, subsample=0.8, colsample_bytree=0.7,
                                  reg_lambda=2.0, random_state=seed, verbosity=0),
    } | ({"kan": __import__("kan_family").KANRegressor(seed=seed)} if KAN_FAMILY else {})


def make_candidates(ctx):
    ctx.batch_norm_audio = BATCH_NORM_AUDIO
    ctx.leace_text = LEACE_TEXT
    if BLENDER == "nnls_batch3":                    # blend groups: "b45" (|duration - 45.06| < 5 ms), else the batch
        Gd = pd.read_csv(ctx.cache / "speaker_groups.csv").set_index(["split", "filename"]).duration
        for attr, split, tab, bt in (("blend_batch", "train", ctx.T, ctx.batch), ("blend_batch_test", "test", ctx.TE, ctx.batch_test)):
            d = Gd.loc[[(split, f) for f in tab.filename]].values
            setattr(ctx, attr, np.where(np.abs(d - 45.06) < 0.005, "b45", bt))
        print("blend groups train", dict(zip(*np.unique(ctx.blend_batch, return_counts=True))),
              "test", dict(zip(*np.unique(ctx.blend_batch_test, return_counts=True))))
    if RUBRIC_LAYER:
        add_rubric_layer(ctx)
    per_seed, replaced = {}, set()
    y_fit = denoised(ctx, DENOISE_LAMBDA) if DENOISE_LAMBDA > 0 else None
    ctx.member_w = ctx.w if MEMBER_WEIGHTS else None
    ctx.y_fit = ctx.y if y_fit is None else y_fit
    if AUDIO_HEADS:
        import audio_heads
        per_seed = audio_heads.build(ctx)
        if AUDIO_HEADS_ONLY is not None:
            per_seed = {c: v for c, v in per_seed.items() if c in AUDIO_HEADS_ONLY}
        replaced = set(audio_heads.REPLACES)
    if LEAK_FREE:
        import leakfree
        lf_seed, lf_fixed, lf_drop = leakfree.build(ctx, DROP_FT_WAVLM)
        per_seed.update(lf_seed)
        for c, (tr, te) in lf_fixed.items():
            ctx.T[c], ctx.TE[c] = tr, te
        replaced |= lf_drop
    if RNC_HEAD:
        import rnc
        per_seed["rnc_pred"] = rnc.oof(ctx)
    if VOX_ATTN:
        import vox_attn
        per_seed["vox_attn_pred"] = vox_attn.oof(ctx)
    if CROP_HEADS:
        import crop30
        import leakfree
        for c, (X, Xt, model) in crop30.member_inputs(ctx).items():
            per_seed[c] = leakfree._oof(X, Xt, ctx.y_fit, model, ctx)
    if QWEN_HEADS:
        import qwen_feats
        per_seed.update(qwen_feats.heads(ctx, ctx.y_fit if QWEN_DENOISED else None))
    if LAYA_HEADS:
        import laya_feats
        per_seed.update(laya_feats.heads(ctx, ctx.y_fit))
    if SVR_HEADS:
        import leakfree
        import svr_heads
        for c, (X, Xt, model) in svr_heads.member_inputs(ctx).items():
            per_seed[c] = leakfree._oof(X, Xt, ctx.y_fit, model, ctx)
    if VOXTRAL:
        import leakfree
        import voxtral_feats
        for c, (X, Xt, model) in voxtral_feats.member_inputs(ctx, VOXTRAL).items():
            per_seed[c] = leakfree._oof(X, Xt, ctx.y_fit, model, ctx)
    if VOXTRAL3:
        import leakfree
        import voxtral_feats
        v3 = [leakfree._oof(X, Xt, ctx.y_fit, m, ctx)
              for X, Xt, m in voxtral_feats.member_inputs(ctx, ("voxtral_pred", "voxtral_b_pred", "voxtral_c_pred")).values()]
        per_seed["voxtral3_pred"] = ([np.mean([v[0][i] for v in v3], 0) for i in range(len(ctx.seeds))],
                                     [np.mean([v[1][i] for v in v3], 0) for i in range(len(ctx.seeds))])
    if VERBATIM_QWEN:
        import verbatim_feats
        per_seed.update(verbatim_feats.qwen_heads(ctx))
    if LAYA_DH:
        L = pd.read_csv(ctx.cache / "laya" / "laya_dh.csv").set_index(["split", "filename"])
        add_columns(ctx, L, list(LAYA_DH))
        for c in LAYA_DH:
            if c not in ctx.FEATS:
                ctx.FEATS.append(c)
            ctx.med[c] = 3.5
    if EMB_PCS:
        import emb_pcs
        Pt = emb_pcs.table(ctx.cache, EMB_PCS, EMB_PCS_SHIFT_Z)
        add_columns(ctx, Pt, list(Pt.columns))
        ctx.pc_cols = list(Pt.columns)
        print(f"EMB_PCS: {len(ctx.pc_cols)} components ({EMB_PCS} per encoder)")
    if AUDIO_QUALITY:
        A = pd.read_csv(ctx.cache / "audio_quality.csv").set_index(["split", "filename"])
        aq = [c for c in A.columns if c.startswith("aq_") and A[c].std() > 0]
        add_columns(ctx, A, aq)
        ctx.aq_cols = aq
        print(f"AUDIO_QUALITY: {aq}")
    if LAYA_CHECKLIST:
        L = pd.read_csv(ctx.cache / "laya" / "laya_checklist_feats.csv").set_index(["split", "filename"])
        add_columns(ctx, L, list(LAYA_CHECKLIST))
        for c in LAYA_CHECKLIST:
            ctx.med[c] = ctx.T[c].median()
            if c not in ctx.FEATS:
                ctx.FEATS.append(c)
    if DEBERTA_LARGE:
        L = pd.read_csv(ctx.cache / "deberta_large" / "deberta_large_pair.csv").set_index(["split", "filename"])
        add_columns(ctx, L, ["deberta_large_pair_pred"])
        ctx.med["deberta_large_pair_pred"] = 3.5
        if DEBERTA_LARGE == "replace":
            replaced.add("deberta_pair_pred")
    if DEBERTA_LARGE_V:
        L = pd.read_csv(ctx.cache / "deberta_large" / "deberta_large_vpair.csv").set_index(["split", "filename"])
        add_columns(ctx, L, ["deberta_large_vpair_pred"])
        ctx.med["deberta_large_vpair_pred"] = 3.5
        if DEBERTA_LARGE_V == "replace":
            replaced.add("deberta_vpair_pred")
    if VERBATIM_DEBERTA:
        import verbatim_feats
        add_columns(ctx, verbatim_feats.deberta_table(ctx), ["deberta_vpair_pred"])
        if "deberta_vpair_pred" not in ctx.FEATS:
            ctx.FEATS.append("deberta_vpair_pred")
        ctx.med["deberta_vpair_pred"] = 3.5
    if LAYA_FT:
        L = pd.read_csv(ctx.cache / "laya" / "laya_ft_pair.csv").set_index(["split", "filename"])
        add_columns(ctx, L, ["laya_ft_pair_pred"])
        if "laya_ft_pair_pred" not in ctx.FEATS:
            ctx.FEATS.append("laya_ft_pair_pred")
        ctx.med["laya_ft_pair_pred"] = 3.5
    if JUDGE_PLUS or JUDGE_MEMBER:
        Jt = judge_table(ctx.cache)
        if JUDGE_PLUS:
            add_columns(ctx, Jt, [c for c in Jt.columns if c.endswith("_lo")])
        if JUDGE_MEMBER:
            import leakfree
            X = Jt.loc[[("train", f) for f in ctx.T.filename]].values
            Xt = Jt.loc[[("test", f) for f in ctx.TE.filename]].values
            mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
            per_seed["judge_pred"] = leakfree._oof(X, Xt, ctx.y_fit, mdl, ctx)
            ctx.judge_inputs = (X, Xt, mdl)
    if QWEN35_HEADS:
        import qwen35_feats
        per_seed.update({c: v for c, v in qwen35_feats.heads(ctx).items() if c in QWEN35_HEADS})
    if QWEN35_JUDGE:
        import leakfree
        import qwen35_feats
        X, Xt, mdl = qwen35_feats.judge_inputs(ctx)
        per_seed["qwen35_judge_pred"] = leakfree._oof(X, Xt, ctx.y_fit, mdl, ctx)
    if PERSEED_DEBERTA:
        D = pd.read_csv(ctx.cache / "deberta_perseed.csv").set_index(["split", "filename"])
        for c in ("deberta_pred", "deberta_pair_pred"):
            if c in replaced:
                continue
            cols = [f"{c}__s{s}" for s in ctx.seeds]
            missing = [k for k in cols if k not in D.columns]
            if missing:
                raise KeyError(f"PERSEED_DEBERTA: {missing} not in cache/deberta_perseed.csv")
            per_seed[c] = ([D.loc[[("train", f) for f in ctx.T.filename], k].values for k in cols],
                           [D.loc[[("test", f) for f in ctx.TE.filename], k].values for k in cols])
            print(f"PERSEED_DEBERTA: {c} from {cols}")
    if QWEN_RUBRIC or QWEN_JUDGE:
        import qwen_feats
        Q = qwen_feats.features(ctx.cache, set(ctx.T.filename))
        add_columns(ctx, Q, (qwen_feats.RUBRIC_COLS if QWEN_RUBRIC else []) + (qwen_feats.JUDGE_COLS if QWEN_JUDGE else []))
    for c in per_seed:
        ctx.med[c] = 3.5
        if c not in ctx.FEATS:
            ctx.FEATS.append(c)

    def use_seed(i):
        for c, (o, t) in per_seed.items():
            ctx.T[c], ctx.TE[c] = np.clip(o[i], *CLIP), np.clip(t[i], *CLIP)

    use_seed(0)
    if CTC_FAIL_NEUTRAL:
        for tab in (ctx.T, ctx.TE):
            fail = tab.repair_wer_nofiller > CTC_FAIL_CUT
            tab.loc[fail, REPAIR_COLS] = np.nan
    feats = [c for c in ctx.FEATS if not (DROP_DURATION and c == "duration") and c not in replaced
             and not (DROP_AMOUNT and c in AMOUNT_COLS)]
    core = [c for c in CORE if c in feats]

    # 1. ablation on the first fold seed: does the explainable core match the full set?
    f0 = ctx.folds[ctx.seeds[0]]
    r_all, _ = ctx.oof_test(lgbm(), feats, f0)
    if len(core) == len(CORE):
        r_core, _ = ctx.oof_test(lgbm(), core, f0)
        gap = rmse(ctx.y, r_core) - rmse(ctx.y, r_all)
        use = core if gap <= 0.02 else feats
        print(f"ablation: all {len(feats)} -> {rmse(ctx.y, r_all):.4f}   core {len(core)} -> "
              f"{rmse(ctx.y, r_core):.4f}   gap {gap:+.4f} -> using {'CORE' if use is core else 'ALL'}")
    else:
        use = feats
        print(f"ablation skipped (core columns missing) -> using ALL {len(feats)}")
    if AUDIO_QUALITY:                               # CORE runs see them too (ALL has them via add_columns)
        use = list(use) + [c for c in ctx.aq_cols if c not in use]
    if EMB_PCS:                                     # CORE runs see them too (ALL has them via add_columns)
        use = list(use) + [c for c in ctx.pc_cols if c not in use]

    # 2. diverse model families, one fit per fold seed
    fam_w = ctx.member_w
    if IW_FAMILIES:
        import shift_weights
        iw = shift_weights.ratios(ctx.cache, ctx.T.filename)
        fam_w = iw if fam_w is None else fam_w * iw
        print(f"IW_FAMILIES: weights min {iw.min():.2f} max {iw.max():.2f}, ESS {shift_weights.ess(iw):.0f} of {len(iw)}")
    V = vsrc = None
    if SHORT_VIEWS:
        from shortview import RECOMPUTED, build_views
        V, vsrc, _ = build_views(ctx)

    def view_frame():
        Tv = ctx.T.iloc[vsrc].reset_index(drop=True).copy()
        for c in RECOMPUTED:
            if c in Tv.columns:
                Tv[c] = V[c].values
        return Tv

    cands = {fam: ([], []) for fam in zoo(0)}
    view_cands = {fam: [] for fam in zoo(0)}
    for i, s in enumerate(ctx.seeds):
        use_seed(i)
        Tv = view_frame() if SHORT_VIEWS else None
        for fam in zoo(0):
            if SHORT_VIEWS:
                o, t, vo = fam_oof(ctx, zoo(s)[fam], use, ctx.folds[s], y_fit, Tv, vsrc, SHORT_VIEWS == "augment")
                view_cands[fam].append(vo)
            else:
                o, t = ctx.oof_test(zoo(s)[fam], use, ctx.folds[s], y_fit=y_fit, sample_weight=fam_w)
            cands[fam][0].append(o); cands[fam][1].append(t)

    # 3. learned columns (OOF on the seed-42 folds upstream) enter directly
    for c in [c for c in ctx.T.columns if c.endswith("_pred") and c not in replaced]:
        if c in per_seed:
            cands[c] = ([np.clip(o, *CLIP) for o in per_seed[c][0]], [np.clip(t, *CLIP) for t in per_seed[c][1]])
            continue
        o = np.clip(ctx.T[c].values, *CLIP)
        t = np.clip(ctx.TE[c].values, *CLIP)
        cands[c] = ([o] * len(ctx.seeds), [t] * len(ctx.seeds))
    if SHORT_VIEWS:                               # learned columns can't be recomputed on a cut: views keep the clip's value
        for c in cands:
            if c not in view_cands:
                view_cands[c] = [np.asarray(o)[vsrc] for o in cands[c][0]]
        ctx.view_cands, ctx.view_src = {c: np.mean(v, 0) for c, v in view_cands.items()}, vsrc
        orig = ctx.T.iloc[vsrc][[c for c in RECOMPUTED if c in ctx.T.columns]].reset_index(drop=True)
        ctx.view_shift = (V[orig.columns] - orig).mean().round(4)
    if TRAIN_FIT:      # cheap members refit on all rows; families see the members' seed-averaged OOF (as they were trained)
        import leakfree
        import qwen_feats
        from common import fit_weighted
        from sklearn.base import clone
        allr = np.arange(len(ctx.y))
        fitted = {}
        if LEAK_FREE:
            for c, (X, Xt, model) in leakfree.member_inputs(ctx).items():
                fitted[c] = clone(model).fit(X, ctx.y_fit).predict(X)
        if QWEN_HEADS:
            for c, (X, Xt, model) in qwen_feats.member_inputs(ctx).items():
                fitted[c] = fit_weighted(clone(model), X, ctx.y_fit if QWEN_DENOISED else ctx.y, None,  # as heads()
                                         groups=ctx.groups).predict(X)
        if LAYA_HEADS:
            import laya_feats
            for c, (X, Xt, model) in laya_feats.member_inputs(ctx).items():
                fitted[c] = fit_weighted(clone(model), X, ctx.y_fit, None, groups=ctx.groups).predict(X)
        if SVR_HEADS:
            import svr_heads
            for c, (X, Xt, model) in svr_heads.member_inputs(ctx).items():
                fitted[c] = clone(model).fit(X, ctx.y_fit).predict(X)
        if VOXTRAL:
            import voxtral_feats
            for c, (X, Xt, model) in voxtral_feats.member_inputs(ctx, VOXTRAL).items():
                fitted[c] = clone(model).fit(X, ctx.y_fit).predict(X)
        if VERBATIM_QWEN:
            import verbatim_feats
            for c, (X, Xt, model) in verbatim_feats.qwen_inputs(ctx).items():
                fitted[c] = fit_weighted(clone(model), X, ctx.y, None, groups=ctx.groups).predict(X)
        for c in per_seed:
            ctx.T[c] = np.clip(np.mean(per_seed[c][0], 0), *CLIP)
        tf = {fam: ctx.oof_test(zoo(ctx.seeds[0])[fam], use, [(allr, allr)], y_fit=y_fit)[0] for fam in zoo(0)}
        for c in cands:
            if c not in tf:
                tf[c] = np.clip(fitted[c], *CLIP) if c in fitted else np.mean(cands[c][0], 0)
        ctx.train_fit, ctx.train_fit_oofcols = tf, sorted(c for c in cands if c not in fitted and c not in zoo(0))
    if TOPIC_OOF:                                 # G11: topic-held-out OOF; DeBERTa / ft_wavlm keep their own OOF
        import leakfree
        import qwen_feats
        from topic_cv import topic_folds, topic_oof
        tf, _ = topic_folds(ctx)
        tcols = {}
        if LEAK_FREE:
            for c, (X, Xt, model) in leakfree.member_inputs(ctx).items():
                tcols[c] = topic_oof(model, X, ctx.y_fit, tf, sw=ctx.member_w)
        if RNC_HEAD:
            import rnc
            tcols["rnc_pred"] = rnc.topic_oof(ctx, tf)
        if VOX_ATTN:
            import vox_attn
            tcols["vox_attn_pred"] = vox_attn.topic_oof(ctx, tf)
        if VOXTRAL3:
            import voxtral_feats
            tcols["voxtral3_pred"] = np.mean([topic_oof(m, X, ctx.y_fit, tf, sw=ctx.member_w) for X, Xt, m in
                                              voxtral_feats.member_inputs(ctx, ("voxtral_pred", "voxtral_b_pred", "voxtral_c_pred")).values()], 0)
        if QWEN35_HEADS:
            import qwen35_feats
            tcols.update({c: v for c, v in qwen35_feats.topic_heads(ctx, tf).items() if c in QWEN35_HEADS})
        if QWEN35_JUDGE:
            import qwen35_feats
            X, Xt, mdl = qwen35_feats.judge_inputs(ctx)
            tcols["qwen35_judge_pred"] = topic_oof(mdl, X, ctx.y_fit, tf, sw=ctx.member_w)
        if JUDGE_MEMBER:
            X, Xt, mdl = ctx.judge_inputs
            tcols["judge_pred"] = topic_oof(mdl, X, ctx.y_fit, tf, sw=ctx.member_w)
        if CROP_HEADS:
            import crop30
            for c, (X, Xt, model) in crop30.member_inputs(ctx).items():
                tcols[c] = topic_oof(model, X, ctx.y_fit, tf, sw=ctx.member_w)
        if QWEN_HEADS:
            tcols.update(qwen_feats.topic_heads(ctx, tf, ctx.y_fit))
        if LAYA_HEADS:
            import laya_feats
            tcols.update(laya_feats.topic_heads(ctx, tf, ctx.y_fit))
        if SVR_HEADS:
            import svr_heads
            for c, (X, Xt, model) in svr_heads.member_inputs(ctx).items():
                tcols[c] = topic_oof(model, X, ctx.y_fit, tf, sw=ctx.member_w)
        if VOXTRAL:
            import voxtral_feats
            for c, (X, Xt, model) in voxtral_feats.member_inputs(ctx, VOXTRAL).items():
                tcols[c] = topic_oof(model, X, ctx.y_fit, tf, sw=ctx.member_w)
        if VERBATIM_QWEN:
            import verbatim_feats
            for c, (X, Xt, model) in verbatim_feats.qwen_inputs(ctx).items():
                tcols[c] = topic_oof(model, X, ctx.y, tf, ctx.groups)
        for c, o in tcols.items():
            ctx.T[c] = np.clip(o, *CLIP)              # families see topic-held-out member values
        if SHORT_VIEWS == "augment":
            Tv = view_frame()
            topic = {fam: fam_oof(ctx, zoo(ctx.seeds[0])[fam], use, tf, y_fit, Tv, vsrc, True)[0] for fam in zoo(0)}
        else:
            topic = {fam: ctx.oof_test(zoo(ctx.seeds[0])[fam], use, tf, y_fit=y_fit, sample_weight=fam_w)[0]
                     for fam in zoo(0)}
        for c in cands:
            if c not in topic:
                topic[c] = np.clip(tcols[c], *CLIP) if c in tcols else np.mean(cands[c][0], 0)
        ctx.topic_cands, ctx.topic_folds, ctx.topic_refit = topic, tf, sorted(set(topic) - set(cands) | set(tcols) | set(zoo(0)))
    return cands
