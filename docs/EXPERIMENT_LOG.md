# Experiment log (lab notebook)

> Public copy of the project's running log, newest at the bottom. Run ids are `MMDD-HHMMSS`. Lines that quoted a transcript or a single clip's
> prediction were removed; everything else is as written at the time, including the mistakes and their corrections.

One entry per experiment, newest at the bottom.

<!-- template
## <run_id> — <change>
- hypothesis: why it should lower CV
- CV: 0.xxxx (Δ vs reference ±0.xxxx), per seed …
- verdict: accepted / rejected / screened out
- lesson: one line
-->

## 1011-154032 — baseline: notebook §9 reproduced (full, seeds 42 1337 2024)
- cache: amritarajput54/notebook26509aa315 **v18** (public 0.3770). v19, the latest run, failed on Kaggle
  (CUDA driver error) and has no outputs, so the version is pinned.
- CV: 0.4750, per seed 0.4818 / 0.4740 / 0.4740, spread 0.0078. Notebook v18 §9 "OOF CV RMSE (honest)": 0.4748 (Δ +0.0002).
- vs notebook_submission.csv: mean |diff| 0.0063, max 0.0269, corr 1.0000.
- blend: xgb 0.367, et 0.275, ridge 0.211, ft_wavlm_pred 0.056, svr 0.056, whisper_med_pred 0.019, deberta_pred 0.017.
- verdict: baseline (reproduction check passed).
- lesson: the local loop matches the notebook. Public/CV ≈ 0.794 for v18.

## 1011-154059 — baseline fast (seed 42)
- CV: 0.4805. Reference for --fast screens.
- verdict: baseline.

## 1011-154802 — DROP_DURATION = True (backlog #1, tested)
- hypothesis: duration is the main train/test shift (KS 0.45, adversarial AUC 0.746 → 0.696 without it;
  median clip 60 s train vs 45 s test). Rate/count features already carry its signal, so dropping it
  should cut length-driven splits that don't transfer.
- CV: 0.4776 (Δ −0.0026 vs 1011-154032), per seed 0.4838 / 0.4753 / 0.4776, worse on 0/3 seeds.
  Fast screen 1011-154716: 0.4819 (Δ −0.0014).
- verdict: rejected (reverted).
- lesson: duration carries real label signal in train (xgb +0.003, lgbm +0.004 without it). CV can't tell
  whether that transfers to shorter test clips. By the CV rule it stays.

## 1011-155150 — repair_wer_nofiller with numbers/symbols spelled out (backlog #2, tested)
- hypothesis: the notebook tokeniser `[a-z']+` drops digits, so Whisper's "2019"/"10th"/"50%" vanish while
  CTC's spelled-out words stay, and every number counts as edits. Spelling both sides out (num2words,
  ordinals, years, %, $, ok→okay) should sharpen the strongest handcrafted feature.
- check: the rebuild reproduces the notebook column exactly (max |diff| 2e-16). Spelling changed 80/948 rows
  (mean −0.013). Spearman vs label −0.5528 → −0.5537.
- CV: 0.4747 (Δ +0.0003 vs 1011-154032), per seed 0.4815 / 0.4737 / 0.4737, better on 3/3 seeds.
  Fast screen 1011-155055: 0.4803 (Δ +0.0002).
- verdict: rejected (below MIN_GAIN, reverted).
- lesson: real but tiny. Only 8% of rows contain numbers. Number formatting isn't what limits repair_wer.

## 1011-155511 — edit operations by type, Whisper vs CTC (backlog #4, tested)
- hypothesis: repair_wer lumps grammar repairs together with misrecognitions. Splitting the alignment
  (CTC → Whisper, numbers spelled out) into sub_rate / ins_rate / del_rate plus the mean character similarity
  of substituted pairs (pronunciation near-miss vs different word) gives the trees cleaner grammar signal.
- check: sub+ins+del correlates 0.989 with repair_wer. Spearman vs label: sub −0.52, ins −0.30, del −0.36,
  sub_char_sim +0.40. Partial vs repair_wer: del −0.163, the others ≤ 0.074.
- CV: 0.4751 (Δ −0.0000 vs 1011-154032), per seed 0.4815 / 0.4737 / 0.4744, better on 2/3 seeds.
  Fast screen 1011-155402: 0.4802 (Δ +0.0003). ET improved 0.4805 → 0.4801. The blend didn't.
- verdict: rejected (reverted). Character-level edit distance (the other half of #4) was not tried separately.
- lesson: the trees already get what the Whisper-vs-CTC comparison offers. del_rate's partial signal
  doesn't survive a blend of 9 families + 7 speech/text model columns.

**Stopped after 3 rejections in a row** (DROP_DURATION, spelled numbers, edit ops). All three were
feature-level refinements of the handcrafted text table, which only carries ~11% of importance
(lexical 8% + ASR repair 3%). The next local tries should be model-level (#6 CatBoost, #8 heads on the
speech embeddings, which carry 59%), not more handcrafted text features.

## 1011-160324 — speech_krr: one nonlinear head over the 3 frozen speech encoders (backlog #8, tested)
- hypothesis: the notebook fits a linear ridge per encoder. One head over wavlm + wavlm_large + whisper_med
  (notebook pooling: per-layer z-score, uniform layer mean; equal weight per encoder) → PCA 128 → RBF kernel
  ridge (alpha/gamma by inner 3-fold CV in each outer fold) could pick up cross-encoder and nonlinear
  structure. Entered as a blend candidate only (as a tree feature it would leak through the stacking).
- check (seed-42 folds): standalone 0.5562, worse than the per-encoder ridges (0.529–0.534), corr 0.94 with
  them. The inner CV picked the smallest gamma in every fold, so the data prefers ~linear. Grid not changed
  after seeing this.
- CV: 0.4744 (Δ +0.0006 vs 1011-154032), per seed 0.4814 / 0.4730 / 0.4737, better on 3/3 seeds.
  Blend weight 0.082. Fast screen 1011-160156: 0.4803 (Δ +0.0002).
- verdict: rejected (below MIN_GAIN, reverted). Layer selection (the other half of #8) not tried: picking
  a layer by score is the selection the notebook deliberately avoids.
- lesson: frozen-encoder information is already in the blend via the three ridge columns. A different head
  on the same embeddings re-expresses it (corr 0.94) rather than adding to it.

## 1011-161551 — CatBoost as a tenth family (backlog #6, tested; "light tuning" half not tried)
- hypothesis: tree families hold ~60% of the blend. CatBoost's ordered boosting and symmetric trees overfit
  less on 732 rows, so it should be strong and different enough from xgb/lgbm to earn weight.
  Fixed config, untuned: depth 4, lr 0.03, 800 iterations, l2_leaf_reg 3.
- CV: 0.4748 (Δ +0.0002 vs 1011-154032), per seed 0.4814 / 0.4743 / 0.4733, better on 2/3 seeds.
  cat alone 0.4821 (2nd best family). Blend weight 0.208, taken mostly from xgb (0.367 → 0.201).
  Fast screen 1011-161359: 0.4803 (Δ +0.0002).
- verdict: rejected (below MIN_GAIN, reverted).
- lesson: another boosted-tree family replaces xgb rather than adding to it. Tree diversity is saturated.

### Duration check (requested; nothing changed)
Baseline 1011-154032 vs no-duration 1011-154802, test predictions (216 rows):
- mean prediction: baseline 3.2556, no-duration 3.2772 (+0.0216). |diff| mean 0.0311, max 0.1113 (1.4% > 0.1).
- corr(diff, test duration): pearson −0.556, spearman −0.547. Dropping duration raises short clips
  (≤45.5 s: 142 rows, +0.036) and slightly lowers long ones (≥59.5 s: 58 rows, −0.010).
- baseline test mean 3.2556 vs train label mean 3.4809 (gap −0.225). Train labels by duration band: ≤45.5 s
  3.080 (n 163), 45.5–59.5 s 3.735 (n 51), ≥59.5 s 3.582 (n 518). Test is 66% short clips vs 22% in train.
  Reweighting the train band means by the test mix gives 3.263, so the baseline's lower test mean is what
  the duration mix predicts. It isn't a calibration drift. CV can't tell whether short test clips follow the
  same duration–label pattern as short train clips.

## 1011-161744 — Huber objective (delta 1.0) for lgbm + xgb (backlog #7, tested; quantile not tried)
- hypothesis: labels at the extremes are noisy (four below 2.0, coarse 5.0s). Squared-error boosting chases
  them. Huber with delta 1.0 (two grade steps) caps their pull and should cut variance.
- CV: 0.4758 (Δ −0.0008 vs 1011-154032), per seed 0.4831 / 0.4746 / 0.4744, worse on 0/3 seeds.
  xgb alone 0.4845 → 0.4909, lgbm 0.4924 → 0.4936. Fast screen 1011-161655: 0.4815 (Δ −0.0010).
- verdict: rejected (reverted).
- lesson: the metric is RMSE, so the extreme labels are what it rewards getting right. Down-weighting them
  costs more than the variance it saves.

**Session stop: 6 local changes in a row rejected** (#1, #2, #4, #8, #6, #7). The best of them gained
+0.0006. Both feature-level and model-level changes are saturated around 0.475 on the current inputs.
Next: new information from the notebook side (DeBERTa on the raw CTC text, proposed via /gpu-run).

## GPU job v20 — DeBERTa on the raw CTC transcript and on the (CTC raw, Whisper clean) pair (pushed 2026-10-11)
- hypothesis: deberta_pred reads Whisper's text, which has already had its grammar repaired (alone 0.638,
  blend weight 0.017). The raw CTC text keeps the speaker's errors. New columns deberta_ctc_pred and
  deberta_pair_pred, same folds (seed 42), 6 epochs, seeds 42/7/2024.
- setup: v18 caches uploaded as private dataset amritarajput54/shl-grammar-caches-v18 and attached. v18
  had restored no caches and ran 7.0 h from scratch. Metadata switched from the TPU machine (why v19 failed)
  to GPU (T4, 4 h cap). From the v19 source: whisper_large/hubert_large off, FT_RUNS back to v18's (cached),
  a guard cell that stops without caches/CUDA. Estimate ~1.5–2 h.
- ran as Kaggle **version 21** (the CLI said 20), T4 x2, ~68 min. The guard passed and all 9 caches restored.
  An extra interactive editor session also used some GPU and was cancelled.
- standalone (3 seeds): deberta_pred 0.6379, deberta_ctc_pred 0.6435, **deberta_pair_pred 0.5849**.
  Notebook stacked LightGBM: no DeBERTa 0.5117 → +pair 0.5045. Notebook §9 honest CV 0.4748 → 0.4719.
- the other 33 feature columns are bit-identical to v18, so the only change is the two new columns.

## 1011-190112 — gpu v21: adds deberta_ctc_pred + deberta_pair_pred
- hypothesis: DeBERTa on the raw CTC text (paired with Whisper's clean text) sees the errors that Whisper
  repairs away.
- CV: **0.4725** (Δ +0.0025 vs 1011-154032), per seed 0.4787 / 0.4715 / 0.4711, better on 3/3 seeds.
  Blend: et 0.381, xgb 0.253, ridge 0.124, ft_wavlm 0.075, svr 0.073, deberta_pair 0.045, whisper_med 0.032.
  et alone 0.4805 → 0.4761 with the new columns as features.
- verdict: **accepted** (1st accepted since the baseline; confirm on fresh seeds after 3).
- lesson: the raw CTC text alone is no better (0.6435). Paired with Whisper's text it is (0.5849): the model
  needs the clean text to tell an error from a misrecognition. New information beat 6 local refinements.
- public: **0.3707** (submitted 1011-190112 at 19:01 UTC). It matches the notebook's own v21 submission (0.3707,
  18:59), so the slot added no new information. Previous best 0.3750 (v16). Rank ~90/425, next team up
  0.3688. Public/CV = 0.785.

## 1011-190725 — drop deberta_ctc_pred (idea 3 from the v21 follow-ups)
- hypothesis: 0 blend weight, r 0.83 with deberta_pred, weakest DeBERTa column. As a tree feature it may only add noise.
- CV: 0.4728 (Δ −0.0003 vs 1011-190112), per seed 0.4787 / 0.4721 / 0.4720, better on 1/3 seeds.
  The fast screen (1011-190619) was scored against the v18 fast baseline, so its "accepted" meant nothing.
- verdict: rejected (reverted). deberta_ctc_pred stays.
- lesson: zero blend weight doesn't mean useless. The trees use it as a feature.
- follow-up: fast rebaseline on the v21 cache recorded, so screens compare like with like.

## 1011-192727 — BERTScore Whisper vs CTC as features (notebook-side idea, run on CPU)
- hypothesis: a grammar repair keeps the meaning (edits but high similarity); a misrecognition changes it.
  BERTScore P/R/F1 (roberta-base layer 10, no idf, lowercased, punctuation stripped) next to repair_wer
  separates the two.
- check: corr with repair_wer −0.95 to −0.97 (mostly redundant). Partial vs repair_wer: bs_p −0.10,
  bs_r −0.16, bs_f1 −0.12. Sign as predicted (at equal edit rate, meaning-preserving edits → lower grade).
  ~5 min on 4 CPU threads; cache in cache/bertscore_roberta-base_L10.csv.
- CV: 0.4727 (Δ −0.0001 vs 1011-190112), per seed 0.4787 / 0.4720 / 0.4711, better on 0/3 seeds.
  Fast screen 1011-192436: 0.4774 (Δ 0.0000 vs the v21 fast baseline).
- verdict: rejected (reverted). torch + transformers (CPU) stay installed in .venv; requirements.txt unchanged.
- lesson: the signal is real but small, and the trees plus repair_wer already cover it.
  The Whisper-vs-CTC comparison is exhausted (edit ops, spelled numbers, BERTScore all ≤ +0.0003).

## GPU job v23 — DeBERTa-v3-large on the (CTC raw, Whisper clean) pair
- hypothesis: the pair input was the win in v21 (0.585 alone). The large model should read it better.
- ran as Kaggle **version 23** (the CLI said 22), T4. Base runs came from cache (dataset v2 adds the v21
  DeBERTa caches). The large model ran out of memory at BS 8, so gradient checkpointing turned on; 135 min.
- result: unstable. **5/15 folds collapsed** to ~1.01 RMSE (predicting the mean). Non-collapsed folds
  0.57–0.74, no better than base. Alone (3 seeds) **0.6831** vs base pair 0.5849. Notebook stacked LightGBM
  0.5045 → 0.5030. §9 honest CV 0.4719 → 0.4716. The other 35 columns are identical to v21.

## 1011-221217 — gpu v23: adds deberta_large_pair_pred
- CV: 0.4718 (Δ +0.0007 vs 1011-190112), per seed 0.4782 / 0.4715 / 0.4699, better on 3/3 seeds. Blend weight 0.
- verdict: rejected (below MIN_GAIN). The cache is restored to v21 and the large run is commented out in
  kaggle/main, so it is not retrained.
- lesson: DeBERTa-v3-large at lr 1e-5 / 6 epochs collapses on ~585 training rows. A retry would need a
  stabiliser (layer-wise lr decay, re-initialised top layers, longer warmup), which means another ~2.5 h
  bet. Not worth it before cheaper options.
- public (v23 notebook submission, 22:09 UTC, not made by the loop): **0.3712** vs v21's 0.3707. That is within
  public noise (one clip ≈ 0.002–0.005) and matches the CV verdict: no real change. Best stays v21.

## 1012-064725 — level-2 stack: non-negative ridge on all candidate OOFs, fitted inside folds (backlog #3, tested)
- hypothesis: joint non-negative weights plus an intercept, fitted inside the folds, beat the greedy convex hill-climb.
  Entered as a candidate (stack_nnr), so the blend only uses it if it helps.
- CV: 0.4725 (Δ −0.0000 vs 1011-190112), per seed 0.4787 / 0.4715 / 0.4711. stack_nnr got 0 weight.
  Its honest in-fold OOF is 0.4792 (fast screen 1012-064605: 0.4884 vs 0.4775).
- verdict: rejected (reverted).
- lesson: the honest stack sits ~0.007 above the in-sample hill-climb. That is roughly the optimism of
  the reported CV. Runs carry the same optimism, so comparisons stay fair, but absolute CV is ~0.005 low.
  It agrees with the established fact that recalibrating the blend gains nothing.

## 1012-065053 — TF-IDF n-grams → RidgeCV as a text candidate (backlog #5, tested)
- hypothesis: sparse surface patterns (Whisper word 1–2 + char 2–5 grams, CTC char 2–5 grams), fitted
  inside folds, give a cheap text model that differs from DeBERTa.
- CV: 0.4725 (Δ −0.0000 vs 1011-190112), per seed 0.4787 / 0.4715 / 0.4711. tfidf_ridge alone 0.707, blend weight 0.
  Fast screen 1012-064843: 0.4775 (Δ 0.0000).
- verdict: rejected (reverted, module deleted).
- lesson: weaker than every DeBERTa column (0.585–0.644) and nothing they don't already carry.

**Local backlog spent:** #1–#8 all tested (untried halves: character-level edit distance, light tuning of one
family, quantile loss, choosing an embedding layer by score). 5 rejections in a row since v21.

# PLAN_RUBRIC — Phase 0: where we stand (2026-10-12)
- rank ~90 / 428 public. Best 0.3707.
- top-10 public: 0.3064, 0.3126, 0.3126, 0.3150, 0.3155, 0.3158, 0.3173, 0.3211, 0.3212, 0.3214.
- **0.3707** = ledger run **1011-190112** (v21: deberta_ctc + deberta_pair, CV 0.4725 on random folds). The v21
  notebook submission gave the same score.
- **0.3712** = v23 notebook submission, which matches ledger run **1011-221217** (v23: + deberta_large_pair,
  CV 0.4718, rejected): mean |diff| 0.005, corr 1.0000.
- 1011-190112 stays one of the two final picks unless a run beats it on the new validation.
- TEAM_NAME in config.env is still empty. Amrita has to set `TEAM_NAME=ambruhsia` (config.env is edit-protected).

# PLAN_RUBRIC — Phase A: validation that mirrors the test set
## A1 speaker groups (src/speaker_groups.py → cache/speaker_groups.csv)
- t = p99 of test→train nearest x-vector cosine = **0.988**. Average-linkage clustering at distance 1−t.
- **503 groups. 53.1% of train clips have a twin.** 160 groups of size ≥ 2 (95 pairs, 63 triples, one 4, one 6).
  Largest group 6 clips (0.8%). 3 test clips sit above t (≈1% by construction).
- twins are real: within-group |label diff| 0.236 vs 1.157 for random pairs. File IDs adjacent (median gap 2).
  96% of twin groups lie within one batch.
## A2 recording batch
- cut at the density minimum between the 45 s and 60 s modes: **47.7 s**. Short = **23.8%** of train (174),
  **68.1%** of test (147). Test-mix weights: short 2.863, long 0.419. Plots in reports/phaseA/.
## A3 folds
- data.py: StratifiedGroupKFold(5, shuffle, seed) on label (1.0→1.5) grouped by speaker. folds="random" kept for
  comparison. 0 groups shared across a split. cache/folds_grouped.csv written for seeds 42/1337/2024.
## A4 metrics + gate
- run.py: cv_grouped, cv_testmix (primary), cv_short, cv_long per seed. Gate: testmix ≥ MIN_GAIN, ≥ 2/3 seeds,
  cv_grouped not worse than +0.002. blend(CO, y[, w]). CLAUDE.md updated. Check: --folds random reproduces
  1011-190112 exactly (0.4725), and shows testmix 0.5115 (short 0.538, long 0.450) even on random folds.
- known conflict: submit.py compares the raw `cv` field across schemes, so it will refuse grouped runs (Amrita's call).
## A5 re-baseline (1012-071022 full, 1012-071052 fast)
- **cv_grouped 0.4748, cv_testmix 0.5122, cv_short 0.5375, cv_long 0.4535** (per-seed testmix 0.5117/0.5186/0.5105).
  Only +0.0023 grouped / +0.0007 testmix vs random folds, because the fixed *_pred columns still carry twin info.
- twin-gap table (reports/phaseA/twin_gap.csv): families +0.003–0.004 (mlp +0.014). Rebuilt on grouped folds:
  whisper_med +0.042, wavlm_large +0.041, wavlm +0.033, **spk_knn +0.074 (0.718 → 0.792)**.
  DeBERTa ×3, ft_wavlm and emb_pred can't be rebuilt locally.
## A6 speaker-dependent inputs (vs grouped baseline 1012-071022)
- (i) drop spk_knn_pred (1012-071412): testmix 0.5124 (−0.0002), grouped +0.0028 → **rejected**.
- (ii) *_pred blend-only (screen 1012-071449): testmix 0.5262 (−0.0142) → **screened out**.
- both are biased: the fixed *_pred columns leak twin labels, so CV favours keeping them. Diagnostics with the
  4 CPU-rebuildable columns rebuilt leak-free on grouped folds (not candidate changes; code reverted):
  - D0 rebuilt (1012-071723): testmix **0.5174** (+0.0052 vs 0.5122), grouped 0.4821 → the leak was worth ~0.005.
  - D1 D0 − spk_knn (1012-071849): 0.5171 (+0.0003, 2/3 seeds): neutral once leak-free.
  - D2 D0 blend-only (1012-072015): 0.5319 (−0.0145): the two-level stack genuinely helps.
  - D3 D0 − v21 columns (1012-072204): 0.5238 → v21's deberta columns are worth **−0.0064 testmix** (short
    0.5503 → 0.5414). v21's advantage is not mainly twins, though the DeBERTa columns' own random-fold
    OOF is still unmeasured (needs C2).
  - as leak-free columns weaken, the contaminated ft_wavlm_pred gains weight (0.13 in D0, 0.44 in D2).
## A7 zero-label guard (src/sentinel_guard.py → reports/phaseA/sentinel_guard.csv)
- audio: the competition zip was downloaded locally (stays on this machine). Read: 37 sentinels, 40 random normal
  train clips, all 216 test clips.
- sentinels are white-ish noise: spectral flatness 0.518–0.554 (normal speech ≤ 0.131), ZCR ~0.45, speech-band
  share 0.44. Threshold flatness > 0.325 (midpoint of the gap, 1.49 pooled std); a nearest-centroid rule on 4
  features agrees. Normal clips misflagged: 0.
- **test: 0 clips flagged** by either rule. The closest test clip has flatness 0.184 (speech).
  Confirms others' report: no sentinel in test.
- side finding: 2 test clips have near-empty Whisper transcripts (one is a 6.5 s clip; the other a 60 s clip where
  Whisper failed while CTC heard the speech): an ASR failure that puts its text features off-distribution. Clip details
  removed for the public copy.

# PLAN_RUBRIC — Phase B: rubric-aligned features
- src/rubric_feats.py → cache/rubric_feats.csv (signature guard). Whisper text through spaCy en_core_web_sm (MIT), one doc
  per Whisper sentence. CTC words chunked by Whisper segment times → local LanguageTool 6.5 (LGPL; 6.6+ needs Java 17,
  this machine has 11), GRAMMAR category only, TYPOS/CASING/PUNCTUATION/STYLE off. Fillers removed everywhere, rates
  only. 90 s for 948 clips. Toy-sentence unit checks match hand counts.
- selection (src/rubric_select.py → reports/phaseB/selection.csv, boxplots.png):
  | feature | exp | ρ | 1 sign | inv | 2 mono | 3 |ρ dur| | partial | 4 | pass |
  |---|---|---|---|---|---|---|---|---|---|
  | finite_clause_ratio | + | −0.075 | ✗ | 5 | ✗ | ✓ | −0.151 | ✓ | ✗ |
  | mean_len_clause | + | +0.046 | ✓ | 4 | ✗ | ✓ | +0.139 | ✓ | ✗ |
  | dep_clause_ratio | + | +0.054 | ✓ | 2 | ✗ | ✓ | +0.043 | ✗ | ✗ |
  | clauses_per_sentence | + | −0.198 | ✗ | 5 | ✗ | ✓ | −0.256 | ✓ | ✗ |
  | mean_dep_distance | + | −0.141 | ✗ | 5 | ✗ | ✓ | −0.144 | ✓ | ✗ |
  | **mattr50** | + | +0.438 | ✓ | 0 | ✓ | ✓ | +0.440 | ✓ | **✓** |
  | **trigram_repeat_rate** | − | −0.246 | ✓ | 1 | ✓ | ✓ | −0.256 | ✓ | **✓** |
  | fragment_ratio | − | +0.052 | ✗ | 3 | ✗ | ✓ | +0.099 | ✓ | ✗ |
  | lt_grammar_per100 | − | −0.157 | ✓ | 2 | ✗ | ✓ | −0.122 | ✓ | ✗ |
  | **self_repair_per100** | data | −0.223 | n/a | 1 | ✓ | ✓ | −0.230 | ✓ | **✓** |
  | cola_frac_bad (existing) | − | −0.506 | ✓ | 1 | ✓ | ✓ | −0.373 | ✓ | ✓ |
  | gpt2_nll_gap (existing) | − | −0.328 | ✓ | 2 | ✗ | ✓ | +0.251 | ✓ | ✗ |
  | repair_wer_nofiller (existing) | − | −0.553 | ✓ | 2 | ✗ | ✓ | n/a | ✓ | ✗ |
  | no_punct (existing) | − | −0.171 | ✓ | 2 | ✗ | ✓ | −0.151 | ✓ | ✗ |
- clause and dependency features come out backwards: weak speakers get unpunctuated run-on "sentences" from Whisper,
  so clauses per sentence and dependency distance rise as the grade falls. lt_grammar has signal (partial −0.12)
  but 2 inversions.
- tension to resolve: self_repair_per100 passes with a NEGATIVE sign (more retraces/restarts → lower grade), so the
  model penalises self-repair, against the plan's "credited, not penalised" rule. Kept in the layer as the
  selection rule defines it; Amrita's call.

## 1012-181255 — rubric layer (mattr50 + trigram_repeat_rate + self_repair_per100) as family features
- hypothesis: rubric-aligned, length-robust lexical range / repetition / self-repair add structure signal the counts miss.
- **cv_testmix 0.5083 (Δ +0.0038 vs 1012-071022), per seed 0.5098 / 0.5135 / 0.5058, better on 3/3**; cv_grouped 0.4739
  (−0.0009), cv_short 0.5318 (was 0.5375), cv_long 0.4544. Fast screen 1012-181205: 0.5078 (+0.0041).
- verdict: **accepted** (1st accepted under the grouped gate).
- lesson: others found such layers neutral. Here lexical range (mattr50, ρ +0.44, partial +0.44 vs repair_wer) is the
  strongest new signal, and it helps the short batch most.

## Decisions 2026-10-13 (Amrita) + 1012-182525 — rubric layer without self_repair_per100
- self_repair_per100 excluded from the model: the CTC restart rate goes with LOWER grades because it measures
  disfluent restarts, not grammatical self-correction. The rubric credits correction and doesn't grade fluency.
  Reported as a finding. PLAN_RUBRIC (updated) replaces it with corrected_error_ratio from C3.
- ledger: 1012-181255 (3-feature layer, 0.5083) marked **superseded** so submit.py can't pick it and the reference
  falls back to the grouped baseline.
- re-test once, mattr50 + trigram_repeat_rate (1012-182525): **cv_testmix 0.5116 (Δ +0.0006 vs 1012-071022)**, per seed
  0.5134 / 0.5170 / 0.5085 (2/3), cv_grouped 0.4748 (±0), short 0.5366 → **rejected**. The model has no rubric layer
  (RUBRIC_LAYER = []). The two passers stay in interpretation and report cards.
- lesson: the Phase B gain was the disfluency signal. Grammar-structure features add ~nothing beyond what the stack
  already has, matching what other participants found.
- submit.py: the new version is still in Amrita's downloads folder, not yet in scripts/ (Amrita to copy).
  TEAM_NAME=ambruhsia is set.

# PLAN_RUBRIC — Phase C1: frozen-encoder heads on grouped folds (src/audio_heads.py)
- bands: WavLM-large 20–22 (a priori; its seed-42 scan best is L21 anyway). WavLM-base 6–8 (scan best L7).
  Whisper-medium 22–24 (scan best L24 = last layer). Whisper-large-v3 is not cached (v19 died on TPU), so it's out.
- heads per encoder: StandardScaler → Ridge (alpha logspace(0,6,31)) and RBF-SVR (C {1,3,10}), with hyperparameters by
  inner GroupKFold(3) on speaker groups. OOF on grouped folds, test = fold mean. 5 min CPU, cached.
- standalone grouped OOF, seeds 1337+2024 (band chosen on 42):
  whisper_med ridge 0.552 (testmix 0.568) / svr 0.559; wavlm_large ridge 0.569 (0.601) / svr 0.575;
  wavlm ridge 0.585 (0.646) / svr 0.595. Notebook-protocol grouped rebuilds: 0.571 / 0.574 / 0.595, so the bands help
  0.005–0.019 standalone. Ridge ≥ SVR everywhere.
- **gate (1012-183422): cv_testmix 0.5192 (Δ −0.0071 vs 1012-071022), 0/3 seeds, cv_grouped 0.4798 → rejected.**
- fair reference, DIAGNOSTIC C1-ref (1012-183537): the notebook protocol rebuilt leak-free, replacing the same 3 columns:
  testmix **0.5170** (0.5166/0.5267/0.5150), grouped 0.4794. C1's 6 heads are 0.0022 worse than that at stack level
  despite better standalone OOF. 6 correlated columns dilute the trees.
- code: wiring kept but OFF (AUDIO_HEADS = False). The check run reproduces the fast baseline 0.5120.
- ledger: fast screen 1012-181205 (superseded 3-feature layer) also marked superseded, so fast screens compare against
  the grouped fast baseline again.
- structural problem: the reference still holds leaky random-fold columns (3 audio, spk_knn, ft_wavlm, 3 DeBERTa),
  worth ~0.005–0.007 of fake gain. Every leak-removal step will be "rejected" by the gate. Needs a decision.

# Overnight run (Amrita asleep; she asked for automation) — plan fixed before the results arrive
Approved: C3 (Qwen, running), C2 (DeBERTa grouped, running in parallel), the leak-free rebaseline, Phase D/E.
Not approved, so not done: any other GPU job (C4), final-pick changes, pushing the Phase F notebook.
1. C3 → pull to cache/qwen/; selection checks for llm_err_per100 (−) and corrected_error_ratio (+); gate experiments
   vs 1012-071022, one per component: (a) qwen heads, (b) the passing Qwen rubric features, (c) the zero-shot judge.
2. C2 → cache/deberta_grouped.csv. C5 decided before the rebaseline: the leak-free stack with vs without
   ft_wavlm_pred (diagnostics); drop it if removing it costs < MIN_GAIN on cv_testmix.
3. ONE --rebaseline (agreed): LEAK_FREE (audio ×3 + emb rebuilt on grouped folds, grouped DeBERTa ×2; spk_knn and
   deberta_ctc dropped; ft_wavlm per C5) + C3 members accepted in step 1 + NESTED_BLEND (blend weights refit inside
   folds). Nesting is needed because per-batch NNLS fits ~2×20 weights on 174 short clips; in-sample it scored
   0.4977 vs 0.5120 in a smoke test, which is not believable. Full + fast rebaseline.
4. Phase D under the gate, one change each, kept cumulatively if accepted: D2 nnls_w, D2 nnls_batch (+ weight std
   across seeds), D3 λ=0.3, D3 λ=0.5, D4 calibration. Confirm on seeds 7/99/555 after 3 accepted.
   C3 members rejected in step 1 get ONE re-test on the leak-free stack (honest text columns may matter more there).
5. Phase E: submit gate-accepted runs via scripts/submit.py, max 2 per UTC day, never chosen by public score.
   NOTE: scripts/submit.py is still the OLD version (the new one is in Downloads), so grouped runs will be refused.

# PLAN_RUBRIC — Phase C3 results (Qwen2.5-7B-Instruct, 4-bit, Kaggle T4; job amritarajput54/shl-qwen-features v1)
- job: 53 min (stage 1 hidden states + judge 9.5 min, stage 2 minimal corrections 30.4 min, stage 3 self-correction JSON
  9.3 min). Model 5.6 GB in 4-bit. Outputs in cache/qwen/. JSON counts parsed 948/948.
- judge: llm_rubric_expected on CTC text mean 3.02 (max 4.0: the garbled text never earns a 5); on Whisper text 3.63.
- selection (Phase B checks):
  | feature | exp | ρ | inv | |ρ dur| s/l | partial vs repair_wer | pass |
  |---|---|---|---|---|---|---|
  | llm_err_per100 | − | −0.585 | 1 | 0.197 / 0.042 | −0.255 | ✓ |
  | corrected_error_ratio | + | +0.075 | 4 | 0.035 / 0.014 | −0.014 | ✗ |
  | llm_rubric_expected (judge) | + | +0.612 | 0 | 0.044 / 0.023 | +0.322 | ✓ |
- findings: llm_err_per100 tracks the grade more strongly than repair_wer (−0.553). The zero-shot judge is the strongest
  single feature yet (no labels used). corrected_error_ratio fails: the LLM's count of self-corrections in CTC text
  doesn't follow the grade either, so self-correction stays a reported finding, not a model input.
- standalone grouped OOF (seed-avg of 3 grouped seeds): **qwen_whisper_pred 0.5885 (testmix 0.5787, short 0.5715)**,
  qwen_ctc_pred 0.5933 (testmix 0.5983). As strong as the best (leaky) DeBERTa column, and honest. Ridge on 21.5k
  dims (last token + mean, layers 16/20/24), alpha by inner GroupKFold; 11 min CPU, cached.
## 1012-195359 — C3a: Qwen heads (qwen_ctc_pred + qwen_whisper_pred)
- **cv_testmix 0.5012 (Δ +0.0110 vs 1012-071022), per seed 0.5061 / 0.5019 / 0.5007, 3/3; cv_grouped 0.4695 (−0.0053);
  short 0.5229 (was 0.5375), long 0.4516.** Fast screen 1012-195308: 0.5058 (+0.0061). → **accepted**.
## 1012-195531 — C3b: Qwen rubric layer (llm_err_per100; corrected_error_ratio failed selection)
- cv_testmix 0.5000 (Δ +0.0012 vs 1012-195359), 3/3 seeds, grouped 0.4694 → rejected (below MIN_GAIN). Kept for interpretation.
## 1012-195658 — C3c: zero-shot judge (llm_rubric_expected, llm_rubric_entropy)
- cv_testmix 0.4999 (Δ +0.0013), 2/3 seeds, grouped 0.4686 → rejected (below MIN_GAIN). For interpretation and
  report cards, as the plan intended.
- lesson: the Qwen hidden-state heads are the gain. The LLM-derived rubric numbers repeat what the heads and
  repair_wer already carry, with small consistent gains that don't clear the bar.
- Phase E: `submit.py 1012-195359 --dry-run` → **REFUSED** ("CV 0.5012 does not beat best submitted run 1011-190112
  (CV 0.4725)"). That is the old guard comparing a grouped cv_testmix with a random-fold cv. The new submit.py (in
  Downloads) fixes it. Not worked around; waiting for Amrita to copy it in.

## C5 + REBASELINE (leak-free stack)
- C2 job (shl-deberta-grouped v1): grouped-fold DeBERTa **deberta_pred 0.5921** (old random-fold 0.6379), **deberta_pair_pred
  0.5611** (old 0.5849); per seed 0.596/0.602/0.600 and 0.571/0.567/0.566. Better although grouped folds are harder,
  probably the newer Kaggle software (the old run had near-collapsed folds). No leak found in the notebook.
- emb_pred rebuilt locally (all-mpnet-base-v2 on CPU, 4.5 min). It reproduces the notebook column exactly on its random
  folds (max |diff| 0.000), then is refit on grouped folds.
- C5 (diagnostics, leak-free stack + Qwen heads, nested blend): WITH ft_wavlm 1012-203402 testmix 0.5139 (short 0.5358,
  long 0.4638); WITHOUT 1012-203506 testmix 0.5198 (short 0.5219, long 0.5153). Removing it costs 0.0059 ≥ MIN_GAIN → **keep**.
  It carries the long batch and hurts the short one; some of its long-batch strength may be twin leakage (unmeasured).
- **REBASELINE 1012-203626**: grouped audio ×3 + emb + DeBERTa ×2 + Qwen ×2, ft_wavlm kept, spk_knn and deberta_ctc dropped,
  NESTED_BLEND on. **cv_testmix 0.5139, cv_grouped 0.4819, short 0.5358, long 0.4638.** Fast: 1012-203652 0.5255.
## D2a 1012-203832 — NNLS blender with test-mix weights
- testmix 0.5091 (+0.0048, 3/3) but **cv_grouped +0.0099** (long 0.4822 vs 0.4638) → rejected by the grouped guard.
  One weighted blend trades long-batch accuracy for short.
## D2b 1012-204012 — per-batch NNLS (separate short / long weights), nested
- **cv_testmix 0.4992 (Δ +0.0147), per seed 0.5129 / 0.4998 / 0.4947, 3/3; cv_grouped 0.4748 (−0.0071); short 0.5160,
  long 0.4612 → accepted.**
- weight stability (3 seed fits / 15 nested fold fits): short: deberta_pair 0.355±0.027 (15/15 folds), lgbm 0.24±0.13,
  qwen_whisper 0.19±0.05 (15/15), whisper_med 0.10±0.03 (13/15), ft_wavlm 0.07±0.06. Long: ft_wavlm 0.357±0.040 (15/15),
  ridge 0.258±0.020 (15/15), rf 0.131±0.009, xgb/svr ~0.06. Sums ~0.97 (short) and ~1.00 (long).
- lesson: text members carry the short batch, audio + trees the long batch. One shared blend can't serve both.
## D3 label denoising (members fit on (1−λ)·y + λ·speaker-group mean; always scored on y)
- λ=0.3 (1012-204321): testmix 0.4990 (+0.0002, 1/3 seeds), grouped 0.4751.
- λ=0.5 (1012-204439): **testmix 0.4979 (+0.0012, 2/3), grouped 0.4745, short 0.5142, long 0.4615.**
- the gate alone rejects both (< MIN_GAIN), but the plan's D3 rule is "keep it only if cv_testmix doesn't get worse by more
  than 0.002". Both pass; λ=0.5 is better on every metric, so **λ=0.5 is kept**. Ledger verdict of 1012-204439 set to
  accepted under that rule, so later runs compare against it.
## D4 + re-tests (vs 1012-204439)
- D4 nested weighted calibration (1012-204559): testmix 0.4980 (−0.0001, 0/3) → rejected (as the established fact predicts).
- C3b re-test llm_err_per100 (1012-204715): 0.4978 (+0.0001) → rejected. C3c re-test judge (1012-204830): 0.4989 (−0.0009) → rejected.
## Confirmation on fresh grouped seeds 7/99/555
- reference = rebaseline config (hill blend, no denoise) 1012-210029: testmix 0.5066, grouped 0.4793.
- current (per-batch NNLS + denoise 0.5) 1012-210134: **testmix 0.4948 (+0.0118), 3/3 seeds, grouped 0.4740 (−0.0053)** → the Phase D
  gain holds on fresh folds (+0.0160 on 42/1337/2024).

# Overnight summary (for Amrita)
- best honest model: **run 1012-204439, cv_testmix 0.4979 / cv_grouped 0.4745 / short 0.5142 / long 0.4615** (speaker-grouped
  folds, blend nested in folds). Leak-free stack: grouped audio ×3 + emb + DeBERTa ×2 (C2) + Qwen heads ×2 (C3) + ft_wavlm (C5);
  per-batch NNLS; label denoising λ=0.5. Confirmed on fresh seeds.
- vs the 0.3707 submission (1011-190112): test predictions mean |diff| 0.154, corr 0.969, batch means similar.
- **not submitted**: the old scripts/submit.py refuses grouped runs (scheme mismatch). After copying the new submit.py:
  `python scripts/submit.py 1012-204439 --dry-run`, then without --dry-run.
- final picks (Phase E): (1) best cv_testmix = 1012-204439 once submitted; (2) the 0.3707 run (no other run's cv_testmix is
  within 0.005 in the same scheme).
- GPU used tonight: Qwen 53 min + DeBERTa ~1.5 h (in parallel). Disk was full; freed 2.2 GB of my scratch files.

## Phase E — submission 2026-10-13 07:09 UTC
- new scheme-aware scripts/submit.py installed by Amrita; dry run OK; **submitted 1012-204439** (cv_testmix 0.4979).
- **public 0.3441** (was 0.3707): −0.0266, far beyond public noise (one clip ≈ 0.003). **Rank ~49 / 436** (was ~90).
  The leak-free, test-mix validation translated into a real test gain.
- final picks now: (1) 1012-204439 (best cv_testmix, public 0.3441); (2) the plan's second pick is the best public among
  runs within 0.005 cv_testmix of the best; for now that is also 1012-204439, with 1011-190112 (0.3707) as the fallback.
- note: scores.py crashes printing the '—' in run descriptions on the Windows console; run it with PYTHONIOENCODING=utf-8.

# PLAN_RUBRIC — Phase G (decisions 2026-10-13: C4 approved and pushed first; no ft_wavlm rebuild; G1–G3 then Checkpoint G2)
- G0: C4 crop job pushed (amritarajput54/shl-crop30-embeddings): first 30 s of every clip, WavLM-large (CC BY-SA 3.0, commercial
  use allowed with attribution/share-alike; licence via the UniSpeech repo) + Whisper-large-v3 encoder (Apache-2.0). All layers
  mean+std, float16. Running.
- disk again near-full (657 MB): deleted my HF caches roberta-base (BERTScore, 12 Oct) and all-mpnet-base-v2 (embeddings cached),
  so 1.6 GB free.
- deadline: Amrita's message left it blank and the Kaggle API doesn't list the competition. Asked.
## G1 insight table (src/insight.py → cache/insight.csv)
- best run reproduced exactly (1013-100418 = 1012-204439, 0.4979) with per-seed OOF saved (run.py now writes oof_seed* columns).
- topics: k-means k=10 (fixed) on mpnet embeddings, train+test, no labels. Top words per topic in reports/phaseG/error_by_topic.csv.
## G2 error analysis (reports/phaseG/, reports/worst_clips.md)
1. **compression of the extremes, much worse on short clips.** Mean residual by level: 2.0 → −0.58 long / −0.52 short (over-predicted);
   5.0 → +0.28 long / **+0.71 short** (RMSE 0.79, n=10); short 4.5 +0.47 (n=4). Mean prediction for true 5.0: 4.72 long, 4.29 short.
   The short training batch has almost no high scores: 14 of 174 short clips are ≥ 4.5, vs 171 of 555 long.
2. **prompt shift between train and test.** Topic "floods" = 10.2% of test (15% of test-short), **0 training clips**. Topic 8
   (role model / books / friends) = 29.6% of test (38.8% of test-short) vs 3.0% of train. The big training prompts, playground
   19% and market 16%, are only 6.5% / 5.1% of test. Test-short is mostly prompts the model has rarely or never seen.
3. **within short clips, the amount of speech moves the prediction** (fluency/length, which the rubric doesn't grade).
   Speech-time quintile Q1 (10–32 s) mean residual +0.21 (under-predicted), Q5 (41–45 s) −0.19 (over-predicted); long batch
   within ±0.05. Smaller: RMSE rises with ASR disagreement (0.34 → 0.55), largely through label level. CTC failures (repair_wer
   > 0.6) on good speakers (≥ 4) are under-predicted by +0.70 (n=16), but such failures are 11.6% of train and only 1.9% of test.
- worst 25: 17 long / 8 short; 10 of them are true 5.0s; 5 have repair_wer > 0.6.
## G3 rubric conformance tests (src/rubric_tests.py → reports/phaseG/rubric_tests.csv; one-sided Mann–Whitney, rank-biserial r)
  | descriptor | test | short | long |
  |---|---|---|---|
  | L1–2 memorized/simple | mattr50 lower at ≤2.5 | ✓ r .37 | ✓ r .38 |
  | L1–2 memorized/simple | trigram_repeat_rate higher at ≤2.5 | ✓ .42 | ✓ .15 |
  | L2 incomplete | fragment_ratio higher at ≤2.5 | ✗ .05 | ✗ −.08 |
  | L2 incomplete | no_punct higher at ≤2.5 | ✓ .14 | ✗ .03 |
  | L3 vs L4 errors | llm_err_per100 lower at 4–4.5 | ✓ .28 | ✓ .47 |
  | L3 vs L4 errors | lt_grammar_per100 lower at 4–4.5 | ✗ .02 | ✓ .18 |
  | L5 complex | dep_clause_ratio higher at 5 vs 4 | ✗ −.11 | ✗ .06 |
  | L5 complex | clauses_per_sentence higher at 5 vs 4 | ✗ −.27 | ✗ .13 |
  | L4–5 self-correction | corrected_error_ratio higher at ≥4 | ✗ .07 | ✗ .08 |
  | model | mean OOF rises at every level | ✓ 2.52→4.29 | ✓ 2.58→4.72 |
- what short clips lose: LanguageTool error density no longer separates level 3 from 4 (r .02 vs .18), and the predicted
  range is narrower (4.29 vs 4.72 at true 5.0). Complex structures and self-correction aren't measurable in either batch.
- C4 done (shl-crop30-embeddings v1; WavLM 8.6 min + Whisper-large-v3 3.2 min on T4). cache/crop30_wavlm_large.npz (948×25×2048)
  and cache/crop30_whisper_large_v3.npz (948×33×2560), float16, all finite. G6 (adversarial check + gate) waits for Amrita's go.

## Is "short ⇒ weaker" true of test? Label-free check (Amrita's point 1; diagnostic only, nothing changed)
- deadline: Amrita's is in 5 days (2026-10-18); the competition closes in ~3 months. The "within two days" rule doesn't
  apply, but Phase F must be done within 5 days, so its skeleton starts now.
- the G8–G11 text was not saved to any PLAN_RUBRIC file (all predate Phase G). Asked for it; not implemented from memory.
- yardstick check (train, at a fixed grade): the label-free measures are NOT batch-neutral. Short clips get a slightly higher
  CTC judge score (+0.1–0.2) and a lower llm_err_per100 (~10 points) at the same label.
- raw means: test-short judge 3.017 vs train-short 3.037; llm_err 34.1 vs 30.5; repair_wer 0.258 vs 0.252. By measures that
  never saw a label, **test-short looks like train-short, not like strong long clips** (if anything slightly weaker).
- label-free implied grade (label ~ judge×2 + llm_err + repair_wer, linear, calibrated on one batch of train; train rows OOF by
  speaker group): short-calibrated, test-short 2.97 vs model 3.05 (agree). Long-calibrated, 3.56 (+0.51 vs model), but the same
  long calibration overshoots train-short by +0.58 against the true labels, so the gap is the measures' batch bias.
- the real short-clip problem is compression within the batch: on train-short the model predicts ≥ 4.0 for 10.3% while 32.8%
  are ≥ 4.0 (pred sd 0.67 vs label sd 0.90). Test-short: 6.1% predicted ≥ 4.0, sd 0.62, the same pattern. This points at a
  per-batch spread correction fitted inside folds (G5(c)) more than at a population-shift hedge.

## G8 — label-free shift check (src/shift_check.py → reports/phaseG/shift_check_*.csv; no model change)
- signal means: judge test-short 3.017 / train-short 3.037 / train-long 3.009; llm_err_per100 34.1 / 30.5 / 35.7; mattr50 0.710 /
  0.717 / 0.721; cola_frac_bad 0.478 / 0.387 / 0.431. Mean |SMD| of test-short: 0.155 to train-short, 0.101 to train-long.
- batch-blind model (ridge on mattr50, trigram_repeat_rate, llm_err_per100, cola_frac_bad, llm_rubric_expected; grouped OOF): RMSE
  0.712 grouped / 0.757 testmix. Mean prediction: train-short true 3.141, main 3.155, blind 3.563 (main − blind −0.408);
  test-short main 3.045, blind 3.406 (−0.361); train-long true 3.587, main 3.601, blind 3.453 (+0.148); test-long +0.074.
- the plan's criteria, literally: (A) closer to train-long = TRUE, (B) main more than 0.1 below blind = TRUE → **SUSPECT**.
- but both are confounded:
  - (B): on train-short, where labels exist, the same gap (−0.41) appears and the main model is right (3.155 vs 3.141) while the
    blind model overshoots by +0.42. Test-short reproduces that gap (−0.36), i.e. it behaves like train-short.
  - (A): test-short is closer to train-long because it has MORE errors (llm_err, cola_frac_bad worse than every train group),
    which is closeness in the weaker direction, not "rated like strong long clips".
- consequence for Phase E: by the literal rule pick 2 = best batch-neutral run within 0.005 of the best cv_testmix. None
  exists in the current scheme (best batch-neutral is the rebaseline at 0.5139, 0.016 behind), so it falls back to 0.3707.
  Decision left to Amrita.
- **Decision (Amrita, 2026-10-13): batch prior treated as NOT suspect; standard Phase E pick-2 rule.** Reason: on train-short
  (labelled) the batch-blind model overshoots by +0.42 and the main model is right; test-short shows the same gap (−0.36), so it
  behaves like train-short. Condition (A) was met in the weaker direction (more errors, worse CoLA), not "rated like strong long
  clips".
- report note: label-free signals don't transfer across batches (the judge is ~3.0 everywhere). Short clips score lower than their
  measurable grammar predicts: a possible rater, recording or amount effect.

## Shortcut-removal gate experiments (accept if cv_testmix not worse by > 0.002 AND the named bias metric at least halves)
- bias metrics, named before the runs (src/bias_metrics.py), baseline = best run reproduction 1013-100418:
  G9 short-clip slope of OOF residual on speech time **−0.092 per 10 s**; G10 mean residual of clips with repair_wer > 0.6
  (cut fixed from G2) at label ≥ 4 **+0.695** (n=16; all labels −0.041).
## G9 1013-104432 — amount-blind: drop duration + n_words (the only raw amount counts; all rates kept)
- cv_testmix 0.4966 (+0.0014), grouped 0.4752 (+0.0007), short 0.5114, long 0.4634 → CV condition met.
- bias: slope −0.092 → **−0.083** (−9%, needs −50%) → **rejected**.
- finding: the amount effect doesn't come from the raw counts. The learned members see the whole transcript and audio and
  carry it themselves.
## G10 1013-104609 — CTC-failure neutralisation: repair_wer_nofiller > 0.6 → missing (median) in train + test
- cv_testmix 0.4979 (+0.0001), grouped 0.4747 → CV condition met.
- bias: +0.695 → **+0.688** → **rejected**.
- finding: when CTC fails, the under-scoring comes from the CTC-based learned members (qwen_ctc_pred, deberta_pair_pred) and GPT-2
  on the raw text, not from the repair feature. A fix would mean neutralising those members on failed clips (not declared here).
## G11 — topic-held-out validation (src/topic_cv.py; cv_newtopic in run.py)
- topic folds: validate on topic T's training clips, train on clips outside T whose speaker group has no clip in T (9 non-empty
  training topics; "floods" has none). cv_newtopic = blend fitted outside T, applied to T's topic-held-out member predictions,
  test-mix weighted.
- members refit on topic folds: audio ×3, emb, Qwen ×2 (cached), 9 families (seeing the topic-held-out heads as features).
  DeBERTa ×2 keep their speaker-grouped OOF; ft_wavlm keeps its random-fold OOF (both stated).
- measure run 1013-105952 (best config, blend on speaker-grouped OOF): cv_testmix 0.4979, **cv_newtopic 0.5043** (plain 0.4832).
  Member gap (topic − speaker-grouped OOF RMSE): emb +0.104, mlp +0.093, krr +0.067, qwen_ctc +0.047, svr +0.037, qwen_whisper +0.029,
  xgb +0.028, lgbm +0.024, knn/rf +0.014, ridge +0.009, et +0.008, wavlm/wavlm_large +0.006, whisper_med +0.000.
  Text members lean on familiar prompts; audio heads don't.
## 1013-110218 — G11: blend weights fitted on topic-held-out OOF (shortcut-removal gate, bias = cv_newtopic)
- cv_testmix 0.4982 (−0.0003, within 0.002), **cv_newtopic 0.5013 (improves 0.0030 ≥ 0.002)**, grouped 0.4754, short 0.5140,
  long 0.4627; seed spread 0.0068 (was 0.0153). → **accepted under the shortcut-removal gate** (ledger verdict set accordingly).
- short-batch weights: rf 0.37, deberta_pair 0.35, qwen_whisper 0.16, whisper_med 0.09 (lgbm, topic-sensitive, drops out).
  Long: ft_wavlm 0.40, et 0.28, ridge 0.23, deberta_pair 0.06.
- lesson: the blend can be made to trust topic-robust members at no CV cost, and seeds agree more. That fits a test set where
  ~40% of clips answer prompts the training set barely has.
- ledger: 1012-204012 (D2b) and 1012-204439 (D3, the submitted run: public 0.3441, submission record untouched) marked
  superseded. The current state is G11 (1013-110218, accepted under the shortcut gate), and the reference rule picks the lowest
  cv among accepted runs, so leaving them accepted would compare later runs against a state no longer in the model.

## G4 — short views of long clips (src/shortview.py; section-4 features ported from the notebook, exact: max|diff| ≤ 1e-14)
- view: each long training clip cut (word timestamps) at a duration drawn from the test-short durations (seed 0). Duration,
  pause, text, repair and no_punct features recomputed. GPT-2/CoLA and the learned *_pred columns keep the clip's values (they
  need the GPU or the audio).
- feature change (view − clip): duration −15.9 s, n_words −34.9, mean_sent_len −5.3, std_sent_len −5.0, text_ttr +0.069,
  no_punct +0.21, midclause_pause_per_min −0.29, the rest ≈ 0.
- score (1013-111126, no training change): views predicted **−0.173 lower on average** (mean |shift| 0.223), most at the top
  (4.5 −0.25, 5.0 −0.26). Same speakers, same grades. Strong short clips (≥ 4.5) residual +0.663 (n=14).
- augment (1013-111431): families trained on originals + 558 views → cv_testmix 0.4984 (−0.0003), 1/3 seeds, short ≥ 4.5
  residual +0.674, view shift −0.166 → **rejected**.
- lesson: the short penalty doesn't sit in the features. In the short-batch blend only rf changes on a view; the learned members
  keep their values. The shift comes mostly from the short-batch blend weights (sum ≈ 0.97 vs 1.00 for long), i.e. the batch
  prior in the per-batch blend. That agrees with G8 (short training clips are graded lower at the same measurable grammar).
- G5(a) (denoising λ=0.5 only) is already in the model since D3.
## G5(c) 1013-111744 — per-batch linear calibration (a_batch + c_batch·pred), nested in folds
- cv_testmix 0.5005 (−0.0023), 0/3 seeds, short 0.5180 (was 0.5140); strong-short residual +0.674 → +0.580 → **rejected**.
  Stretching short predictions helps the 14 strong short clips but costs more on the many mid-level ones.
## G6 adversarial check (src/crop30.py): short vs long from EQUAL-LENGTH first-30-s crops (bands fixed a priori)
- logistic regression, GroupKFold(5) by speaker: **AUC 0.932** (WavLM-large L20–22), **0.945** (Whisper-large-v3 L29–32); full-length
  WavLM-large 0.970. → the batches differ in recording (channel/device/environment), not just length. Report it; keep relying on
  per-batch blending.
## G5(b) 1013-114041 — test-mix sample weights (short 2.86 / long 0.42) in member training
- families except knn (no sample_weight), leak-free heads, Qwen heads (refit with weights); DeBERTa/ft_wavlm unchanged.
- cv_testmix 0.4989 (−0.0007), 1/3 seeds, short 0.5147, strong-short residual +0.66 → **rejected**. The per-batch blend already
  carries the batch weighting; weighting the members adds nothing.
## G6 1013-114357 — crop30 heads (ridge on WavLM-large L20–22 / Whisper-large-v3 L29–32 of the first 30 s)
- standalone grouped OOF 0.599 / 0.600 (full-length heads: wavlm_large 0.575, whisper_med 0.573). Topic-held-out 0.622 / 0.623.
- cv_testmix 0.4993 (−0.0011), 0/3 seeds, cv_newtopic 0.5039 (worse), short 0.5153; blend weight 0 → **rejected**.
- lesson: equal-length crops carry less signal and remove nothing the model needed removed. The batches differ in recording
  (adversarial AUC 0.93–0.95 even at equal length), so per-batch blending stays the right tool.
- **G4–G6 done. Current state = G11 (1013-110218): cv_testmix 0.4982, cv_newtopic 0.5013.**

# New methods (approved by Amrita 2026-10-13, outside the plan; brainstorm "transform the problem")
## 1013-122821 — Rank-N-Contrast member rnc_pred (src/rnc.py)
- design fixed in advance: Qwen ×2 + audio ×3 embeddings, StandardScaler → PCA 64 per source inside the fold, MLP 320→256→32 (dropout 0.2),
  L2-normalised, cosine/τ 0.2, Adam 1e-3 / wd 1e-3, 300 full-batch epochs. RNC loss with positives from a DIFFERENT speaker AND topic,
  same-speaker clips excluded. Ridge on the 32-d embedding. Grouped OOF + topic OOF, cached. Toy check: ordered 2.61 vs shuffled 5.45.
- **standalone: grouped 0.541, testmix 0.561, topic-held-out 0.583.** The best single learned member so far (deberta_pair 0.561, qwen_whisper 0.588).
- stack: cv_testmix 0.4978 (+0.0004, 2/3), cv_grouped 0.4727 (−0.0026), long 0.4587, **cv_newtopic 0.5050 (worse by 0.0037)** → rejected
  (normal gate not met, and cv_newtopic got worse by more than 0.002). Blend weight 0.15 long / 0.08 short.
- lesson: a grade-ordered, speaker- and topic-crossing embedding makes the strongest single model, but it's built from the same embeddings
  the stack already uses, so it re-expresses rather than adds.
## 1013-123102 — ordinal distribution + EM prior adaptation per batch (src/ordinal_em.py)
- proportional-odds model of the blend prediction per batch (nested), Saerens EM on the unlabelled target set, posterior mean.
  Toy check OK (simulated 75% high-grade target: EM estimates 71%, RMSE 0.443 → 0.427).
- **cv_testmix 0.5294 (−0.031), 0/3 seeds → rejected.** Topic-held-out simulation (testmix): raw 0.5013, ordinal 0.4994, **ordinal+EM 0.5351**.
- EM test-prior estimate: test-short share ≥ 4.0 = 0.066 (train 0.328), mean 2.955 (train 3.144). This is NOT evidence about test: EM on
  compressed posteriors drifts toward the model's centre (the known failure mode). The simulation shows EM hurts under a real shift.
## 1013-123336 — ordinal control (EM off)
- cv_testmix 0.4980 (+0.0002, 0/3 seeds), grouped 0.4733 → rejected.
- lesson: the gains in this data come from what the members see (new information, honest validation), not from reshaping the output.
- current state unchanged: **G11 1013-110218, cv_testmix 0.4982, cv_newtopic 0.5013.** Nothing new beats the submitted run, so nothing submitted.

# Phase F — final notebook (started 2026-10-14; Amrita's deadline 2026-10-18)
- chosen run = final pick 1 = **1012-204439** (best cv_testmix among submitted runs; public 0.3441). Reproduced bit-for-bit as 1014-052555
  (max |diff| 1.8e-15) and as 1014-052942 with the candidate matrices + blend weights saved (runs/<id>/cands.npz) for the notebook.
- **TRAINING RMSE 0.3667** (final pipeline refit on all 732 rows, scored in-sample; cheap members refit; DeBERTa ×2 and ft_wavlm keep
  their OOF because their fold models weren't saved). Honest: cv_grouped 0.4745, cv_testmix 0.4979, short 0.5142, long 0.4615.
- experiment.py restored to the loop's current state (G11: TOPIC_OOF + BLEND_ON_TOPIC on).
- final notebook (kaggle/final/final_notebook.ipynb, 17 cells): sections 0–8 (report with findings, setup, preprocessing plots,
  transcription, rubric layer + selection + G3 tests + G2 tables, members, per-batch NNLS stacking, results with the TRAINING RMSE box,
  metrics, plots, three report cards, submission). Default path reads the saved candidates and redoes the blend; RECOMPUTE_MEMBERS
  refits the CPU members.
- **verified on Kaggle (fresh session, Save & Run All)**: private notebook amritarajput54/shl-grammar-scoring-final (CPU, internet
  off), private dataset amritarajput54/shl-grammar-final-artefacts (81 MB, plus copies of train/test csv, transcripts and x-vectors
  used for local testing) + the caches dataset. All asserts pass; submission.csv byte-identical to run 1012-204439 at %.10f (raw max
  |diff| 1.3e-15; floating-point order differs across machines, hence the fixed format); total runtime 8.7 s. Downloaded output
  re-checked locally: identical, 216 rows.
- metrics shown: TRAINING RMSE 0.3667; cv_grouped 0.4745, cv_testmix 0.4979 (95% CI 0.455–0.542), short 0.5142, long 0.4615; MAE 0.366,
  Pearson 0.886, Spearman 0.878, QWK 0.862, within 0.5 72.7%, within 1.0 95.4%.
- report text: the public/CV gap explained (CV pessimistic by design; errors concentrate at extreme grades, which the public 130 clips
  likely hold fewer of). Findings: restarts ≠ self-correction; short clips score below their measurable grammar; amount spoken isn't
  graded but the model rewards it; CTC failures penalise accented speakers. Placeholders left for Amrita: credits, her reading of
  the ten true 5.0s.
- RECOMPUTE_MEMBERS path tested locally: every CPU member refit from cached embeddings reproduces the cached candidates (min corr
  1.0000), same cv numbers, submission max |diff| 1.8e-15; 826 s.

## G12 — transductive pseudo-labelling, simulated on held-out topics (pre-registered 2026-10-14; src/pseudo_label.py)
- hypothesis: test has prompts train lacks (floods: 10% of test, 0 train clips); letting the refittable members see the new topic's
  clips (labelled with the G11 pipeline's own predictions, weight 0.5, one round) adapts them to the topic without labels.
- simulation: for each of the 9 topics with train clips and each seed, G11 state fitted outside T (speaker-safe) predicts T; T joins
  the training data of the 4 leak-free ridge heads, 2 Qwen ridge heads and the 9 families; refit once; re-predict T with the same
  per-batch blend weights. "before" reproduces G11 (seed 42 cv_newtopic 0.5013).
- cv_newtopic before → after: seed 42 0.5013 → 0.4997, 1337 0.5026 → 0.5014, 2024 0.5022 → 0.5003; **mean gain +0.0015** (need ≥ 0.002),
  better on 3/3 seeds (need 2), better in 5/9 topics (need 5). Biggest gains on topic 7 (−0.013) and 5 (−0.005); topic 3 +0.002 worse.
- **verdict: FAIL → not applied to test** (gain below MIN_GAIN; the other two conditions pass). No re-run with other weights or rounds:
  that would be choosing the setting on the same held-out topics.
- lesson: self-training on the pipeline's own predictions moves it only slightly, consistently but by less than the noise threshold;
  the topic shortcut needs new information (labels or a topic-free member), not a re-fit on its own outputs.
- note: the first attempt crashed (LightGBM access violation) because it ran under miniconda's Python, not .venv; pseudo_label.py
  now imports _bootstrap like run.py. Results table: reports/phaseG/g12_pseudo_sim.csv. Final notebook's "what didn't" list updated
  (local source only; the Kaggle copy gets it at the next push).
- current state unchanged: **G11 1013-110218 (loop), chosen run for the final notebook 1012-204439.**

## Phase F revisions (Amrita's list, 2026-10-14)
- **1014-063051: FINAL artefact v2** (pick-1 config, TRAIN_FIT). Fixes the TRAIN_FIT refit, which refit the Qwen heads on the
  denoised target although the run's Qwen heads were fitted on raw y. **TRAINING RMSE 0.3688** (was 0.3667). Candidates, blend
  and submission identical to 1012-204439 (max |diff| 1.8e-15). Its ledger verdict "rejected" is only the comparison with the
  loop's reference; this is an artefact rebuild.
- **1014-063230: diagnostic**, pick-1 config without ft_wavlm_pred (as a feature and as a blend column): cv_testmix **0.5173**
  (vs 0.4979), long 0.5134 (vs 0.4615), short 0.5191 (vs 0.5142). Its blend weight in the chosen run: short 0.023, long 0.276.
  Used in the notebook's Limitations.
- **1014-064559: Qwen heads on the denoised target** (new flag QWEN_DENOISED; normal gate vs G11 1013-110218):
  - cv_testmix 0.4998 (−0.0017, 0/3 seeds), grouped 0.4761, cv_newtopic 0.5013 (unchanged) → **rejected**; flag stays False.
  - lesson: the raw-y target was an oversight from adding D3 (denoising) after C3 (Qwen heads), but raw y is the better target
    for these heads. TRAIN_FIT now follows whatever heads() uses.
  - note: the G11 topic-held-out Qwen heads (`topic_heads`) fit y_fit; G12's simulation did the same, so its "before"
    reproduced G11.
- **final notebook** (kaggle/final/final_notebook.py → .ipynb, 34 cells):
  1. Appendix A0–A8 holds every audio / GPU stage, ported from the job notebooks, behind RECOMPUTE_GPU = False:
     sentinel flatness, Whisper, CTC, GPT-2 + CoLA + mpnet, frozen encoders, x-vectors, the WavLM fine-tune, DeBERTa ×2, Qwen.
     Computed live from the cached transcripts and embeddings, each checked against the run:
     - batches, test-mix weights, speaker groups (ARI 1.0) and grouped folds (identical);
     - the 19 handcrafted columns (max |diff| 7e-15);
     - the rubric features (spaCy 3.8.0 model bundled), the selection table (same pass/fail) and the conformance tests (same p).
     LanguageTool stays cached. Section 2's text now says what is computed and what is cached.
  2. Asserts train_fit_pred exists and is finite; Amrita's explanation added under the box.
  3. New public-vs-CV paragraph: 130 OOF clips drawn with p ∝ test-mix weight, 2,000 times → **0.427–0.567**
     (lowest resample 0.376); 0.344 is outside.
  4. Rubric attribution:
     - surrogates on the 14 rubric features → final OOF prediction; out-of-fold R² ridge 0.588, LightGBM (4 leaves) 0.665;
     - grouped permutation importance (held-out R² drop), ridge / LightGBM: accuracy 0.31 / 0.60, structure 0.24 / 0.15,
       range 0.18 / 0.18, completeness 0.03 / 0.00;
     - plotted next to the blend-weight chart.
  5. Imports and constants in the first code cell; three-line TL;DR; model/licence table; ft_wavlm weight and the
     cv_testmix without it in Limitations; self-correction stated once (Finding 1); error analysis and finding numbers
     recomputed live. Licences were checked on the model cards: the CoLA model (textattack/roberta-base-CoLA) states none.
  6. Credits text as given; the 5.0 reading stays a placeholder.
- artefacts rebuilt (29 files, 105.7 MB): + qwen_gen.jsonl and the spaCy model zip; transcripts, x-vectors and the csvs are now
  listed in build_artefacts.py; final_cands.npz from 1014-063051.
- local run: all asserts pass, submission byte-identical to 1012-204439, 28 s. The Kaggle re-run waits for Amrita's OK to push
  (artefacts dataset version + notebook).
- RECOMPUTE_MEMBERS path rechecked on the revised notebook: every CPU member refit reproduces the cached candidates (min corr
  1.0000), same cv numbers, submission byte-identical at %.10f (raw max |diff| 1.8e-15); 684 s. kernel-metadata id set to the
  live slug amritarajput54/shl-grammar-scoring-final (the old id shl-grammar-final doesn't exist).
- **Kaggle re-verify of the revised notebook** (fresh session, Save & Run All, pushed by Amrita; artefacts dataset v2): all asserts
  pass, submission.csv byte-identical to 1012-204439 at %.10f (also rechecked locally), TRAINING RMSE 0.3688, interval 0.427–0.567,
  surrogate R² 0.588 / 0.665, bundled spaCy 3.8.0 loaded (all rubric features identical to the run), 46.9 s. One difference
  from the local run: the speaker-grouped folds rebuilt live don't match the run's under Kaggle's scikit-learn. That's a soft
  check by design; the run's saved folds are used throughout. The message now names the version (goes out with the next push).
  The first push failed in papermill because the rebuilt .ipynb had lost its kernelspec (fixed in 8694cd9).
- **submitted 1013-110218 (G11) by Amrita's override** (`ALLOW_FORCE=1 ... --force`, run by her; the guarded submit refused it:
  CV 0.4982 vs 0.4979): **public 0.3426** (vs 0.3441), rank ~46 of 442 (was ~51). The 0.0015 is within public-board noise; the case
  for it is the private board (less reliance on the prompt topic: cv_newtopic 0.5013 vs 0.5043), not this number.
- Laya job pushed by Amrita (amritarajput54/shl-laya-features v1, T4, internet on for the model download); running.

## 1014-110044 — Laya members (laya_ctc_pred / laya_whisper_pred; normal gate vs G11 1013-110218)
- hypothesis: a rubric-conditioned decision model (Laya, ModernBERT-large, Apache-2.0) adds information the stack lacks. Each
  transcript was asked 5 rubric questions (overall grammar on the rubric, accuracy, complexity, incomplete sentences,
  self-correction); ridge heads were fit on its encoder state + the question probabilities. Kaggle T4 job, 2.4 min.
- zero-shot signal (Spearman with the grade, train): ctc accuracy +0.45, ctc incomplete −0.40, overall grammar +0.24 / +0.22
  (vs Qwen judge +0.61); "self-correction" probability −0.19 (higher for lower grades, as in Finding 1).
- heads alone: grouped OOF 0.664 (CTC) / 0.652 (Whisper), weaker than Qwen 0.59 and DeBERTa pair 0.56.
- stack: cv_testmix 0.4974 (+0.0008, 2/3 seeds), grouped 0.4749 (−0.0005), short 0.5130, long 0.4624, cv_newtopic 0.4993
  (vs 0.5013) → **rejected** (gain < MIN_GAIN); LAYA_HEADS back to False. Blend weight 0.06 short / 0.01 long.
- lesson: a general decision model, even when asked rubric questions, sees grammar less sharply than an LLM's minimal
  correction or a fine-tuned pair model; it slightly helps on unseen topics, but not enough to clear the gate.

## GPU job: Laya encoder fine-tuned on the (CTC raw, Whisper clean) pair (approved 2026-10-14; amritarajput54/shl-laya-ft-pair)
- hypothesis: the pair input is the strongest text member (deberta_pair_pred, 0.56); a different backbone (Laya's ModernBERT-large,
  RL-trained for typed decisions) fine-tuned the same way adds new information on the short batch.
- recipe fixed before running: deberta_pair's recipe plus stabilisers (layer-wise lr decay 0.9 from 2e-5, head 1e-3, 20% warm-up,
  clipping, micro-batch 4 x 2, gradient checkpointing); grouped seed-42 folds, seeds 42/7/2024; stops early if 2 of the first
  5 folds collapse (val RMSE >= 0.97 x predict-the-mean). Local side: LAYA_FT flag, normal gate. Pushed by Amrita; running.
- job ran as version 1, 91 min on a T4 (a duplicate push, v2, was cancelled by Amrita). **No fold collapsed**: val RMSE
  0.551–0.658 against ~1.0 for predicting the mean; the stabilisers worked where DeBERTa-v3-large had collapsed on 5/15 folds.
  Alone (3 seeds) **0.610** vs deberta_pair 0.561; correlation with deberta_pair 0.93.
## 1014-135216 — + laya_ft_pair_pred (LAYA_FT; normal gate vs G11 1013-110218)
- cv_testmix 0.5009 (−0.0027, 0/3 seeds), grouped 0.4764, short 0.5179 (vs 0.5140), long 0.4627, cv_newtopic 0.5015 →
  **rejected**; LAYA_FT back to False. The blend gave it 0.16 short weight, taken mostly from deberta_pair; out of fold that
  swap loses accuracy.
- lesson: a second fine-tuned pair model with another backbone repeats deberta_pair (r 0.93) and adds blend variance, not
  information. Two GPU members today (Laya frozen, Laya fine-tuned) both failed: the transcript side looks saturated for this
  data. Further gains would need a different input (what the speaker said, not how a text model reads it), not another
  text backbone.

# Wider research and the approved idea list (2026-10-14; Amrita: "run all, most promising first"; GPU 17 h available)
- public write-ups below our 0.3426: AMITABHSHUKLA (#1, 0.3064), Techy-A (0.3116; 0.3178 before public-board tuning), PRINCE2-AI
  (0.3262). All three use Voxtral-Mini-3B audio-LLM hidden states as their strongest single view (0.535-0.540 on unseen
  speakers); their CV (0.495-0.505) is worse than ours (cv_grouped 0.4745). Order agreed: #1 Voxtral (GPU), #2 SVR heads (CPU, in
  parallel), #3 verbatim Whisper-large-v3-turbo, #4 Laya checklist, #5 Laya decision-head fine-tune, #6 Phi-4-multimodal,
  #7 attention pooling over Voxtral frames.
## 1014-151034 — idea #2: RBF-SVR head on WavLM-large layer 20 (SVR_HEADS; normal gate vs G11 1013-110218)
- head alone 0.5745 grouped (ridge on all layers 0.5750; r 0.95); blend weight 0. cv_testmix 0.4982 (±0.0000, 1/3 seeds),
  grouped 0.4755, cv_newtopic 0.5002 → **rejected**; SVR_HEADS back to False.
- lesson: the head type isn't the gap; our ridge heads already match the #1's WavLM-SVR view (0.566 on their folds).
- GPU jobs prepared (all pushes by Amrita; each output gets its own normal-gate test, local wiring behind flags that are off):
  #1 kaggle/voxtral (Voxtral-Mini-3B LM audio-token states + encoder layers; src/voxtral_feats.py, flag VOXTRAL),
  #3 kaggle/verbatim (Whisper-large-v3-turbo verbatim transcripts -> Qwen heads + DeBERTa pair; src/verbatim_feats.py, flags
  VERBATIM_QWEN / VERBATIM_DEBERTA), #4 kaggle/laya_checklist (8 yes/no questions per sentence + verbatim rubric score;
  src/laya_checklist.py, flag LAYA_CHECKLIST), #5 kaggle/laya_dh (Laya decision-head fine-tune on threshold + score questions).
## 1014-161629 — idea #1: Voxtral-Mini-3B audio-LLM head (VOXTRAL=("voxtral_pred",); normal gate vs G11 1013-110218)
- job: 948 clips in 16.9 min on a T4, every state finite (v1 failed: transformers was imported before mistral-common was
  installed, so the processor got a tokenizer without a chat template; fixed).
- head alone: **0.5328** grouped (short 0.5497, long 0.5274), the best single member we have (deberta_pair 0.5611); but r 0.94
  with deberta_pair and 0.90 with wavlm_large.
- stack: cv_testmix 0.4978 (+0.0004, 2/3 seeds), **cv_grouped 0.4727 (−0.0027)**, long 0.4586 (−0.0043), short 0.5151
  (+0.0011), cv_newtopic 0.5012 → **rejected** (gain < MIN_GAIN). Blend weight: long 0.12, short 0.
## 1014-161921 — idea #1b: Voxtral's audio encoder (Whisper-large-v3) layers 16-32
- cv_testmix 0.4990 (−0.0008, 0/3 seeds) → **rejected**. VOXTRAL back to ().
- lesson: the leaderboard leaders' strongest view is strong here too, but on its own; our stack already combines the same
  information (text pair + speech encoders + trees), so it adds little. Voxtral helps long clips, not the short ones the test
  is mostly made of.
## 1014-164335 — Voxtral head re-tested under MIN_GAIN=0 (Amrita set it on 2026-10-14) → **ACCEPTED**
- identical to 1014-161629: cv_testmix 0.4978 (+0.0004, 2/3 seeds), cv_grouped 0.4727, short 0.5151, long 0.4586,
  cv_newtopic 0.5012. New loop state = G11 + VOXTRAL=("voxtral_pred",). To be confirmed on fresh fold seeds (7 99 555).
- Voxtral submitted (1014-164335, Amrita's call, budget raised to 4 by her): **public 0.3429** (best stays 0.3426).
- fresh-seed check (7 99 555): without Voxtral 1014-164642 cv_testmix **0.4967**; with Voxtral 1014-165010 **0.4971**
  (short 0.5131 vs 0.5116, long 0.4611 vs 0.4634, grouped 0.4740 vs 0.4753). The +0.0004 didn't hold: Voxtral consistently
  helps long clips and slightly hurts short ones, which the test is mostly made of. The public score agrees.
## 1014-165257 — idea #4 gate A: Laya checklist features in the families (on G11 + Voxtral) → **ACCEPTED**
- zero-shot checklist (5,466 sentence units x 8 questions, 11 min on a T4): the six rubric questions mostly fail the checks,
  several with the wrong sign (basic error, incomplete, formulaic rise with the grade: minimal edits are the model's
  documented weakness). The raw-vs-clean inference questions work: same_meaning / grammar_only pass (ρ +0.26 to +0.33) and
  separate CTC failures (same_meaning 0.62 vs 0.75). rub_ctc_ev passes (ρ +0.28); self-correction passes the rubric test on
  long clips (r 0.37). Passing all four checks: chk_same_meaning_mean, chk_same_meaning_max, chk_grammar_only_max, rub_ctc_ev.
- stack: **cv_testmix 0.4954 (+0.0024, 3/3 seeds)**, cv_grouped 0.4712 (−0.0015), short 0.5121, long 0.4577,
  cv_newtopic 0.5011. Passes the old 0.002 rule too.
- checklist confirmations: on fresh seeds (7 99 555) with Voxtral 1014-165727 **0.4949** (vs 0.4971 without the checklist:
  the checklist gain holds, +0.0022). Without Voxtral: main seeds 1014-170108 **0.4939** (3/3 better than with it; grouped
  0.4734, short 0.5082, long 0.4620), fresh seeds 1014-170406 **0.4940**. Both fold sets say dropping Voxtral is better on
  cv_testmix (it helps long clips, hurts short ones).
- **current state: G11 + Laya checklist (LAYA_CHECKLIST, 4 features), VOXTRAL off: cv_testmix 0.4939 (1014-170108).**
- submitted 1014-170108 (G11 + Laya checklist, cv_testmix 0.4939; forced past the advisory grouped-CV guard, at Amrita's
  instruction, budget raised by her): **public 0.3434** (best 0.3426, rank ~47). Its CV beats every submitted run by
  0.004; the public slice doesn't show it (0.3426 / 0.3429 / 0.3434 are within one clip of each other).
- ledger: 1014-170108 verdict set to accepted (Amrita's override of the advisory grouped-CV guard), so later gates compare
  with the current state (0.4939).
## verbatim job (#3) done: Whisper-large-v3-turbo verbatim transcripts (48 min; 2 clips without word times; no prompt leak;
  108 vs 100 words per clip), Qwen hidden states on them (7.7 min), DeBERTa pair (verbatim, clean): alone 0.5841 (CTC pair 0.5611)
## 1014-175304 — idea #3a: Qwen heads on the verbatim transcript (qwen_verbatim_pred), on G11 + checklist
- cv_testmix 0.4934 (+0.0005, but better on 1/3 seeds), grouped 0.4731, cv_newtopic 0.5008; blend weight 0 → **rejected**.
## 1014-175542 — idea #3b: DeBERTa on (verbatim, Whisper clean) (deberta_vpair_pred, alone 0.5841), on G11 + checklist
- cv_testmix 0.4950 (−0.0010, 1/3 seeds), grouped 0.4738 → **rejected**. But **cv_newtopic 0.4967** (current state 0.5011):
  the largest gain on held-out prompts so far; it takes 0.21 of the short-clip blend weight. A hedge candidate for the
  private board's new prompts, not a cv_testmix gain.
- lesson: verbatim large-v3-turbo text keeps fewer of the speaker's errors than the CTC raw text (pair alone 0.584 vs 0.561);
  its value is topic robustness, not accuracy.
## Laya decision-head fine-tune (#5) done: 213 min on a T4, 5 folds, none collapsed; OOF alone thresholds 0.6063, score 0.6097
## 1015-064553 — idea #5: laya_dh_thr_pred as a member, on G11 + checklist → **rejected**
- cv_testmix 0.4979 (−0.0040 vs 0.4939). Weaker alone than DeBERTa-pair (0.561) and it adds nothing the stack lacks.
- 2026-10-15: Amrita submitted the three latest runs (forced): verbatim Qwen 1014-175304 **public 0.3414** (new best public;
  CV 0.4934, lowest mean CV but better on only 1/3 seeds), verbatim DeBERTa 1014-175542 0.3444, Laya fine-tune 1015-064553 0.3433.
  All within ~0.003 of each other: public-slice noise. Final-pick candidates: 1014-170108 (best confirmed CV) + 1014-175304 (best public, lowest mean CV).

# Phase I — closing the gap (2026-10-15 → 17; Amrita: full freedom, only licence + no-external-data rules hard)
## research: what the teams below 0.34 do (READMEs: Techy-A 0.3116, PRINCE2-AI 0.3262, nakulgrover118 0.3211, #1 0.3064)
- public metric decoded by Techy-A with shift/scale probes: **0.6 × RMSE + 0.4 × (1 − Pearson)**.
- their CV is no better than ours (#1 0.505 / r 0.867 on 732 clips; ours 0.473 plain), so the gap is test handling:
  bias removal from ±shift probes (0.3350 → 0.3262), stretch of full-length test preds ×1.15 (→ 0.3241), training on test-length
  crops incl. crop transcripts (0.3454 → 0.3350), Voxtral (+ attention pooling over its frames on random crops), a separate
  stacker for the 45 s batch (PRINCE2-AI 0.339 → 0.326), pooling a speaker's own test clips (nakulgrover118), averaging diverse files.
## within-test speaker pooling (simulated on OOF of 1014-170108; grouped folds keep twins in one validation fold, like test)
- test: 41 of 216 clips in 17 same-voice groups (x-vector cos > t = 0.9882), sizes 2–3, adjacent file IDs (as train twins).
- p' = (1 − s) p + s · group mean: s = 0.5 → cv_testmix 0.4939 → 0.4917, plain 0.4734 → 0.4658, 3/3 seeds; twin clips 0.433 → 0.417.
  Twin residual corr 0.24. Adjacent-ID train pairs with cos 0.97–0.988 still agree (|Δlabel| 0.31–0.43 vs 1.16 random).
## ft_wavlm_pred is leaky and over-weighted (key finding)
- RMSE on clips whose speaker has a twin in train 0.462 vs no-twin clips **0.646** (gap +0.184; leak-free grouped members ~+0.04).
  Test speakers are all new, so on test it behaves like the 0.646 column, yet the long-batch blend gave it 0.40.
- leak-free yardstick = the 343 no-twin training clips. With ft_wavlm (1014-170108) no-twin testmix 0.5172 (long 0.5142);
  without (1015-070206) **0.5179 (long 0.5065)**. All-clip cv_testmix 0.4939 → 0.5140: the 0.02 was the leak.
- honest base = DROP_FT_WAVLM: 1015-070206 cv_testmix 0.5140 / cv_newtopic 0.5177. + Voxtral 1015-070513: 0.5133 (long 0.5063,
  short 0.5166 worse).
## bias probes prepared: runs/probes/shift-0.25 / -0.50 of 1014-175304 (public 0.3414); with S0 they solve bias, RMSE, Pearson
  (simulated precision ±0.001 on the bias at 4-decimal scores).
## bias probes (submitted 2026-10-15 by Claude, Kaggle submit permission granted by Amrita): 1014-175304 shifted −0.25 → 0.3639, −0.50 → 0.4462
- with S0 0.3414 and public = 0.6 RMSE + 0.4 (1 − r): **public RMSE 0.488, Pearson 0.879, mean(pred − true) +0.049** (rounding range
  bias 0.048–0.050). Optimal shift −0.049 would give 0.3399 (−0.0015): calibration is not our gap. The #1's 0.3064 implies public
  RMSE ≈ 0.44 at a CV of 0.505; ours is 0.488 at a (leaky) CV of 0.473.
## honest re-tests, batch 1 (vs honest base + Voxtral 1015-070513, cv_testmix 0.5133 / no-twin 0.5217 / newtopic 0.5199)
| change | run | cv_testmix | no-twin | newtopic |
|---|---|---|---|---|
| + RNC | 1015-071145 | **0.5103** | 0.5184 | 0.5223 |
| + SVR head WavLM-large L20 | 1015-071451 | 0.5125 | 0.5200 | 0.5213 |
| + Qwen verbatim heads | 1015-072818 | 0.5131 | 0.5210 | 0.5221 |
| + DeBERTa vpair | 1015-073039 | 0.5123 | 0.5194 | **0.5180** |
| + Laya heads | 1015-073437 | 0.5132 | 0.5204 | 0.5213 |
| + Laya ft pair | 1015-073702 | 0.5151 | 0.5245 | 0.5203 |
| − Laya checklist | 1015-073930 | 0.5149 | 0.5223 | 0.5203 |
| + Voxtral encoder head | 1015-074201 | 0.5129 | 0.5207 | 0.5208 |
| + crop30 heads | 1015-074436 | 0.5130 | 0.5204 | 0.5224 |
- lesson: under honest validation every addition is worth ≤ 0.003; RNC (rejected under the leaky gate) is now the best one.
## other checks
- same-speaker pooling on the honest RNC OOF: hard threshold s=0.3 −0.0015, cos ramp from 0.98 s=0.3 −0.0023 (train); test has
  far fewer twins (45–57 clips vs 456–509/732), so ≈ −0.0005 expected on test. Kept as POOL_S (src/pooling.py), strict threshold.
  Test x-vectors are more alike overall (142 test clips have a neighbour > 0.96), so a loose threshold would pool different speakers.
- Voxtral last-prompt-token state (heard audio + read question): ridge 0.580 alone (audio-token mean 0.538), concatenation 0.5465 → no.
- crop augmentation of the WavLM-large head with the first-30-s crops (cache/crop30_*): crop layer 24 is on another scale (no final
  layer norm in the crop job), and RidgeCV's LOO is fooled by near-duplicate crops. With layers ≤ 23 and the alpha from full clips:
  neutral (0.6020 → 0.6035; weight 0.5: 0.6002). Short clips don't improve: their difficulty is the recording batch, not length.
- the exact-45.06 s recordings: train 86 (mean label 2.91, ALL singletons), test 84 (39% of test); other short clips train 88 (mean
  ~3.3), test 63. OOF bias: b45 +0.085 (pred sd 0.57 vs label sd 0.82), other short singletons −0.091. Three-way NNLS blend
  (BLENDER="nnls_batch3"; run.py generalised to any blend groups, reproduces 1015-071145 exactly): 1015-075726 0.5105 (vs 0.5103),
  newtopic 0.5262 (worse) → no. The per-batch stacker other teams added is the one we already have.
## GPU jobs pushed by Claude (kernel push permission granted by Amrita 2026-10-15)
- amritarajput54/shl-voxtral-frames v1 (kaggle/voxtral_frames): view-A frames (LM layers 12–14, PCA-256 on train frames) for the
  attention-pooling member (src/vox_attn.py, VOX_ATTN), + question-first views B (rubric) and C (error checklist), ~1 h on one T4.
- amritarajput54/shl-deberta-large-pair v1 (kaggle/deberta_large_pair): DeBERTa-v3-large on (CTC raw, Whisper clean), grouped
  seed-42 folds, lr 1e-5 + LLRD 0.9, final epoch, collapse guard on prediction spread only, seeds 42/7/2024 over both T4s.
## Voxtral frames job done (shl-voxtral-frames v1, 51.9 min on a T4): all states finite; first audio token A 3 / B 102 / C 40
  (question first confirmed); frames 640,427 × 256 (PCA on 150k of 508,774 training frames, 75.4% of variance).
- standalone ridge (3 grouped seeds, plain / testmix / short / long): A 0.5378 / 0.5462 / 0.5522 / 0.5332; B (rubric first)
  0.5366 / **0.5390 / 0.5408** / 0.5353; C (errors first) 0.5377 / 0.5428 / 0.5465 / 0.5349; mean of inputs 0.5365 / 0.5416;
  **mean of the three predictions 0.5315 / 0.5362** (views correlate 0.994–0.998). → VOXTRAL3 member (voxtral3_pred).
## honest batch 2 (RNC base 1015-071145 0.5103 unless noted)
| change | run | cv_testmix | newtopic | no-twin |
|---|---|---|---|---|
| + DeBERTa vpair | 1015-080442 | 0.5099 | 0.5193 | 0.5170 |
| + vpair + SVR head | 1015-080734 | **0.5088** | **0.5178** | **0.5156** |
| + Qwen2.5 judge (CTC EV + entropy) as family inputs | 1015-081017 | 0.5109 | 0.5234 | 0.5187 |
| + llm_err_per100 | 1015-081256 | 0.5108 | 0.5241 | 0.5186 |
| blend on speaker-grouped OOF (G11 off) | 1015-081537 | 0.5089 | 0.5185 | 0.5186 (seed spread 0.0190 vs 0.0093; per-seed mean worse) |
- the Qwen2.5 judge's log-odds (Whisper / verbatim views) correlate +0.13 with the honest residuals: JUDGE_MEMBER / JUDGE_PLUS
  flags added (batch 3). Delexicalised TF-IDF members (src/delex.py): 0.74–0.80 alone, topic gap still +0.05–0.06 → not used.
## GPU: shl-deberta-large-pair v1 OOM at batch 8 (T4) → v2: batch 4 × 4 accumulation + gradient checkpointing. shl-qwen35-judge v1
  pushed (Qwen/Qwen3.5-9B, Apache-2.0, NF4 on each T4: rubric + accuracy prompts × verbatim / CTC / Whisper; digit probabilities
  + answer-position states at 50/65/80% depth).
## GPU results (2026-10-15 afternoon)
- shl-qwen35-judge v1 (Qwen/Qwen3.5-9B NF4, 2 T4s, 27 min): rubric + accuracy × verbatim / CTC / Whisper; a digit was the top
  token 100% of the time; calibrated (mean max-prob 0.44–0.53 vs Qwen2.5's ~0.99). Spearman with the grade up to **+0.734**
  (Whisper, rubric) vs Qwen2.5 +0.677.
- shl-deberta-large-pair v2 (batch 4 × 4 accumulation + checkpointing, 66 min): no collapse; OOF 0.5637 / 0.5551 / 0.5552 per
  seed, **3-seed mean 0.5512** (base pair 0.5611).
- shl-qwen35-judge2 v1 (43 min): complexity / completeness / CEFR questions, and the rubric + CEFR on two transcripts side by side.
## honest batch 5 (vs RNC + vpair + SVR 1015-080734: 0.5088 / newtopic 0.5178 / no-twin 0.5156)
| change | run | cv_testmix | short | newtopic | no-twin |
|---|---|---|---|---|---|
| + Qwen3.5 judge member (12 numbers) | 1015-094625 | 0.5097 | 0.5146 | 0.5214 | 0.5172 |
| **+ Qwen3.5 answer-state heads ×3** (alone 0.613–0.623) | 1015-101517 | **0.5069** | **0.5099** | 0.5224 | **0.5142** |
| Voxtral A → 3-view member | 1015-102036 | 0.5086 | 0.5128 | **0.5183** | 0.5151 |
| + Qwen2.5 judge member | 1015-102504 | 0.5107 | 0.5155 | 0.5196 | 0.5180 |
| + pooling s=0.3 | 1015-110900 | 0.5081 | 0.5137 | 0.5178 | 0.5156 |
- the raw residual correlation (+0.14) overstated the judge: shrunk predictions make any grade-correlated feature correlate with the
  residual. Partial correlation with the grade GIVEN the stack prediction: round-2 complexity +0.11–0.12, CEFR +0.10, two-transcript
  rubric +0.10, round-1 rubric (Whisper) +0.09, CTC views +0.01–0.06. A ridge on all judge numbers + the prediction predicting the
  residual, fitted inside grouped folds: plain RMSE 0.5028 → **0.4976** (seed 42) → RESID_CORR in run.py (batch 7).
- NNLS can't use them as members (non-negative weights, collinear with the members); a residual stage can.
## ops notes: TaskStop leaves Git-Bash batch loops alive on Windows (a stopped batch launched a stray run); stop by command line.
  Local torch training starves when run.py runs in parallel (vox_attn got ~1 core for 3 h) → moved to a T4 (shl-vox-attn-gpu,
  folds/target/groups embedded so they match the stack exactly).
## honest batches 6–8 (2026-10-15 evening)
| run | config (on RNC + DeBERTa vpair + SVR head unless noted) | cv_testmix | short | long | newtopic | no-twin |
|---|---|---|---|---|---|---|
| 1015-111200 | + DeBERTa-large pair (added) | 0.5085 | 0.5146 | 0.4953 | 0.5156 | 0.5182 |
| 1015-111426 | DeBERTa pair base → large | **0.5060** | 0.5108 | 0.4956 | **0.5123** | 0.5143 |
| 1015-111731 | + Qwen3.5 heads + RESID_CORR | 0.5052 | 0.5097 | 0.4953 | 0.5224 | 0.5146 |
| 1015-112103 | + Qwen3.5 heads + RESID_CORR, 3-view Voxtral | 0.5043 | 0.5088 | 0.4946 | 0.5228 | 0.5135 |
| 1015-112433 | large + Qwen3.5 heads | 0.5060 | 0.5105 | 0.4963 | 0.5148 | 0.5150 |
| 1015-112805 | large + Qwen3.5 heads + 3-view Voxtral | 0.5052 | 0.5094 | 0.4962 | 0.5155 | 0.5140 |
| **1015-113131** | + RESID_CORR (21 judge numbers) | **0.5037** | 0.5093 | 0.4915 | 0.5155 | 0.5145 |
| **1015-113504** | + pooling s=0.3 | **0.5021** | 0.5092 | 0.4866 | 0.5155 | 0.5145 |
- today on the honest yardstick: 0.5133 → 0.5037 (0.5021 with pooling; pooling's test gain will be smaller, test has few twins).
  newtopic 0.5199 → 0.5155. Qwen3.5 heads add nothing once DeBERTa-large is in (both text).
- vox_attn on a T4 (shl-vox-attn-gpu, 11 min): OOF 0.5896 / 0.5756 / 0.5893, topic 0.5981 — weaker alone than the Voxtral ridge
  (0.538); stored under the cache key (VERSION vox-attn-v1-gpu2); stack test in batch 9.
- GPU: shl-deberta-large-vpair v1 pushed (large on (verbatim, Whisper clean), same recipe).
## honest submissions (2026-10-15 ~17:25 IST; Amrita allowed forced submits)
- 1015-113504 (best honest combo + RESID_CORR + pooling, cv_testmix 0.5021): **public 0.3371**
- 1015-070513 (honest base: no ft_wavlm + Voxtral A + checklist, cv_testmix 0.5133): **public 0.3344** (best ever; rank ~40)
- lesson: dropping the leaky ft_wavlm transferred (0.3414 → 0.3344, −0.007). The day's member/blend additions (−0.011 honest CV)
  did not show on public (+0.0027, within one clip of noise). Treat composition gains as unproven; prefer procedure fixes.
## fresh-seed confirmation (7/99/555): honest base 1015-114531 0.5100 (short 0.5112, long 0.5074); best combo 1015-120014 0.5061
  (short 0.5118, long 0.4937); + pooling 1015-120329 0.5044. On fresh folds ALL of the combo's gain is on long clips; the main-seed
  short gain (0.5166 → 0.5093) was fold luck. That explains public (68% short): base 0.3344 vs combo 0.3371.
- averaging the two pipelines on short clips (long = combo): fresh 0.5031 (short 0.5075), main 0.5038 → robust on both fold sets.
- **submitted hybrid** (runs/probes/hybrid_070513_113504.csv: short = ½ 1015-070513 + ½ 1015-113504, long = 1015-113504):
  **public 0.3309** (best; was 0.3344). Direct kaggle submit (Amrita allowed forced submits); not a ledger run.
## short blend shrunk toward long (BLENDER "nnls_batch_shrink", SHRINK_LAMBDA 0.25)
- main seeds: combo 0.5037 → 1015-120724 **0.5024** (short 0.5074); + pooling 1015-121105 0.5008. Fresh seeds: 1015-120014 0.5061 →
  1015-121446 **0.5050** (short 0.5102) → holds (modestly).
- hybrid with the shrunk combo (short = ½ base + ½ shrunk combo, long = shrunk combo): fresh 0.5026, main 0.5031 (hybrid 1:
  0.5031 / 0.5038). **Submitted hybrid2** (runs/probes/hybrid2_070513_121105.csv): **public 0.3297** (hybrid 1 0.3309).
## pushed everything good (Amrita: run freely, submit all good performers), 2026-10-15 night
| submission | public |
|---|---|
| 1015-121105 shrink combo + pool | 0.3360 |
| 1015-120724 shrink combo | 0.3377 |
| 1015-163241 + DeBERTa-large vpair | 0.3412 |
| 1015-114248 combo + vox_attn + resid | 0.3363 |
| 1015-113901 combo + vox_attn | 0.3401 |
| hybrid3 short avg(base, shrink+pool, vox_attn+resid), long shrink+pool | 0.3307 |
| hybrid4 / hybrid6 (4-pipeline short avg) | 0.3327 / 0.3335 |
| hybrid5 short avg(base, shrink+pool), long avg of 3 combos | 0.3305 |
| hybrid8 all clips avg(base, shrink+pool) | 0.3328 |
| **hybrid7** seed-bagged: short avg(base 070513+114531, combo 121105+121446), long combo bag | **0.3283** (best, was 0.3297) |
- lessons: the baseline is the key partner on short clips (more combo variants dilute it); bagging over fold seeds helps.
  Running: base + combo on seeds 11/22/33 and 44/66/88 for a larger bag.
- bagging runs: base seeds 11/22/33 1015-165432 (0.5192), 44/66/88 1015-172136 (0.5134); combo seeds 11/22/33 1015-170722
  (0.5045), 44/66/88 1015-173446 (0.5051). **hybrid10** (4-seed-set bags, hybrid7 recipe): **public 0.3280** (rank ~23).
- GPU running: shl-deberta-large-pair-s2 (3 more seeds of DeBERTa-large on the CTC pair, for a 6-seed column).
## 2026-10-16: 6-seed DeBERTa-large pair (shl-deberta-large-pair-s2 seeds 11/22/33, 0.5527; 6-seed column 0.5503 vs 0.5512)
- combo re-run on 4 seed sets with it: 1016-100139 0.5011, 1016-140135 0.5022, 1016-140601 0.5038, 1016-141007 0.5052.
- **hybrid11** (hybrid10 recipe with these combos): **public 0.3279** (best by 0.0001; rank ~24 of 447). Bagging/seed gains saturated.

# Phase I final round (2026-10-16, Amrita: one substantive short-clip idea, timeboxed to today)
## what goes wrong on short clips — diagnosis first
- ASR reliability is NOT the cause (new cache/asr_reliability.csv: Whisper/verbatim segment avg_logprob, no_speech_prob,
  temperature fallback, word-probability mean and low-share, CTC word scores). Partial r with the grade given the stack's
  prediction: |r| <= 0.055 everywhere, and short clips are recognised BETTER than long (w_logprob −0.250 vs −0.288). The 45.06 s
  batch is no worse than other short clips (−0.250 vs −0.251).
- the zero-shot examiner (Qwen3.5 judge rounds 1+2) is where the unused information sits, and it is almost ENTIRELY a short-clip
  signal. Partial r with the grade given the stack's prediction: long ~0.00–0.04, short +0.07–+0.17, **exact-45.06 s batch
  +0.11–+0.31** (strongest: whisper complexity, verbatim completeness, verbatim accuracy, two-transcript CEFR).
## idea: give the examiner scores their own slope on the 45.06 s batch (RESID_B45)
- offline on the saved candidate matrices: judge-only correction main 0.5024 / fresh 0.5050 → with the b45 interaction
  **0.5015 / 0.5018** (short 0.5050 / 0.5047), better on both fold sets. Full runs: 1016-145221 **cv_testmix 0.5005** (short
  0.5059), 1016-145539 **0.4994** (short 0.5033) — the best honest CV of the project.
- **submitted hybrid12: public 0.3311** (hybrid11 0.3279) → WORSE.
- diagnosis (src-level blind spot found): cv_newtopic is computed BEFORE the residual-correction stage in run.py, so neither
  RESID_CORR nor RESID_B45 was ever checked against prompt shift. Simulating it (fit the correction outside topic T, apply to T):
  | correction | newtopic testmix | short | b45 | long |
  |---|---|---|---|---|
  | none | 0.5160 | 0.5209 | 0.5386 | 0.5055 |
  | judge numbers | 0.5164 | 0.5235 | **0.5309** | **0.5011** |
  | + b45 interaction | **0.5178** | 0.5248 | **0.5402** | 0.5024 |
- so the b45 slope is fitted on 86 training clips whose prompts (topics 7/1/5/3) are not the test b45 prompts (8/6/4/5); it
  helps when prompts are shared and hurts when they are not. Speaker-grouped CV cannot see this; the public board can.
- **lesson for the report: with 86 clips in a batch, a batch-specific correction learns the batch's PROMPTS, not its recordings.**
  Procedure changes that transferred (leak removal, seed bagging, averaging two pipelines) were all prompt-independent.
## clean A/B of the examiner (judge) correction — matched bagging, one change (2026-10-16)
| version | cv_testmix main / fresh | newtopic (sim) | public |
|---|---|---|---|
| combo, 2 seed sets, WITH RESID_CORR (hybrid14) | 0.5011 / 0.5022 | 0.5164 | **0.3280** |
| same, WITHOUT RESID_CORR (hybrid13) | 0.5037 / 0.5055 | 0.5160 | 0.3335 |
| same, WITH RESID_CORR + b45 slope (hybrid12) | 0.5005 / 0.4994 | 0.5178 | 0.3311 |
- the examiner correction is worth **0.0055 public** and CV agreed → keep it in the final pick.
- the batch-specific slope costs 0.0031 public at matched bagging → dropped (RESID_B45 stays False), consistent with the
  prompt-shift simulation. One sentence for the report: the examiner's GRADE generalises; a per-batch SLOPE on 86 clips does not.
- hybrid14 (2 seed sets) 0.3280 ≈ hybrid11 (4 seed sets) 0.3279 → bagging beyond 2 fold-seed sets adds nothing measurable.
## error profile (shl-qwen35-judge3, 50 min on 2 T4s): counts of each error type, self-corrections, structure range
- all 11 question x transcript passes returned a digit; counts behave sensibly (per clip: articles 3.5, verb form 3.0, word order
  2.8, prepositions 2.9, fragments 2.7, agreement 1.8, self-corrections 1.4, structure range 2.1).
- each count correlates with the grade on its own (Spearman −0.29 to −0.47; structure range **+0.57**), but given the stack's
  prediction almost nothing is left: partial r −0.06 to +0.01, except structure range +0.087.
- in the correction (global, no batch slope): judge rounds 1+2 alone 0.5028 / 0.5038 → **with the profile 0.5046 / 0.5059**
  (short 0.5081 → 0.5105 / 0.5084 → 0.5115); profile alone 0.5102 / 0.5084. **Rejected on CV, not submitted.**
- lesson: 22 itemised counts dilute the ridge on 732 rows; the examiner's single overall GRADE carries what the stack lacks, and
  asking it to itemise its reasons adds noise, not information. Only "range of complex structures" is partly independent —
  the one rubric descriptor (L5 complex structures) our features never measured (Phase B clause features had the wrong sign).
- follow-up: adding ONLY the structure-range (and/or self-correction) count to the correction is neutral (±0.0003, inconsistent
  across fold sets) → no change. The examiner's overall grade already carries it.
## Phase I round-4 summary (2026-10-16, end of timebox)
- **public best unchanged: hybrid11 0.3279** (rank ~24/447). Nothing from today's round improved it.
- what today added: a diagnosis of the short-clip gap (it is not ASR reliability; it is the examiner signal), a clean A/B
  showing the examiner correction is worth 0.0055 public, and two rejected variants with the reason understood.
- **the validation blind spot found today should be fixed before any further correction work**: run.py computes cv_newtopic
  before the residual-correction stage, so corrections are never prompt-shift tested. src/..." corrsim5-style simulation is the
  check to port in (temp script, not committed).
- final picks (to confirm 2026-10-17): 1) hybrid11; 2) honest base 1015-070513 (public 0.3344) as the diverse hedge.
  Reproduction note for the notebook: hybrid11 averages 8 runs (4 base + 4 combo seed sets); hybrid14 (4 base + 2 combo) scores
  0.3280 and hybrid2 (1 + 1) scores 0.3297, so most of the bag's value is in the first two seed sets.
## 2026-10-17: checks prompted by an external review of the notebook
- metric mismatch: stretching the final OOF around its mean (x0.9 / 1.0 / 1.1 / 1.15 / 1.2) on 86-clip draws at the test mix gives
  board metric 0.3648 / **0.3581** / 0.3586 / 0.3614 / 0.3657 → the least-squares scale is already optimal for 0.6 RMSE +
  0.4 (1 − r) (Pearson ignores scale, and the least-squares direction is the correlation-maximising one).
- private-board simulation (86 clips, test batch mix, 4000 draws, OOF): hybrid11 0.3575 (sd 0.036); base only 0.3666 (P hybrid11
  better 0.90); combination only 0.3576 (0.49); short 25% / 75% base 0.3570 / 0.3590. Best-of-two for pick 2: combination only
  +0.0016, short 25% base +0.0011, base only +0.0004 (OOF cannot see the prompt shift that cost the combination on public).
- **bagged single pipelines submitted**: bag_base (4 seed sets) **0.3334**, bag_combo 0.3338; hybrid11 0.3279 beats both by ~0.0055
  → averaging the pipelines is a real gain. On public the two hedges tie; base is the more different (OOF corr of scores 0.978 vs
  0.993) and hedges both main risks (combination members, examiner correction).
- test CTC failures (repair rate > 0.6): 4 clips, predicted 2.18 / 3.24 / 2.52 / 2.16 (a failure branch would touch ~1.6 private clips).
## examiner round 4 (shl-qwen35-judge4, 2026-10-17): self-consistency + fold-aware few-shot — NOT adopted
- 5 rubric paraphrases x (Whisper, verbatim) + few-shot (6 playground anchors from other seed-42 folds) on Whisper; a digit was the
  top token 100% of the time. EXAMINER variants in experiment.py; offline replica of run.py (verified exact vs 1016-100139).
| variant | cv_testmix main / fresh | topic check after correction main / fresh |
|---|---|---|
| old (round-1 prompt) | 0.5011 / 0.5022 | 0.5149 / 0.5174 |
| sc (mean + spread of 6 prompts) | 0.5016 / 0.5022 | 0.5156 / 0.5182 |
| sc_fs (+ few-shot in the Whisper mean) | 0.5013 / 0.5020 | 0.5145 / 0.5171 |
| fs (few-shot replaces Whisper rubric) | 0.5010 / 0.5028 | 0.5149 / 0.5167 |
- pre-set rule (improve both speaker-grouped CV and the post-correction topic check) met by none → not submitted; pick unchanged.
- lesson: the paraphrased grades agree closely, so averaging removes little noise; few-shot anchors help slightly under prompt shift.
## public-tuned blend (Amrita chose to try it, 2026-10-17; Techy-A's averaging identity extended to the board metric, src/pubmix.py)
- predictor: anchor 1014-175304 (probes: public RMSE 0.4881, r 0.8787, bias +0.049); each of 32 submitted files gets one unknown
  (its error's covariance with the anchor's error on the public clips), solved from its public score; any blend's public score
  is then predicted. Check: hybrid8 (= ½ 1015-070513 + ½ 1015-121105) predicted 0.3328, actual 0.3328.
- optimum over all 32 files (non-negative, sum 1): 87.8% hybrid11 + 8.4% shift-probe(−0.50) + 3.8% 1011-190112, predicted
  0.3266. Decomposition: models only (no probes) 0.3277; hybrid11 shifted by a constant: best −0.041 → 0.3268 → **the gain is
  a bias correction** (public bias of hybrid11 ≈ +0.041, same sign as the anchor's +0.049); our files are too alike to gain from
  mixing.
- **submitted: pubmix_a 0.3270 (predicted 0.3266), pubmix_c = hybrid11 − 0.041: 0.3271 (predicted 0.3268).** Predictor within 0.0004.
- caveat: fitted to the ~130 public clips; the bias estimate is ≈ 1 standard error from zero (130-clip mean), so on the private
  86 clips the shift is worth about ±0.001.
## final notebook pushed and verified on Kaggle (2026-10-17)
- artefacts dataset amritarajput54/shl-grammar-final-artefacts new version (52 files, 36 MB: 8 runs' candidates / OOF / submissions,
  judge files, folds for 12 seeds); notebook amritarajput54/shl-grammar-scoring-final v7: all 8 replays equal to the runs' files
  (≤ 3.6e-15), TRAINING RMSE 0.4800, cv_testmix 0.5013, submission identical to pubmix_c at %.10f, 36.9 s on CPU.
- final picks: 1) pubmix_c (hybrid11 − 0.041, public 0.3271); 2) bag_base (public 0.3334). Amrita selects them on Kaggle.
## external training data (Amrita asked, 2026-10-17): not usable under the commercial-licence rule
- Speak & Improve Corpus 2025: licence for non-commercial research and educational purposes only; releasing derived items
  (models) needs prior approval of Cambridge University Press & Assessment.
- ICNALE: the site's Terms of Use grant no licence (no purpose stated, no commercial permission), prohibit redistribution, and
  require registration; copyright reserved by Dr. Shin Ishikawa. Labels are speakers' test levels on two fixed topics.
## per-batch audio normalisation (BATCH_NORM_AUDIO, src/batchnorm.py), 2026-10-17
- combo main 1017-072306 0.5037 (was 0.5011), fresh 1017-072943 0.5077 (0.5022); base main 1017-072554 0.5196 (0.5133), fresh
  1017-073223 0.5197 (0.5100); topic check after correction 0.5200 (0.5149) → worse on every check.
- Amrita overruled the rejection: hybrid_bn (same recipe and bagging as hybrid14) submitted: **public 0.3342** (hybrid14 0.3280).
- lesson: the recording-batch offset in the embeddings is partly signal here (the 45 s batch is graded lower), not nuisance.
## final picks revised (2026-10-17, Amrita, after an outside review): 1) hybrid11 (0.3279), 2) pubmix_c (0.3271)
- reason: CV can't judge a constant offset (the model is unbiased on train by construction); the two bias estimates share the same
  ~130 public clips (≈1 SE); hybrid11 matches the notebook's validation-first argument; pubmix_c as pick 2 hedges the test-bias
  question at no cost (Kaggle keeps the better of the two). Topic check for LEACE: topic alone predicts the grade with OOF R² 0.039.
- notebook v9 on Kaggle (v8 failed: it mounted the previous artefacts version before the new one was ready): submission.csv =
  hybrid11 and submission_pick2_offset.csv = pubmix_c, both verified (max |diff| 5e-11 at the 10-decimal format).
## LEACE topic erasure on the text heads (LEACE_TEXT, src/leace.py), 2026-10-17
- topic-only model predicts the grade with OOF R² 0.039 (between-topic share 5.6%), so erasing topic removes little grade.
- combo main 1017-080549 0.5023 (was 0.5011), fresh 1017-083114 0.5040 (0.5022); short 0.5099 / 0.5113 (0.5079 / 0.5085);
  **topic check after the correction 0.5121 / 0.5118 (0.5149 / 0.5174)**. Base: worse everywhere (0.5190 / 0.5157).
- trade-off: a little accuracy on familiar prompts for robustness to unseen ones; fails the pre-set both-checks rule.
- board tie-breaker, matched to hybrid14 (base bag 4 sets + combo 2 sets): **hybrid_leace 0.3272** vs 0.3280 (−0.0008, < 1 clip).
## sanity checks from an outside review (2026-10-17): shuffled labels, independent rebuild, train vs test per member
- **shuffled labels, stacking level** (all 8 runs of hybrid11 replayed with the label permuted; members kept): hybrid11 1.064
  vs null (training-fold mean) 1.029, mean of 5 permutations; base bag 1.193, combo bag 1.040; speaker-group-preserving
  permutation the same (1.066 vs 1.031). Nothing below the null. Control: the seed-42 base blend fitted on all rows and scored on
  them gives 1.176 vs 1.194 refit inside folds → the test sees in-sample fitting; nesting removes it. (Above the null because
  the members still predict the true grade and NNLS has no intercept; only the correction ridge can pull toward the mean.)
- **independent rebuild** of wavlm_large_pred from its description (numpy SVD ridge with exact leave-one-out alpha, own
  StratifiedGroupKFold folds, own layer z-scoring and denoised target; no project code): folds identical to folds_all.csv on
  all 6 fold seeds; OOF and test predictions equal to the runs' to < 1e-4 (e.g. seed 42 OOF RMSE 0.5994 vs 0.5994).
- **train vs test per member and batch** (45.06 s / other short / long): every weighted member moves with the consensus (median
  test−train shift +0.03 / −0.20 / +0.14; sd ratios 1.07 / 0.92 / 1.11); hybrid11 itself robust z −0.5 / −0.1 / +0.1. Two
  outliers, both at blend weight 0 in all 8 runs: emb_pred (mpnet text) 0.10–0.17 lower than the others on test in every batch,
  agreement with the other members rising on test (topic content, not a broken test pipeline); qwen35_ctc_pred +0.15 on
  other-short test clips. Neither reaches the submissions.
## paired speaker-group bootstrap (src/paired_bootstrap.py; printed by run.py after every VERDICT), 2026-10-17
- resample the 503 speaker groups, score new and old on the same draw, 10 000 draws; 95% interval of new − old (cv_testmix).
| comparison | new − old | 95% interval | P(new better) |
|---|---|---|---|
| hybrid11 vs base bag (4 sets) | −0.0113 | [−0.0177, −0.0045] | 1.00 |
| hybrid11 vs combo bag | +0.0000 | [−0.0059, +0.0055] | 0.49 |
| combo vs base, per seed set | −0.0122 / −0.0078 / −0.0154 / −0.0082 | each interval ±0.012 | 0.96 / 0.89 / 0.99 / 0.90 |
| … long clips only | −0.020 on every set | excludes 0 on every set | 1.00 |
| … short clips only | −0.009 / −0.003 / −0.012 / −0.003 | each includes 0 | 0.63–0.93 |
| examiner correction on/off (combo bag) | −0.0035 | [−0.0103, +0.0034] | 0.84 |
| pooling 0.3 vs 0 | −0.0015 | [−0.0033, +0.0006] | 0.93 |
| shrink on/off | −0.0012 | [−0.0026, +0.0001] | 0.97 |
| LEACE text, main / fresh | +0.0012 / +0.0018 | [−0.0026, +0.0050] / [−0.0006, +0.0043] | 0.26 / 0.07 |
| batch-norm audio combo, main / fresh | +0.0026 / +0.0055 | fresh [+0.0015, +0.0097] | 0.10 / 0.00 |
- lesson: CV resolves a pipeline-level change (~±0.006–0.012) and a light post-stage tweak (~±0.002), not a 0.001–0.003 member
  or judge change. The combination's gain is real on long clips only; on short clips no change has ever been resolvable. The
  hybrid vs combo-alone choice is invisible to CV (on public it was 0.3279 vs 0.3338, ~130 clips).
- seed-42-only fine-tune columns: the seed-42 set (the only one where they are fully out-of-fold) scores 0.0009 (base) and
  0.0026 (combo) *better* than the mean of the other three sets → the feared optimism on other fold seeds is not visible.
## pairwise anchored examiner (shl-qwen35-pairwise v1, pushed 2026-10-17 ~15:20; Amrita: "start pairwise now")
- job: Qwen3.5-9B NF4, thinking off; each clip vs 8 graded anchors of its own batch (b45 / other short / long; grades 2.0–5.0,
  5.0 twice), "which shows better GRAMMAR, A or B", both orders; next-token P(A) vs P(B); verbatim transcripts. 15.2k passes.
- anchors (kaggle/qwen35_pairwise/make_anchors.py): single-clip speakers (groups never cross train/test), ≥ 50 verbatim words,
  closest out-of-fold hybrid11 prediction to the label at each grade; the short batches share their 4.5 / 5.0 anchors.
- **pre-registered before any result**: feature = pw_theta, the Bradley–Terry grade (MAP, fixed slope 3 per grade point,
  N(3.5, 1.5²) prior) from the order-averaged P(clip better) against the anchors; recomputed inside every fold from that fold's
  training anchors only (topic check: anchors of other topics only; test: all anchors). It enters the combination's examiner
  correction ridge next to the 21 judge numbers (nested in folds); the hybrid is rebuilt over the 4 seed sets.
  **Adopt only if** the paired speaker-group bootstrap 95% interval of (new hybrid − hybrid11) on cv_testmix lies below 0
  AND the post-correction topic check (seed-42 combination run) is not worse by more than 0.001. Otherwise not adopted, whatever
  the point estimate. Diagnostics (not part of the decision): pw_theta's correlation with the grade per batch, its partial
  correlation with hybrid11's out-of-fold residual, position bias, short-clip interval.
- **result (shl-qwen35-pairwise v1, 86 min on 2×T4)**: 15 126 comparisons, top token A/B 100%; position bias toward answer B
  (mean P(A) 0.348 with the clip first, 0.500 with the anchor first; the order average cancels it); order agreement r 0.855.
  pw_theta vs grade: all +0.583, b45 +0.729, short +0.624, long +0.568 (round-1 verbatim judge +0.596 / +0.737 / +0.606 / +0.616).
  Partial correlation with hybrid11's out-of-fold residual +0.039.
  **Pre-registered test: NOT ADOPTED.** hybrid cv_testmix 0.5013 → 0.5013, bootstrap −0.0000 [−0.0001, +0.0001]; topic check after
  the correction 0.5149 → 0.5150.
- why: the 21 judge numbers already in the correction explain R² 0.81 of pw_theta (r 0.83 with the round-1 verbatim judge). Comparative
  judging fixes the judge's *scale* (mean pw_theta on short clips from true 2.0 to 5.0: 2.73 → 4.28, absolute judge 3.17 → 3.71)
  but not its *ranking*, and the correction ridge rescales linearly anyway. Same model + same transcript = same information.
- lesson: the next examiner signal has to come from a different source (a different model family, or the audio), not a different
  question to Qwen3.5-9B on the same transcript.
## sanity check 1b: the whole local pipeline on shuffled labels (every member retrained), 2026-10-17
- wrapper (scratchpad) patches the label read, diverts runs/, the ledger and cache/folds_grouped.csv; the combination's flags
  (never stored with the run: EXP_OVERRIDES) were reconstructed and verified on true labels: 0.5011, per seed 0.5071 / 0.5107 /
  0.5004 = 1016-100139 exactly. Override string: {"RNC_HEAD": true, "SVR_HEADS": true, "DEBERTA_LARGE": "replace",
  "VERBATIM_DEBERTA": true, "QWEN35_HEADS": ["qwen35_verbatim_pred", "qwen35_ctc_pred", "qwen35_whisper_pred"], "VOXTRAL": [],
  "VOXTRAL3": true, "BLENDER": "nnls_batch_shrink", "RESID_CORR": true, "POOL_S": 0.3}
| pipeline | shuffle (corr with true labels) | stack cv_testmix | null (training-fold mean) | stack − null |
|---|---|---|---|---|
| base | global 1 (+0.048) | 1.0307 | 1.0321 | −0.0014 |
| base | global 3 (+0.014) | 1.0456 | 1.0338 | +0.0119 |
| base | global 4 (−0.020) | 1.0138 | 1.0266 | −0.0128 |
| base | global 5 (+0.022) | 1.0136 | 1.0097 | +0.0039 |
| combination | global 1 (+0.048) | 1.0407 | 1.0321 | +0.0086 |
| combination | global 200 (−0.000) | 1.0365 | 1.0293 | +0.0072 |
| combination | group 2 (−0.120) | 1.0542 | 1.0595 | −0.0053 |
- reading: per-permutation scatter ±0.012 (noise of a single permutation on 732 clips); mean stack − null +0.0004 (base, 4) and
  +0.0079 (combination, the two global ones). The true-label CV (0.50) is 0.46 below the null, ~40× that scatter. The one combination
  dip (group 2) is the chance channel: those labels correlate −0.12 with the true grade, and the label-free judge features alone
  (ridge in folds) also beat its null (1.0564 vs 1.0595); the correction stage lands at that level (blend 1.0696 → 1.0551).
- not run: base seeds 6/7 and combination group 105/103 (the machine ran out of process slots running two loops at once; one
  loop at a time from now on). Shuffled-run member caches deleted from cache/; folds_grouped.csv and train.csv unchanged (md5).
## rejection rules lifted (Amrita, 2026-10-17 evening): "overrule every rejection rule ... run whatever is left; push promising findings"
- GPU: shl-phi4-judge v1 (microsoft/phi-4, MIT: round-1 prompts on 3 transcripts + answer-position hidden states; a different model
  family from the Qwen judges) and shl-deberta-perseed v1 (pilot: DeBERTa-v3-base whisper + pair refitted on fold seeds 7/99/555,
  same recipe as the seed-42 columns) pushed ~17:40.
- src/variants.py: replay-based variants of hybrid11 (members untouched; every unchanged replay checked against the 8 runs' oof.csv,
  submission.csv and hybrid11.csv). 4-set hybrids, paired bootstrap vs hybrid11, topic check after the correction (seed-42 runs):
| variant | cv_testmix (Δ) | 95% | topic check | public |
|---|---|---|---|---|
| base_corr: examiner correction also on the base pipeline | 0.5007 (−0.0007; short −0.0010) | [−0.0046, +0.0032] | base 0.5199 → 0.5208 | **0.3270** |
| base_corr_pairwise: + pw_theta in both corrections | 0.5006 (−0.0007) | [−0.0046, +0.0031] | base → 0.5191, combo 0.5150 | 0.3272 |
| exam_sc_fs (round-4 examiner) | 0.5017 (+0.0004) | [−0.0003, +0.0012] | combo 0.5149 → 0.5145 | 0.3279 |
| exam_sc / exam_fs | 0.5016 / 0.5016 | ±0.0006 | 0.5156 / 0.5149 | not submitted |
| pairwise (combo only) | 0.5013 (0.0000) | ±0.0001 | 0.5150 | not submitted (max |diff| 0.005 vs hybrid11) |
- reading: base_corr is hybrid11 with an already-validated stage applied to the other pipeline too; CV and public both move the
  same way (−0.0007 / −0.0009) but each by less than one clip. Best public without a public-tuned shift so far (hybrid11 0.3279).
## Phi-4 examiner (shl-phi4-judge v1, 44 min on 2×T4; microsoft/phi-4, MIT; Qwen3.5 round-1 prompts, 3 transcripts)
- top token a digit: accuracy question 100% / 100% / 99.9% (verbatim / CTC / Whisper); rubric question only 41% / 1.7% / 9.2%
  (digit probabilities renormalised anyway). Judge numbers vs grade +0.54..+0.66 (best: CTC rubric +0.645 / +0.663); R² from the 21
  Qwen numbers 0.66–0.84 (more independent than the pairwise round's 0.81, CTC view least overlapping); partial correlation with
  hybrid11's residual ≤ +0.084. Hidden-state heads (2×5120 dims, ridge, inner speaker-grouped CV): OOF RMSE 0.608–0.612, topic-held-out
  0.624–0.657 (they lean on prompt content).
| variant | cv_testmix (Δ vs hybrid11) | 95% | topic check after correction | public |
|---|---|---|---|---|
| phi4_corr (combo correction + 12 Phi-4 numbers) | 0.5016 (+0.0002) | [−0.0003, +0.0008] | combo 0.5149 → 0.5166 | – |
| phi4_corr_both (base_corr + Phi-4 numbers in both) | 0.5009 (−0.0004) | [−0.0041, +0.0032] | base 0.5199 → 0.5205, combo → 0.5166 | **0.3264** |
| phi4_heads (3 heads as candidates in both pipelines) | 0.5026 (+0.0012) | [−0.0004, +0.0029] | base → 0.5266, combo → 0.5186 | – |
| phi4_all | 0.5021 (+0.0008) | [−0.0030, +0.0046] | base → 0.5301, combo → 0.5194 | – |
- reading: on CV and the topic check Phi-4 adds nothing to base_corr (−0.0004 vs −0.0007, topic worse); on public phi4_corr_both
  0.3264 vs base_corr 0.3270 — opposite direction, both under one clip. Submitted because Amrita lifted the rejection rules; it is
  not evidence that Phi-4 helps.
## per-seed DeBERTa pilot (shl-deberta-perseed v1: queued ~2.5 h, then 96 min on 2×T4; 90 fine-tunes, none failed)
- deberta_pred / deberta_pair_pred refitted on fold seeds 7/99/555 (recipe of the seed-42 columns); standalone OOF 0.590 / 0.565.
  PERSEED_DEBERTA flag (experiment.py) feeds them in as per-seed members (families and blend see the per-seed values).
- base defaults on 7/99/555 reproduce 1015-114531 exactly (1017-163113: 0.5100, per seed 0.5160 / 0.5189 / 0.5051).
- base with per-seed columns 1017-163340: 0.5108 (+0.0008, 95% [−0.0054, +0.0071]); combination 1017-163711 (deberta_pred per seed;
  the pair column is the seed-42 DeBERTa-large): 0.5018 vs 1016-140135 0.5022 (−0.0004, [−0.0014, +0.0006]).
- → the optimism of seed-42-only fine-tune columns on other fold seeds is ≤ ~0.001: the full refit (11 fold seeds × 4 models, ~1 day
  of GPU) would not change a decision. Not run.
- hybrids with these two runs in the 7/99/555 slot: hybrid11_perseed 0.5013 (±0.0000 vs hybrid11, test max |diff| 0.010, not
  submitted); base_corr_perseed 0.5005 (−0.0001 vs base_corr) → **public 0.3273** (base_corr 0.3270).
## 1018-205257 — AUDIO_QUALITY: recording-quality features in the families (backlog "SNR estimate per clip", tested)
- hypothesis: a noisy or band-limited recording degrades the transcripts and embeddings, which reads as weak grammar; label-free
  recording measures (kaggle/audio_quality, CPU job: SNR, noise floor, speech level, activity, clipping, noise flatness, roll-off,
  centroid; 25 ms frames) let the families separate the recording from the speaker.
- data: short batches are much noisier than long (SNR 36–42 vs ~50 dB), and **test short clips are noisier than train short clips**
  (SNR 37.0 vs 42.2 dB, noise floor −63.6 vs −68.1 dBFS; b45 35.8 vs 37.5). Correlation with the grade: activity +0.28, SNR +0.19.
- fast screen (seed 42): reference 1018-204639 0.5071 → 1018-204856 0.5087 (+0.0016; short 0.5151 → 0.5179; topic after the
  correction 0.5149 → 0.5199).
- full (combination flags + AUDIO_QUALITY, seeds 42/1337/2024): **0.5016 vs 1016-100139 0.5011 (+0.0005)**, per seed 0.5087 /
  0.5097 / 0.4995 (better on 2/3), bootstrap +0.0005 [−0.0027, +0.0037]; short +0.0009, long −0.0005; **topic check after the
  correction 0.5149 → 0.5199**. (run.py's gate still compares with the leaky-era 1014-170342, so its VERDICT is uninformative.)
- verdict: rejected (no gain; worse under prompt shift). Flag kept off by default (reproducible), pipeline unchanged; not submitted.
- lesson: recording quality is shifted between train-short and test-short, so a recording→grade association learned on train
  cannot transfer; the noise shift itself is worth knowing for the short-clip weak spot (test short clips are harder recordings).
## 1018-211303 — EMB_PCS 16: speech-embedding principal components as direct inputs to the 9 families (Amrita's "convert the audio into embeddings a wider variety of model families can use")
- hypothesis: the families see each speech encoder only through its prediction, so they cannot combine how the voice sounds with
  the text features; 16 PCs per encoder (WavLM-base, WavLM-large, Whisper-medium; leak-free pooling: per-layer z-score, layer mean;
  PCA on all 948 clips, label-free) let the trees find audio × text interactions. src/emb_pcs.py, flag EMB_PCS (default 0).
- one PC alone carries a lot: pc_whisper_med_3 r = −0.588 with the grade (pc_wavlm_large_3 +0.431).
- fast screen (seed 42): 0.5071 → 1018-210900 0.5048 (short 0.5151 → 0.5123; topic after the correction 0.5149 → 0.5101). Families:
  lgbm 0.531 → 0.517, xgb 0.526 → 0.516, rf 0.507 → 0.500, et 0.507 → 0.503, svr 0.556 → 0.536; ridge / knn / mlp worse; krr breaks
  (fixed gamma; blend weight 0).
- full (combination flags + EMB_PCS 16, 42/1337/2024): **0.4974 vs 1016-100139 0.5011 (−0.0037)**, per seed 0.5048 / 0.5057 / 0.4964
  (better 3/3), cv_grouped 0.4897 (0.4915), short 0.5029 (−0.0050), long 0.4855; bootstrap −0.0037 [−0.0119, +0.0047], P 0.80;
  topic check after the correction 0.5149 → **0.5101**.
- fresh seeds (7/99/555): **0.4969 vs 1016-140135 0.5022 (−0.0053)**, better 3/3; short −0.0065, long −0.0026 [−0.0045, −0.0006];
  bootstrap testmix [−0.0137, +0.0030], P 0.89; topic check after the correction 0.5100 (reference 0.5174).
- verdict: **ACCEPTED** on the project's rule against the honest reference (gain ≥ 0.002, ≥ 2/3 seeds, cv_grouped not worse), held on
  fresh folds and on unseen prompts (run.py's own VERDICT still compares with the leaky-era 1014-170342 and is uninformative).
  Each set's bootstrap interval still includes 0; two independent fold-seed sets agree.
- combination override string now: {"RNC_HEAD": true, "SVR_HEADS": true, "DEBERTA_LARGE": "replace", "VERBATIM_DEBERTA": true,
  "QWEN35_HEADS": ["qwen35_verbatim_pred", "qwen35_ctc_pred", "qwen35_whisper_pred"], "VOXTRAL": [], "VOXTRAL3": true,
  "BLENDER": "nnls_batch_shrink", "RESID_CORR": true, "POOL_S": 0.3, "EMB_PCS": 16}
- lesson: the embeddings' information was reaching the stack only through linear per-encoder heads; giving the nonlinear families the
  raw components is worth more than another head or judge. Next: the base pipeline and the 4-set hybrid (running).
- **4-set hybrids** (8 new runs: combination + EMB_PCS 1018-211303 / 211655 / 212101 / 212434: −0.0037 / −0.0053 / −0.0019 / −0.0032
  vs their references, 4/4 better; base + EMB_PCS 1018-212727 / 213106 / 213459 / 213747: +0.0012 / −0.0011 / −0.0019 / +0.0056,
  mixed → the PCs help the combination only). Replays verified against all 8 runs' files.
| hybrid | cv_testmix | Δ vs counterpart, 95% | public (counterpart) |
|---|---|---|---|
| base_corr_pcs_combo (base unchanged + correction; combination + PCs) | **0.4978** | −0.0028 [−0.0071, +0.0014] vs base_corr | 0.3302 (0.3270) |
| hybrid11_pcs_combo | 0.4985 | −0.0028 [−0.0070, +0.0013] vs hybrid11 | 0.3306 (0.3279) |
| hybrid11_pcs_both / base_corr_pcs_both | 0.4999 / 0.4990 | −0.0014 / −0.0017 (wider) | not submitted |
  topic check after the correction: combination 0.5149 → 0.5101; corrected base 0.5208 → 0.5178 (base + PCs).
- **public worse by ~0.003 (≈ one clip) on both.** Label-free diagnosis: within each recording batch the 48 PCs separate train from
  test far more than the handcrafted features (adversarial AUC b45 0.638 vs 0.432, other short **0.761 vs 0.637**, long 0.758 vs
  0.597), and on the other-short batch the most grade-related PCs sit off the training distribution in the "lower grade" direction
  (pc_whisper_med_3, r −0.588: +0.46 train-sd on test; pc_wavlm_6, r +0.377: −0.57). Same mechanism as AUDIO_QUALITY: test short
  clips are different recordings, CV cannot see that shift, raw components let the trees turn it into grade.
- status: accepted on CV (both fold-seed sets, unseen prompts) but carries a measured covariate-shift risk on the batch that is 68% of
  test → **not proposed as a final pick**; final picks unchanged. Next: keep only the PCs whose within-batch test distribution matches
  train (selection on unlabelled test covariates, no labels), so the trees get the interactions without the recording shift.
## 1018-215356 — EMB_PCS_SHIFT_Z 2.5: keep only the speech PCs without a within-batch train/test mean shift (cycle 3)
- hypothesis: the EMB_PCS gain (cycle 2) lost on public because raw PCs carry the train→test recording shift; keeping only PCs whose
  within-batch test-minus-train mean difference is within 2.5 standard errors in every batch (unlabelled test inputs only) keeps the
  audio × text interactions without the shift.
- pre-set rule: accept only if (1) CV gain ≥ 0.002 over the no-PC reference on main and fresh seeds AND (2) the kept PCs' within-batch
  adversarial AUC is no higher than the 9 handcrafted features' (b45 0.432, short 0.637, long 0.597).
- filter kept 27 of 48 (dropped the two most grade-related, pc_whisper_med_3 r −0.59 and pc_wavlm_6 +0.38). Adversarial AUC kept vs
  all 48: b45 0.554 (0.638), short **0.598** (0.761), long 0.651 (0.758) → condition (2) **fails on b45 and long**: a per-component
  mean filter leaves multivariate shift.
- informational CV (decision already made): fast 1018-214822 0.5048; full 1018-215356 **0.4979** vs no PCs 0.5011 (−0.0032 [−0.0099,
  +0.0037], P 0.82) vs all 48 PCs 0.4974 (+0.0005); topic check after the correction 0.5118 (none 0.5149, all 0.5101). Fresh seeds not run.
- verdict: **rejected** (pre-set condition 2). Flag EMB_PCS_SHIFT_Z kept off by default (reproducible); nothing submitted.
- lesson: the CV gain does not live in the most shifted directions (27 unshifted PCs keep ~90% of it), so stronger shift handling is
  worth building. But erasing the shift outright already failed once (BATCH_NORM_AUDIO: part of the batch offset is real grade
  difference; members agree test short-other clips predict ~0.20 lower), so the next step should down-weight rather than erase:
  importance weighting of training clips by test-likeness (covariate-shift correction) for the PC-using families.
## 1019-153333 — IW_FAMILIES: covariate-shift (importance) weights in the 9 families, with EMB_PCS 16 (cycle 4)
- hypothesis: the PC families learn from all training clips equally; weighting each by how test-like it is within its recording batch
  (src/shift_weights.py: logistic regression train-vs-test on the 48 PCs per batch, out-of-fold p, ratio clipped [0.2, 5], mean 1;
  no labels) should keep the audio × text gain on the test distribution.
- weights: ESS b45 48 of 86, **other short 28 of 88**, long 257 of 558 (the short batch is strongly shifted).
- new yardstick, importance-weighted CV (validation clips weighted by the same ratios on top of the batch test-mix weight), on the
  EXISTING runs: 48 PCs vs none, plain −0.0037 / −0.0053 (main / fresh) but importance-weighted only −0.0016 / −0.0013; the 27
  shift-filtered PCs (cycle 3) −0.0032 plain but **−0.0048 importance-weighted** (main) → the full set's gain sits mostly on
  train-like clips, consistent with its public loss. Weights come from the same PCs and the short ESS is 28: noisy.
- pre-set rule: accept if importance-weighted cv_testmix improves ≥ 0.002 over the unweighted PC run on main AND fresh, and plain
  cv_testmix is not worse than the no-PC reference.
- fast 1019-152715 0.5050; full main 1019-153333 plain 0.4993 (PCs 0.4974), IW 0.4992 (PCs 0.4979): **+0.0013** [−0.0026, +0.0051];
  fresh 1019-153836 plain 0.4973 (0.4969), IW 0.4954 (0.4959): −0.0005. Topic check after the correction 0.5069 (PCs 0.5101).
- verdict: **rejected** (pre-set rule). Flag kept off; nothing submitted.
- lesson: down-weighting train-like clips costs more variance (ESS 326 of 732) than it removes shift; the shift has to be handled in
  the inputs (which components), not the sample weights. Importance-weighted CV is worth keeping as a second yardstick for any
  audio-derived input.
## 1019-155122 — cycle 5: the 27 shift-filtered speech PCs re-tested under importance-weighted CV (fresh seeds)
- hypothesis: the shift-filtered PCs had the largest importance-weighted gain on main seeds (−0.0048), so they may transfer to test
  where the full 48 did not. Re-test of cycle 3's rejected change under a yardstick adopted afterwards → fresh seeds decide.
- pre-set rule: importance-weighted gain over no PCs ≥ 0.002 on main AND fresh, plain cv_testmix not worse than no PCs.
- main (1018-215356, seen before): plain −0.0032, IW −0.0048 [−0.0132, +0.0035]. **fresh 1019-155122: plain 0.4990 vs 0.5022 (−0.0032)
  but IW 0.4968 vs 0.4972 (−0.0004 [−0.0073, +0.0057]), short clips IW +0.0016**; topic check after the correction 0.5135 (ref 0.5174).
- verdict: **rejected**. Flag stays off; nothing submitted. Third rejection in a row (cycles 3–5) → stopped, per the loop rule.
- what the three rule out: the speech-PC gain is real on plain CV (−0.003 to −0.005 on two fold-seed sets) but every attempt to make
  it hold on test-like recordings fails (per-component filter, importance weights, filtered set on fresh seeds), and the board
  agreed (−0.003 public). Raw audio inputs to the families do not transfer to this test set's recording conditions; local
  audio-shift ideas are spent. The binding constraint is the recording shift of the test short clips (noisier: SNR 37 vs 42 dB).
## KANs and symbolic regression (Amrita, 2026-10-20) — pykan 0.2.8 (MIT) installed in .venv (no pinned deps; numpy / torch unchanged)
- smoke test (synthetic): symbolic recovery exact (tanh(2·x1) + x2² − 0.5·x3); with 44 inputs and 585 rows the KAN overfits after
  ~3 LBFGS steps → design fixed before real data: additive width [n, 1], λ 0.01, steps (≤ 20) chosen on an inner 15% split, refit.
### Cycle A — KAN_FAMILY: additive KAN as a 10th model family (src/kan_family.py), combination flags
- fast 1019-201351 0.5069 (ref 0.5071); main 1019-201900 **0.5009 vs 0.5011** (−0.0002 [−0.0009, +0.0006]); fresh 1019-202613 **0.5010 vs
  0.5022** (−0.0012 [−0.0037, +0.0013]); topic check after the correction 0.5167 (ref 0.5149) / 0.5152 (ref 0.5174).
- the KAN alone is as accurate as the best families: OOF 0.502 / 0.500 (rf 0.499, et 0.500), but carries the same information:
  blend weight 0 on main, 0.13 long / 0.30 short on fresh.
- verdict: rejected (gain < 0.002 on both sets). Flag KAN_FAMILY off.
### Cycle B — interpretable additive KAN on 8 signals + symbolic formula (src/kan_explain.py → reports/kan/), no score claim
- OOF cv_testmix (3 fold-seed sets): spline KAN **0.6958**, its formula 0.7207, ridge on the same 8 inputs 0.7252; the full stack 0.5011.
- auto_symbolic chose a straight line for every input (simplicity preference) → formula ≈ ridge; the curves carry the shape:
  examiner grade S-shaped (floor below average, steep middle), repair rate falling (mostly between −1 SD and the mean), n_words the
  largest effect (1.39 grade points over the middle 90%, partly recording length: 45 vs 60 s), vocabulary variety rising; lm_gap,
  articulation rate and repetition ~flat given the others.
- formula (z-scores): grade = 3.47 + 0.409·n_words + 0.324·examiner + 0.294·ttr − 0.292·repair − 0.106·sent_len − 0.044·lm_gap
  − 0.028·artic_rate − 0.018·repetition.
- **correction (Amrita challenged the summary, rightly):** "barely any weight" was wrong for the fresh run and the overlap was asserted,
  not measured. Measured: KAN OOF testmix main 0.5005 (best family; stack 0.5009), short 0.4996 (ET 0.5024, RF 0.5065); fresh 0.5023
  (ET 0.4984, RF 0.4993). Weights: main 0 (blend weights are fitted on topic-held-out OOF, where ET 0.5089 beats KAN 0.5117); fresh
  0.13 long / 0.30 short. Overlap: corr with ET 0.995, RF 0.993, lgbm 0.988; corr(y − stack, kan − stack) +0.05 → little new
  information, so the stack moves −0.0002 / −0.0012 and the pre-set rejection stands. Separate finding: one additive KAN on the
  families' inputs matches the 24-member stack (0.5005 / 0.5023 vs 0.5009 / 0.5010) with one readable curve per input.
## denoise-then-transcribe diagnostic (shl-denoise-asr v1, 37 min on 2×T4; SpeechBrain SepFormer DNS-4, Apache-2.0; Whisper-large-v3-turbo, MIT)
- question: do noisier test-short recordings make the recogniser produce errors that read as weak grammar? Each of 459 clips (all
  short: train 174 / test 147; control: 69 test + 69 train long) transcribed twice by the same Whisper, original vs denoised audio.
- denoiser works: SNR b45 test 35.2 → 41.7, train 37.0 → 44.2 dB; other short test 36.5 → 44.1, train 42.3 → 50.0.
- pre-set rule (step 2 = full re-transcription only if both): (a) test-short transcripts change more than train-short: **test 0.032
  vs train 0.041 WER, diff −0.009 [−0.030, +0.013]** ✗; (b) change correlates ≥ +0.15 with hybrid11's under-prediction on train
  short clips: **+0.023 (Spearman +0.038)** ✗. Change vs recording SNR −0.07. → **NO-GO.**
- caveat: the largest "changes" were whole openings dropped by the diagnostic's own chunked decoding (transformers
  chunk_length_s, "experimental" for seq2seq); the production transcripts are not affected (checked on the same clips;
  Whisper/CTC word ratio median 0.93 test / 0.95 train). Transcript quotes removed for the public copy. The artefact adds
  noise to (a) but none of the three signals points to noise-driven transcript errors. Recording noise does not distort our transcripts; the short-clip shift is in the audio inputs only.
## KAN as the replacement for the 24-member blend (pre-set non-inferiority rule, margin 0.003; Amrita: "run next")
- base + KAN_FAMILY runs: main 1019-210356 (stack 0.5115), fresh 1019-210934 (0.5077); combination KAN runs 1019-201900 / 1019-202613.
- "KAN hybrid": the NNLS blend replaced by the KAN family's prediction in each pipeline, post-stages as hybrid11; "KAN alone": the
  combination's KAN, nothing after it. Reference: the hybrid11 recipe built from the same fold-seed set (main 0.5016 / fresh 0.5004;
  held-out prompts after the correction 0.5149 / 0.5174). Reference replays verified against the runs' files.
| | main cv_testmix (short) | main held-out prompts | fresh cv_testmix (short) | fresh held-out prompts |
|---|---|---|---|---|
| KAN hybrid | 0.4974 (−0.0042; −0.0127) [−0.0158, +0.0080] | 0.5227 (**+0.0078**) | 0.4985 (−0.0019; −0.0083) | 0.5186 (+0.0012) |
| KAN alone | 0.5005 (−0.0011; −0.0090) | 0.5189 (**+0.0041**) | 0.5023 (+0.0020) | 0.5142 (−0.0032) |
- verdict: **both rejected** (held-out-prompt check on the main set beyond +0.003).
- lesson: replacing the blend by one additive KAN fits familiar prompts better (cv_testmix lower on both sets, short clips −0.008 to
  −0.013) but generalises worse to unseen prompts on one of the two sets — the trade-off that also sank LEACE and the batch-specific
  examiner slope. Board files written for a like-for-like check if wanted: runs/probes/kan_hybrid_2set.csv vs ref_hybrid_2set.csv.
## KAN in the examiner-correction stage (Amrita: "use it for feature correlation and as a loss function"), src/kan_correction.py
- the correction learns how the 21 judge numbers + the prediction relate to the stack's residual; ridge → additive KAN (λ 0.01,
  steps on an inner split), loss either weighted squared error or the competition metric 0.6·RMSE + 0.4·(1 − r) of the corrected
  prediction. Replays of all 8 runs (members untouched).
- pre-set rule: vs the ridge counterpart, plain cv_testmix or the board metric ≥ 0.002 better, better on both the main and fresh
  combination runs, topic check after the correction not worse by > 0.001.
| variant | cv_testmix Δ (95%) | board metric Δ | IW Δ | combo main / fresh | topic check |
|---|---|---|---|---|---|
| kan_corr (mse) vs hybrid11 | −0.0008 [−0.0035, +0.0019] | −0.0007 | +0.0002 | +0.0020 / −0.0018 | 0.5149 → 0.5164 |
| kan_corr_board vs hybrid11 | −0.0004 [−0.0030, +0.0023] | −0.0003 | +0.0005 | −0.0001 / −0.0005 | → 0.5171 |
| base_corr_kan vs base_corr | −0.0007 [−0.0059, +0.0046] | −0.0006 | +0.0011 | as kan_corr | → 0.5164 |
| base_corr_kan_board vs base_corr | −0.0005 [−0.0056, +0.0046] | −0.0004 | +0.0009 | as kan_corr_board | → 0.5171 |
- verdict: **all rejected** (gains < 0.002, inconsistent across fold-seed sets, worse on unseen prompts). Not submitted.
- lessons: the metric as the loss changes little vs squared error (consistent with the earlier finding that the least-squares scale is
  already optimal for 0.6·RMSE + 0.4·(1 − r)); the curved corrector moves some test predictions by up to 0.43 (base_corr_kan) —
  spline extrapolation on test inputs, a risk the linear ridge does not carry.
- KAN line closed: as a 10th family (no gain), as the blend's replacement (better on familiar prompts, worse on unseen), in the
  correction stage (no gain, worse on unseen prompts). The interpretable 8-signal KAN (reports/kan/) is the useful artefact.
