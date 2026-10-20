# Results timeline

Public = Kaggle public leaderboard (0.6 × RMSE + 0.4 × (1 − Pearson), about 130 clips; one clip ≈ 0.002–0.005). CV is
out-of-fold RMSE. Until 11 Oct that was on random folds; from 12 Oct it is speaker-grouped and test-mix-weighted
(`cv_testmix`, see [VALIDATION.md](VALIDATION.md)). The two schemes aren't comparable, and runs before 15 Oct still included
a leaky column. Ranks are approximate, read from the board on the day.

## Phase 1: the original notebook (7–11 Oct)
| date | submission | what changed | public |
|---|---|---|---|
| 7 Oct | notebook v3 | first submitted version | 0.4804 |
| 7–8 Oct | notebook v4–v12 | iterations of the notebook | 0.4623 → 0.4175 |
| 8 Oct | notebook v14 (the notebook's own "v3") | handcrafted + grammar-aware features, sentence embedding, a frozen WavLM, fine-tuned DeBERTa, 9 families × 3 seeds | 0.3860 |
| 8 Oct | notebook v16 ("v4") | more speech encoders, fine-tuned WavLM | 0.3750 |
| 10 Oct | notebook v18 ("v5") | five frozen encoders, fine-tuned WavLM, speaker neighbours (the experiment loop's baseline, CV 0.4750) | 0.3770, rank ~90 / 425 |
| 11 Oct | v21 / run 1011-190112 | DeBERTa on the (CTC raw, Whisper clean) transcript pair (CV 0.4725) | **0.3707** |

## Phase 2: a validation scheme that mirrors the test (12–14 Oct)
| date | run | what changed | cv_testmix | public |
|---|---|---|---|---|
| 13 Oct | 1012-204439 | speaker-grouped folds, members rebuilt leak-free (one leaky column kept), Qwen2.5 heads, per-batch blend, label denoising | 0.4979 | **0.3441**, rank ~49 / 436 |
| 14 Oct | 1013-110218 | blend fitted on topic-held-out predictions (unseen prompts 0.5043 → 0.5013) | 0.4982 | 0.3426, rank ~46 |
| 14 Oct | 1014-164335 | + Voxtral-Mini-3B audio-LLM head | 0.4978 | 0.3429 |
| 14 Oct | 1014-170108 | + Laya checklist features, no Voxtral | 0.4939 | 0.3434 |
| 15 Oct | 1014-175304 | + Qwen heads on verbatim transcripts | 0.4934 | 0.3414, rank ~44 |

## Phase 3: the honest stack, and two pipelines (15–16 Oct)
| date | submission | what changed | cv_testmix | public |
|---|---|---|---|---|
| 15 Oct | 1015-070513 | **leaky fine-tuned WavLM removed**: the honest base | 0.5133 (0.02 *worse*: the leak's inflation) | **0.3344**, rank ~40 |
| 15 Oct | 1015-113504 | honest combination: + RNC, DeBERTa-large pair, verbatim pair, examiner correction, pooling | 0.5021 | 0.3371 |
| 15 Oct | hybrid | short clips = mean of base and combination; long = combination | robust on 2 fold-seed sets | **0.3309**, rank ~27 |
| 15 Oct | hybrid2 | the same with the shrunk short-clip blend | — | 0.3297 |
| 15 Oct | hybrid7 | each pipeline bagged over 2 fold-seed sets | — | 0.3283 |
| 15 Oct | hybrid10 | bagged over 4 fold-seed sets | — | 0.3280, rank ~23 |
| 16 Oct | **hybrid11** | with the 6-seed DeBERTa-large column (CV 0.5013) | **0.5013** | **0.3279**, rank ~24 / 447 |
| 16 Oct | hybrid13 vs hybrid14 | clean A/B of the examiner correction, matched bagging | — | 0.3335 without, **0.3280 with** |
| 16 Oct | hybrid12 | + examiner slope for the 45.06 s batch (best CV of the project, 0.4994) | 0.4994 (one set) | 0.3311 (worse) |

## Phase 4: checks, hedges and the last variants (17–19 Oct)
| submission | what it is | public |
|---|---|---|
| bag_base / bag_combo | each pipeline alone, 4-set bags (the hybrid beats both by ~0.0055) | 0.3334 / 0.3338 |
| **pubmix_c** | hybrid11 − 0.041, the bias estimated from public scores (≈ 1 standard error) | **0.3271** |
| pubmix_a | the best mix of earlier files by the public-score predictor (mostly hybrid11 + a shift) | 0.3270 |
| hybrid_bn | audio embeddings z-scored within each recording batch | 0.3342 |
| hybrid_leace / leace_shift | prompt topic erased from text embeddings; the same − 0.042 | 0.3272 / **0.3264** |
| base_corr | the examiner correction on both pipelines (CV 0.0007 better) | 0.3270 |
| base_corr_pairwise / exam_sc_fs | + pairwise examiner / self-consistent examiner | 0.3272 / 0.3279 |
| phi4_corr_both | + Phi-4 examiner numbers (CV not better; board within a clip) | **0.3264** |
| base_corr_perseed | fine-tunes refitted per fold seed in one seed set | 0.3273 |
| base_corr_pcs_combo / hybrid11_pcs_combo | speech-embedding components in the families (CV 0.0028 better) | 0.3302 / 0.3306 |

Best public score: **0.3264**, rank ~22 of 449 (20 Oct). At that point the top of the board was 0.2913, then 0.2962. The
second-placed team's write-up says public-board feedback informed its blend weights.

## Final picks
1. **`hybrid11`** (public 0.3279): the validated design, chosen on CV, fresh fold seeds and unseen prompts.
2. **`pubmix_c`** (public 0.3271): the same predictions shifted by −0.041. CV can't judge a constant offset; this one is
   estimated from the public board, about one standard error from zero. Kaggle keeps the better of the two picks, so this is
   a hedge on the test's bias at no cost.

The final notebook ([kaggle/final](../kaggle/final)) reproduces both files exactly.

## What moved the score
| step | public |
|---|---|
| the transcript pair (DeBERTa on raw + clean text) | 0.3770 → 0.3707 |
| validation that mirrors the test, plus the per-batch blend and denoising | 0.3707 → 0.3441 |
| removing the column that memorised voices | 0.3414 → 0.3344 |
| averaging two pipelines on short clips | 0.3344 → 0.3309 |
| shrunk short-clip blend and bagging over fold seeds | 0.3309 → 0.3279 |
| bias offset from the public board (pick 2) | 0.3279 → 0.3271 |

Everything else was within one clip of noise, or worse.
