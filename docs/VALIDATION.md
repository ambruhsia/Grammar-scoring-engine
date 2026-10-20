# How results were measured and decided

## Why the leaderboard wasn't the yardstick

The public board scores about 130 clips. One clip landing a quarter-point differently moves the score by 0.002–0.005. On
130-clip resamples of our own out-of-fold predictions, the board metric's 95% interval is **0.306–0.420**. Most single
changes in this project are worth less than 0.003, so the board can't separate them. The board was used to *confirm*
large, independent effects, never to choose between close variants. The one exception is the bias offset in final pick 2,
which is labelled as such.

## The yardstick, step by step

The speaker-twin and recording-batch problems behind steps 1 and 2 were first reported in other participants' public
write-ups; the steps themselves, and the rest of the table, were built in this project.

| step | what changed | why |
|---|---|---|
| 0 | stratified 5-fold, random | the original notebook; CV 0.475 |
| 1 | **speaker-grouped folds** | x-vectors (WavLM-base-plus-sv) clustered at cosine 0.988, the 99th percentile of test→train nearest similarity: 503 speakers, 53% of training clips have a twin. `StratifiedGroupKFold(5)` never splits a speaker |
| 2 | **test-mix weighting** (`cv_testmix`) | recording batches cut at 47.7 s, the density minimum between the 45 s and 60 s modes. Short clips are 24% of train but 68% of test, so each training clip is weighted by test share / train share (short 2.86, long 0.42) |
| 3 | **leak-free members** | every learned column rebuilt out-of-fold on the grouped folds; columns that can only be built on random folds dropped. Blend weights refit inside every fold: a per-batch blend fitted and scored on the same rows had scored an implausible 0.4977, against a 0.5120 baseline |
| 4 | **unseen-prompt check** (`cv_newtopic`) | ten prompt topics. Train outside topic T (and outside every speaker who appears in T), score T. Fitting the blend on these topic-held-out predictions cost 0.0003 on CV and gained 0.0030 on unseen prompts; the spread across seeds more than halved |
| 5 | **fresh fold seeds** | every accepted change confirmed on fold seeds it was never developed on (7/99/555; later 11/22/33 and 44/66/88). This is how the combination's short-clip gain was shown to be fold luck |
| 6 | **unseen prompts after the correction stage** | a blind spot, found on 16 Oct: the prompt check ran *before* the examiner correction, so corrections were never prompt-tested. A batch-specific correction slope passed CV and failed on the board. The check now runs after every stage |
| 7 | **paired speaker-group bootstrap** | resample the 503 speakers 10,000 times and score old and new on the same draw: a 95% interval for every comparison (below) |
| 8 | **importance-weighted CV** | for audio-derived inputs: validation clips weighted by how test-like they are within their batch (train-vs-test classifier), estimating error on the test's recordings rather than the training ones |

### A leak found by the yardstick
The fine-tuned WavLM column came from the original notebook, built on random folds:
- On clips whose speaker also had a clip in training it scored **0.462**; on the rest, **0.646**. The leak-free members show
  a gap of about +0.04, not +0.18.
- The blend had given it 40% of the long-clip weight.
- Without it, CV went from 0.4939 to 0.5140. That 0.02 was the leak. **The public score improved from 0.3414 to 0.3344.**
- Under random folds this column looked like the best audio member. Under speaker-grouped folds, it was memorising voices.

### Pre-set decision rules
Every experiment had its acceptance rule written in the log *before* the result:
- **Gate:** CV gain ≥ 0.002, better on at least 2 of 3 fold seeds, and grouped CV not worse.
- **Bias-removal changes:** accepted only if CV stays within 0.002 *and* a named bias measure at least halves.
- **Simplifications:** a non-inferiority margin of 0.003.
- **Cycle stop:** three rejections in a row end a cycle of experiments.

No setting was re-run with new values after its result was seen.

## What the bootstrap says CV can resolve

| comparison | new − old | 95% interval | P(new better) |
|---|---|---|---|
| hybrid11 vs base pipeline (4-set bags) | −0.0113 | [−0.0177, −0.0045] | 1.00 |
| hybrid11 vs combination pipeline | 0.0000 | [−0.0059, +0.0055] | 0.49 |
| combination vs base, long clips (each fold-seed set) | −0.020 | excludes 0 on every set | 1.00 |
| combination vs base, short clips (each fold-seed set) | −0.003 to −0.012 | includes 0 on every set | 0.63–0.93 |
| examiner correction on / off | −0.0035 | [−0.0103, +0.0034] | 0.84 |
| same-speaker pooling on / off | −0.0015 | [−0.0033, +0.0006] | 0.93 |
| short blend shrunk / not | −0.0012 | [−0.0026, +0.0001] | 0.97 |

On 732 clips, CV resolves a pipeline-level change (±0.006–0.012) and a light post-processing tweak (±0.002). It can't
resolve a 0.001–0.003 change from a member or a judge. Those were judged on consistency across fold-seed sets and on the
unseen-prompt check, and reported with their intervals.

## Sanity checks, all passed

| check | result |
|---|---|
| **shuffled labels, stacking level**: the 8 runs behind hybrid11 replayed against permuted labels, members fixed | 1.064 vs a predict-the-mean null of 1.029 (10 permutations, half of them keeping speakers together); nothing below the null. A control shows the check *detects* in-sample fitting (1.176 vs 1.194) |
| **shuffled labels, full pipeline**: every member retrained on permuted labels | mean stack − null: +0.0004 (base, 4 permutations), +0.008 (combination); single-permutation scatter ±0.012. The one dip below the null is explained: those permuted labels happened to correlate −0.12 with the true ones, and the label-free examiner features alone exploit that correlation too |
| **independent rebuild** of one member from its description only (numpy ridge with exact leave-one-out, own folds) | identical folds on 6 fold seeds; predictions equal to < 1e-4 |
| **train vs test per member and batch** | every member with blend weight shifts with the consensus; the two outliers have zero weight |
| **zero-label sentinels in test?** | none: static-noise clips have spectral flatness 0.52–0.55, speech ≤ 0.13, test max 0.18 |
| **reproductions** | the original notebook's CV reproduced to 0.0002; the final notebook replays all 8 runs to ≤ 3.6e-15 and its submission byte for byte |
| **fine-tunes built on one fold seed, used on others** | refitted on fold seeds 7/99/555: the optimism is ≤ 0.001 |

## What transferred to the board, and what didn't

| change | CV | public |
|---|---|---|
| leak removal (fine-tuned WavLM dropped) | 0.02 *worse* (the leak's inflation) | **−0.0070** |
| leak-reduced stack on grouped folds + Qwen heads + per-batch blend + denoising | — (new scheme) | **−0.0266** (0.3707 → 0.3441) |
| examiner correction (clean A/B) | −0.0035 | **−0.0055** |
| two pipelines averaged on short clips | robust on 2 fold-seed sets | **−0.0035**, then −0.0012 with the shrunk blend |
| bagging over fold-seed sets | small | −0.0014 (2 sets), −0.0018 (4 sets) |
| new members in the combination (one day's additions) | −0.011 | +0.0027 (worse) |
| examiner slope for the 45.06 s batch | −0.0006 / −0.0028 (main / fresh; best CV of the project) | +0.0031 (worse) |
| speech embedding components as family inputs | −0.0028 | +0.0027 / +0.0032 (worse) |
| per-batch audio normalisation | worse | +0.006 (worse) |

The pattern: **procedure fixes that don't depend on the prompt transferred, and so did the examiner's overall grade.**
Gains that came from new members, or from corrections fitted to one batch or to the training recordings, mostly didn't.
Those learn something specific to the training prompts or recordings, which CV shares with training and the test doesn't.
