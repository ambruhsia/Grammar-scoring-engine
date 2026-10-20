# Amrita's ideas: what they were, what was built, what happened

This page lists the ideas Amrita originated or directed: the reasoning, what was built, and the measured outcome. Techniques
adopted from other participants' public write-ups, and methods developed in the project's experiment loop, are credited
separately at the end ([Where the rest came from](#where-the-rest-came-from)).

Numbers are out-of-fold RMSE unless marked "public", which means the Kaggle public leaderboard (0.6 × RMSE + 0.4 × (1 −
Pearson) on about 130 clips). From mid-October, CV is the speaker-grouped, test-mix-weighted `cv_testmix` described in
[VALIDATION.md](VALIDATION.md). Run ids (`MMDD-HHMMSS`) point into [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

| # | Amrita's idea | outcome |
|---|---|---|
| 1 | A human-centred reading of the model: accent exposure, report cards, pauses | Framed every finding and the final notebook's limitations |
| 2 | Restarts are not self-correction | A rubric-faithful exclusion, kept as a finding |
| 3 | "Is short ⇒ weaker true of test?" | Label-free check: no; the batch prior stayed |
| 4 | Laya, a rubric-conditioned decision model | Checklist features **accepted into the final model** |
| 5 | A research brief aimed at the weak spots | Set the agenda of the final phase; the weak spots it named became its targets |
| 6 | Turn the audio into embeddings that more model families can use | **The biggest honest CV gain of the final week**; it then failed on test recordings, and the failure was diagnosed |
| 7 | Kolmogorov–Arnold networks and symbolic regression | **A model that explains the grade with readable curves**; tested in four roles |
| 8 | Challenge the evidence | Caught an unmeasured claim and had it corrected |
| 9 | Check external data licences before downloading | Both candidate corpora excluded |
| 10 | Final picks as a hedge, and letting the board judge | Validated model + bias-corrected twin |

---

### 1. A human-centred reading of the model
**Idea.** A score used on people should come with reasons, and with its biases measured and stated rather than hidden.
**What was built.**
- Report cards that use only explainable features. For example, "38% of your sentences were judged ungrammatical" is
  something a candidate can act on.
- An accent-exposure analysis of the strongest signals.
- A reading of the pause measures against the rubric.

**Outcome.**
- **Accent exposure.** The Whisper-versus-CTC repair rate correlates −0.94 with the CTC model's own confidence. Controlling
  for that confidence, its link to the grade falls from −0.55 to about −0.2, so part of it measures how *recognisable*
  someone's speech is.
- **A halo effect, stated openly.** A speech model that never sees the transcript predicted the *grammar* rating better
  than a transformer reading it (0.56 against 0.63). Amrita's reading: the human ratings partly judge how a speaker
  *sounds*, a known problem in human-rated speaking tests. She kept the column and stated the cost of removing it (about
  0.07) instead of dropping it silently.
- **Pauses.** Mid-clause pauses carried the *opposite* sign to the hypothesis (+0.15): capable speakers pause while
  building complex sentences. "Counting silences was the wrong instrument — not the audio."
- **Consequences.** These framed the final notebook's findings and limitations. Later checks kept returning to them: CTC
  failures under-score good speakers, the amount said is rewarded though the rubric doesn't grade it, and short clips
  score below their measurable grammar.

### 2. Restarts are not self-correction (decision, 13 Oct)
**Idea.** The rubric *credits* self-correction at level 5. The measurable proxy, the CTC restart rate, went with *lower*
grades, because it captures disfluent restarts rather than repairs.
**Outcome.**
- Amrita excluded it from the model and kept it as a finding, rather than let the model penalise a behaviour the rubric
  rewards.
- An LLM count of corrected errors, tried as a replacement, failed the same checks.
- Lesson: a rubric descriptor and its measurable proxy can point in opposite directions, so check the sign before trusting
  the name.

### 3. "Is short ⇒ weaker true of test?" (13 Oct)
**Idea.** Short training clips are graded lower. If the model simply learned "short = weak", it would mis-score a test set
that is 68% short clips.
**Outcome.**
- A check using only signals that never saw a label: test-short clips look like train-short ones, not like strong long ones
  (examiner score 3.017 against 3.037; LLM-counted errors 34.1 against 30.5 per 100 words).
- Amrita decided the batch prior was not suspect.
- The real short-clip problem turned out to be compression *within* the batch (predicted spread 0.67 against a label spread
  of 0.90), which became the target of later work.

### 4. Laya, a rubric-conditioned decision model (14 Oct)
**Idea.** A model trained to make typed decisions (`convaiinnovations/laya`, Apache-2.0) could answer rubric questions that
a regressor can't express.
**Outcome.** Tried four ways:

| use | result |
|---|---|
| **checklist: 8 yes/no questions per sentence** | **accepted: CV 0.0024 better, held on fresh fold seeds (0.0022 better)**; in the final base pipeline |
| frozen heads on its encoder + question probabilities | 0.65–0.66 alone; stack 0.0008 better, below the 0.002 bar |
| fine-tuned on the transcript pair, stabilised | no fold collapsed, where DeBERTa-large had collapsed on 5 of 15; 0.610 alone, but r 0.93 with the existing pair model |
| decision-head fine-tune (213 min on a T4) | 0.606 alone; stack 0.0040 worse |

The questions that work compare the two transcripts ("same meaning?", "grammar-only change?"). Questions about a specific
rubric level mostly fail or flip sign.

### 5. A research brief aimed at the weak spots (15 Oct)
**Idea.**
- **Where to look:** look past tuning toward models that add *new* information. That meant disfluency-preserving
  ("verbatim") transcription, audio LLMs, grammar-error detectors and speech-native assessors, plus the speaking-assessment
  literature and earlier competitions.
- **What to aim at:** the weak spots. Short clips (68% of test), unseen prompts, compressed extremes (a true 5.0 predicted
  4.3 on short clips), and speech-recognition failures under-scoring good speakers.
- **The hard limits:** commercial-use licences only, and no competition data sent to external services.

**Outcome.** It set the agenda of the final phase (15–17 Oct), and the main results of that phase each answer a weak spot
it named:
- **Short clips:** averaging two pipelines on short clips (public 0.3344 → 0.3309 → 0.3279).
- **Unseen prompts:** the unseen-prompt check moved after the correction stage, which caught a correction that had
  learned the training prompts.
- **Hidden shortcuts:** removing a column that had memorised voices (public 0.3414 → 0.3344).
- **New information:** a Qwen3.5-9B examiner as a residual correction (public −0.0055 in a clean A/B).

Several of the specific techniques came from the public write-ups the brief asked to study; they are credited
[below](#adopted-from-other-participants-public-write-ups).

### 6. Turn the audio into embeddings that more model families can use (19 Oct)
**Idea.** The model families see each speech encoder only through its single prediction. Convert the audio into a compact
embedding table instead, 16 principal components per encoder, so tree models can combine *how the voice sounds* with the
text features. It recasts the problem for a wider range of model families.
**Outcome.**
- **On CV:** the largest honest gain of the final week. −0.0037 on the main fold seeds and −0.0053 on fresh ones, better on
  all four fold-seed sets and on unseen prompts. The tree families gained most (LightGBM 0.531 → 0.517).
- **On the public board:** about 0.003 worse.
- **The diagnosis**, without labels:
  - within each recording batch, the components separate train from test far more than the handcrafted features do
    (adversarial AUC 0.76 against 0.64 on short clips);
  - the most grade-related components sit shifted toward "lower grade" on test short clips, which are noisier recordings
    (SNR 37 against 42 dB).
- **Three repairs failed:** dropping the shifted components, importance-weighting toward test-like clips, and re-testing
  on fresh seeds.
- **Lesson:** CV can't see a recording shift, so every audio-derived input needs an adversarial check per recording batch.

### 7. Kolmogorov–Arnold networks and symbolic regression (20 Oct)
**Idea.** A KAN learns one smooth curve per input instead of thousands of tree splits. Symbolic regression turns those
curves into a formula, so the model explains itself. Amrita proposed two further uses:
- the KAN as the model that links extra signals to the stack's errors (the correction stage);
- the KAN trained with the competition metric itself as its loss.

![Additive KAN: each signal's effect on the predicted grade](../reports/kan/kan_curves.png)

| role | result |
|---|---|
| **interpretable model on 8 understandable signals** | curves 0.696, formula 0.721, against the full stack 0.501. The curves show four things: the examiner grade acts as an S-curve with a floor; the repair rate pulls the grade down; *amount said* is the largest effect, which partly reflects recording length (a validity point); vocabulary variety helps. See [reports/kan](../reports/kan) |
| 10th model family | alone 0.5005, the best family and as accurate as the whole 24-member stack, but r 0.993–0.995 with the forests, so the stack moves only 0.0002 / 0.0012 |
| replacing the 24-member blend | better on familiar prompts (−0.004 / −0.002; short clips −0.013 / −0.008), worse on unseen prompts (+0.008 on one fold set) → rejected under a pre-set non-inferiority rule |
| correction stage, squared error or the competition metric as the loss | no gain (−0.0008 / −0.0004). The metric as the loss behaves like squared error, because the least-squares scale is already optimal for that metric |

The lasting result is the interpretable model, together with the finding that one additive KAN matches the full stack on
familiar prompts.

### 8. Challenge the evidence (20 Oct)
**Idea.** A summary said the KAN family "barely gets any weight because it overlaps with the blend". Amrita disputed it.
**Outcome.** She was right:
- the KAN got 30% of the short-clip blend on one fold-seed set;
- the overlap had been asserted, not measured.

The correction, with measurements (r 0.993–0.995 with the forests; its disagreement with the stack correlates +0.05 with
the stack's errors), is in the log.

### 9. Check external data licences before downloading (17 Oct)
**Idea.** More labelled speech would target the short-clip weakness directly, but only if its licence allows commercial use.
**Outcome.** Both were excluded, and nothing was downloaded:
- **Speak & Improve 2025:** non-commercial research only; models derived from it need the publisher's approval.
- **ICNALE:** no licence granted.

### 10. Final picks as a hedge, and letting the board judge (15–19 Oct)
**Idea.** Kaggle keeps the better of two picks, so pick two that fail differently.
**Outcome.**
- **Pick 1: `hybrid11`**, the validated design (public 0.3279).
- **Pick 2: `pubmix_c`**, the same predictions shifted by −0.041, the public-board bias estimated from 32 earlier
  submissions (public 0.3271). It hedges the test's bias at no cost.
- Late in the competition Amrita also lifted the rejection rules for a while, so that variants CV had rejected could still
  get one board test. None beat the picks by more than a clip.
- She passed on an external review of the notebook. Its suggestions became the sanity checks and the paired speaker-group
  bootstrap in [VALIDATION.md](VALIDATION.md), and every check passed.

---

## Where the rest came from

### Adopted from other participants' public write-ups
| technique | from | how it's used here |
|---|---|---|
| Transcribe twice (Whisper, and a CTC model without a language model); grammar read from the gap between the transcripts | participants' public write-ups, adopted into the original notebook | the repair rate is the strongest handcrafted feature (Spearman −0.55). DeBERTa on the (raw, clean) transcript pair is the best text member (0.561; DeBERTa-large 0.550) |
| Speaker twins and the two recording batches as the data's structural problems | participants' public write-ups | the basis of the speaker-grouped folds and test-mix weighting ([VALIDATION.md](VALIDATION.md)) |
| Voxtral-Mini-3B audio-LLM states as the strongest single view | the top public write-ups (AMITABHSHUKLA, Techy-A, PRINCE2-AI) | extended here with question-first views (0.5315) |
| Qwen3.5-9B scoring each transcript with the official rubric | the #1 public write-up (AMITABHSHUKLA) | used here as a *residual corrector* inside the folds, not a member (public −0.0055) |
| The board metric decoded with shift probes; an identity for predicting the public score of a blend | Techy-A | bias probes (calibration isn't the gap); the bias offset in pick 2 |
| Pooling a speaker's own test clips | nakulgrover118, Anurag Nayak | p' = 0.7·p + 0.3·(voice-group mean) |
| A separate stacker for the 45 s batch; training on test-length crops | PRINCE2-AI; other write-ups | tried; no gain here ([WHAT_FAILED.md](WHAT_FAILED.md)) |

### Developed in the project's experiment loop
These were developed in the experiment loop, under Amrita's rules (see [APPROACH.md](APPROACH.md) and
[VALIDATION.md](VALIDATION.md)):
- test-mix-weighted CV and the unseen-prompt check;
- members rebuilt leak-free, which exposed a fine-tuned WavLM column that had memorised voices (public 0.3414 → 0.3344);
- per-batch blending fitted on topic-held-out predictions;
- label denoising with speaker means;
- the examiner as a residual corrector;
- averaging two pipelines on short clips;
- the Rank-N-Contrast member;
- pre-set decision rules, and running the unseen-prompt check after every stage.
