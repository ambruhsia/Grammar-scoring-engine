# Amrita's ideas: what they were, what was built, what happened

Each entry gives the idea, why it should help, what was built, and the measured outcome. Numbers are out-of-fold RMSE unless
marked "public", which means the Kaggle public leaderboard (0.6 × RMSE + 0.4 × (1 − Pearson) on about 130 clips). From
mid-October, CV is the speaker-grouped, test-mix-weighted `cv_testmix` described in [VALIDATION.md](VALIDATION.md). Run ids
(`MMDD-HHMMSS`) point into [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

| # | idea | outcome |
|---|---|---|
| 1 | Transcribe twice, read grammar from the gap | **Founding design**: strongest handcrafted feature; the (raw, clean) transcript pair became the best text input |
| 2 | Grammar-aware text features | In the final model; CoLA passes every rubric check |
| 3 | Let the audio speak: frozen speech encoders | In the final model; revealed a halo effect worth stating openly |
| 4 | Human-centred analysis: report cards, accent exposure, pauses | Shaped the findings and the final notebook |
| 5 | Restarts are not self-correction | A rubric-faithful exclusion, kept as a finding |
| 6 | "Is short ⇒ weaker true of test?" | Label-free check: no; the batch prior stayed |
| 7 | Laya, a rubric-conditioned decision model | Checklist features **accepted**; heads and fine-tunes rejected |
| 8 | Research brief: verbatim ASR, audio LLMs, error detectors | Voxtral, verbatim transcripts and the Qwen3.5 examiner all entered the final system |
| 9 | Turn the audio into embeddings for more model families | **Biggest honest CV gain of the final week**, which then failed on test recordings; diagnosed |
| 10 | KANs and symbolic regression | An interpretable model of the grade; no score gain in any of four roles |
| 11 | Challenge the evidence | Caught an unmeasured claim and had it corrected |
| 12 | Check external data licences first | Both candidate corpora excluded |
| 13 | Final picks as a hedge | Validated model + bias-corrected twin |
| — | Two notebook columns later reversed | The fine-tuned WavLM and speaker neighbours learned voices, not grammar |

---

### 1. Transcribe twice, read grammar from the gap
**Idea.** Whisper is trained on subtitles. Its decoder repairs subject–verb agreement, completes fragments and drops "um".
So features built from a Whisper transcript measure Whisper, not the speaker. Transcribe every clip a second time with a
CTC model that has no language model (`wav2vec2-large-robust-ft-swbd-300h`) to get the words as spoken. The edit distance
between the two transcripts, `repair_wer`, measures how much Whisper had to rewrite the speaker.
**Outcome.**
- `repair_wer` is the strongest handcrafted feature: Spearman −0.55, monotone across six bins (mean grade 4.48 → 2.83).
  On its own it takes LightGBM from 1.01 to 0.84.
- The idea grew beyond the feature. A DeBERTa fine-tuned on the *pair* (CTC raw, Whisper clean) became the best text
  member: 0.561 speaker-grouped, and 0.550 for the large model. The raw CTC text alone is no better (0.644).
- The model needs both versions to tell a speaker's error from a misrecognition (v21, run 1011-190112: public 0.3770 → 0.3707).
- Every later text input reuses the pair, including the verbatim transcripts and the examiner's two-transcript questions.

### 2. Grammar-aware text features
**Idea.** Counts such as type-token ratio or sentence length can't see "he go to school yesterday". So ask direct
questions:
- GPT-2's surprise on the raw transcript, against the cleaned one (`gpt2_nll_gap`);
- a CoLA acceptability classifier on each sentence;
- whether sentences end at all, measured per Whisper segment (`no_punct`: −0.171, against +0.036 for the naive version).
**Outcome.**
- CoLA correlates +0.53 with the grade and passes every rubric-conformance check (Phase B).
- It adds little accuracy on top of `repair_wer` and GPT-2, but it is the explainable one ("38% of your sentences were
  judged ungrammatical"), so the report cards use it.
- All three are in the final model.

### 3. Let the audio speak
**Idea.** Every feature so far reached the score through text. Add frozen speech encoders: WavLM-base-plus, WavLM-large
and Whisper-medium's encoder. The protocol is fixed in advance: mean and standard deviation over time for every layer,
layers averaged uniformly, a ridge inside the folds. No layer is chosen by its score.
**Outcome.**
- A speech model that never sees the transcript predicted the *grammar* rating better than a fine-tuned transformer
  reading it (0.56 against 0.63 in v3). Amrita's reading: the human ratings partly judge how a speaker *sounds*, a halo
  effect familiar from human-rated speaking tests.
- She kept the column and stated the accent-exposure cost openly: dropping it costs about 0.07.
- Later, under prompt-held-out validation, the audio heads were the members that *least* depended on familiar prompts
  (topic gap +0.006, against +0.03 to +0.10 for the text members).
- In the final model.

### 4. Human-centred analysis
**Idea.** A score used on people should come with reasons and with its biases measured.
**Outcome.**
- **Report cards** use only explainable features.
- **The fairness analysis** found that `repair_wer` correlates −0.94 with the CTC model's confidence. Controlling for that
  confidence, its link to the grade falls from −0.55 to about −0.2, so part of it measures how recognisable the speech is,
  an accent channel.
- **Mid-clause pauses** carried the *opposite* sign to the hypothesis (+0.15). Capable speakers pause while building complex
  sentences. "Counting silences was the wrong instrument — not the audio."
- These framed the final notebook's findings and its limitations section.

### 5. Restarts are not self-correction (decision, 13 Oct)
**Idea.** The rubric *credits* self-correction at level 5. The measurable proxy, the CTC restart rate `self_repair_per100`,
went with *lower* grades, because it captures disfluent restarts rather than repairs.
**Outcome.**
- Amrita excluded it from the model and kept it as a finding rather than let the model penalise a behaviour the rubric
  rewards.
- An LLM count of corrected errors, tried as a replacement, failed the same checks.
- Lesson: a rubric descriptor and its measurable proxy can point in opposite directions, so check the sign before trusting
  the name.

### 6. "Is short ⇒ weaker true of test?" (13 Oct)
**Idea.** Short training clips are graded lower. If the model just learned "short = weak", it would mis-score the test set,
which is 68% short clips.
**Outcome.**
- A check using only signals that never saw a label found that test-short clips look like train-short ones, not like
  strong long ones (examiner score 3.017 against 3.037; LLM-counted errors 34.1 against 30.5 per 100 words).
- So the batch prior was kept. The real short-clip problem turned out to be compression *within* the batch (predicted
  spread 0.67 against a label spread of 0.90), and that became a target for later work.

### 7. Laya, a rubric-conditioned decision model (14 Oct)
**Idea.** A model trained to make typed decisions (`convaiinnovations/laya`, Apache-2.0) could answer rubric questions that a
regressor can't express.
**Outcome.** Tried four ways:

| use | result |
|---|---|
| frozen heads on its encoder + question probabilities | 0.65–0.66 alone; stack 0.0008 better, below the 0.002 bar → rejected |
| fine-tuned on the transcript pair (stabilised: layer-wise lr decay, warm-up, collapse guard) | no fold collapsed (DeBERTa-large had collapsed on 5/15); 0.610 alone, but r 0.93 with the DeBERTa pair → rejected |
| decision-head fine-tune (213 min on a T4) | 0.606 alone; stack 0.0040 worse → rejected |
| **checklist: 8 yes/no questions per sentence** | **accepted: CV 0.0024 better, held on fresh fold seeds (0.0022 better)** |

The questions that work compare the two transcripts ("same meaning?", "grammar-only change?"). Questions about a specific
rubric level mostly fail or flip sign. Four checklist features are in the final base pipeline.

### 8. Research brief (15 Oct): verbatim ASR, audio LLMs, grammar-error detectors
**Idea.** Study what the teams above us do and look for open models that add *new* information: disfluency-preserving
transcription, audio LLMs, grammar-error detectors, speech-native assessors.
**Outcome.** All of these entered the final system:
- **Voxtral-Mini-3B** (an audio LLM) became the best single member (0.533). Averaging three views (the grammar question
  after the audio, or before it in two wordings) gave 0.5315.
- **Verbatim Whisper-large-v3-turbo transcripts.** DeBERTa on the (verbatim, clean) pair is the best member on unseen
  prompts (0.4967 against 0.5011).
- **Qwen3.5-9B as a zero-shot examiner.** Its rubric grades correlate up to +0.73 with the human grade. Used as a residual
  correction, it was worth 0.0055 on the public board in a clean A/B ([APPROACH.md](APPROACH.md)).
- From the public write-ups: the board metric itself (0.6 × RMSE + 0.4 × (1 − Pearson)) and the bias probes.

### 9. Turn the audio into embeddings that more model families can use (19 Oct)
**Idea.** The model families see each speech encoder only through its single prediction. Give them the embeddings
themselves, compressed to 16 principal components per encoder, so trees can combine *how the voice sounds* with the text
features.
**Outcome.**
- The largest honest CV gain of the final week: −0.0037 on the main fold seeds and −0.0053 on fresh ones, better on all
  four fold-seed sets and on unseen prompts. The tree families gained most (LightGBM 0.531 → 0.517).
- **On the public board it was worse by about 0.003.**
- Label-free diagnosis: within each recording batch, the components separate train from test far more than the
  handcrafted features do (adversarial AUC 0.76 against 0.64 on short clips).
- The most grade-related components sit shifted toward "lower grade" on test short clips, which are noisier recordings
  (SNR 37 against 42 dB).
- Three repairs failed: dropping shifted components, importance-weighting toward test-like clips, and re-testing on fresh
  seeds. Not used in the final picks.
- Lesson: CV can't see a recording shift. Every audio-derived input needs an adversarial check per batch.

### 10. Kolmogorov–Arnold networks and symbolic regression (20 Oct)
**Idea.** A KAN learns one smooth curve per input instead of thousands of tree splits. Symbolic regression can turn those
curves into a formula, so the model explains itself. Amrita also proposed two further uses: the KAN as the model that
links extra signals to the stack's errors (the "feature correlation" stage), and training it with the competition metric
as its loss.

| role | result |
|---|---|
| **interpretable model on 8 understandable signals** | curves 0.696, formula 0.721, the full stack 0.501. The curves: the examiner grade acts as an S-curve with a floor; the repair rate pulls the grade down; *amount said* is the largest effect, which partly reflects recording length, a validity point; vocabulary variety helps. See [reports/kan](../reports/kan) |
| 10th model family | alone 0.5005, the best family and equal to the stack, but r 0.993–0.995 with the forests, so the stack moves only 0.0002 / 0.0012 |
| replacing the 24-member blend | better on familiar prompts (−0.004 / −0.002; short clips −0.013 / −0.008), worse on unseen prompts (+0.008 on one fold set) → rejected under a pre-set non-inferiority rule |
| correction stage, squared error or the competition metric as the loss | no gain (−0.0008 / −0.0004); the metric as the loss behaves like squared error, since the least-squares scale is already optimal for that metric |

### 11. Challenge the evidence (20 Oct)
**Idea.** A summary said the KAN family "barely gets any weight because it overlaps with the blend". Amrita disputed it.
**Outcome.** She was right on both counts:
- the KAN got 30% of the short-clip blend on one fold-seed set;
- the overlap had been asserted, not measured.

The correction, with measurements (r 0.993–0.995 with the forests; its disagreement with the stack correlates +0.05 with
the stack's errors), is in the log.

### 12. Check external data licences before downloading (17 Oct)
**Idea.** More labelled speech would target the short-clip weakness directly, but only if its licence allows commercial use.
**Outcome.** Both were excluded, and nothing was downloaded:
- **Speak & Improve 2025:** non-commercial research only; models derived from it need the publisher's approval.
- **ICNALE:** no licence granted.

### 13. Final picks as a hedge (17 Oct)
**Idea.** Kaggle keeps the better of two picks, so pick two that fail differently.
**Outcome.**
- **Pick 1: `hybrid11`**, the validated design (public 0.3279).
- **Pick 2: `pubmix_c`**, the same predictions shifted by −0.041. That's the public-board bias estimated from the public
  scores of 32 earlier submission files, about one standard error from zero (public 0.3271).
- Amrita also passed on an external review of the notebook. Its suggestions became the sanity checks (shuffled labels,
  an independent rebuild, train-versus-test shift per member) and the paired speaker-group bootstrap
  ([VALIDATION.md](VALIDATION.md)). Every check passed.

### Two notebook columns that were later reversed
Amrita's notebook already flagged the risk in both:
- **Speaker neighbours** (`spk_knn_pred`, from x-vectors). The notebook said it "recognises voices… measures nothing about
  grammar". Under speaker-grouped folds its error rose from 0.718 to 0.792, and it was dropped.
- **Fine-tuned WavLM** (`ft_wavlm_pred`). On clips whose speaker also had a clip in training it scored 0.462; on the rest,
  0.646. The old random folds let it memorise voices, and the blend had given it 40% of the long-clip weight.
  Removing it made the CV 0.02 *worse* (that was the leak's inflation) and improved the public score from 0.3414 to 0.3344:
  the clearest case in the project of a validation choice changing the result.
