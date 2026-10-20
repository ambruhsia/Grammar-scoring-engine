# Grammar Scoring Engine

**Predicting a human grammar rating (1–5, half-point steps) from a 45–60 second recording of spoken English.**
Built for the Kaggle competition *SHL Hiring Assessment 2026* by **Amrita (Kaggle: ambruhsia)**.

The model listens to the answer twice. A speech recogniser with a language model (Whisper) writes down what the speaker
*meant*, and one without a language model (a CTC model) writes down what they *said*. The distance between the two
transcripts is a direct measure of how much grammar had to be repaired. Around that idea sit speech encoders, fine-tuned
transcript models, an LLM examiner and a validation scheme built to mirror the test set. That last part turned out to
matter as much as any model.

| | out-of-fold RMSE (speaker-grouped, weighted to the test's mix) | public leaderboard |
|---|---|---|
| Starting point: the original notebook (v18) | 0.475 on random folds (optimistic, see below) | 0.3770, rank ~90 / 425 |
| **Final pick 1, `hybrid11`**: two pipelines × 4 fold-seed sets, examiner correction | **0.5013** | **0.3279** |
| Final pick 2, `pubmix_c`: hybrid11 shifted by −0.041 (a public-board bias estimate) | — (a test-time offset; CV can't judge it) | 0.3271 |
| Best public score of the project | — | 0.3264, rank ~22 / 449 (20 Oct 2026) |

Other numbers for pick 1 (out of fold): Pearson 0.875, MAE 0.377, QWK 0.856, 72.7% of clips within half a point and 94.8%
within one point. Board metric at the test's batch mix 0.358; its 95% interval on 130-clip resamples is 0.306–0.420,
and the public 0.3279 sits inside it.

> **Why the two columns look so different.** The leaderboard scores 0.6 × RMSE + 0.4 × (1 − Pearson), on about 130
> clips. The CV column is a plain RMSE on 732 clips, with folds that never split a speaker and weights that make the
> short recordings (68% of test, 24% of train) count as much as they will on test. It's pessimistic on purpose. Adding a
> clip-level leak made the old random-fold CV look 0.02 better while the leaderboard got worse.

## What's distinctive here

1. **Two recognisers, one grammar signal** (Amrita's founding design). The Whisper-versus-CTC edit distance (`repair_wer`)
   is the strongest handcrafted feature: Spearman −0.55 with the grade, monotone across six bins.
2. **A validation scheme that mirrors the test set.**
   - Speaker twins are found from voice embeddings (53% of training clips have one), and folds never split them.
   - Every recording batch is weighted to its share of test.
   - Unseen prompts get their own held-out check.
   - Decisions are made with pre-set rules and a paired speaker-group bootstrap.

   This scheme exposed a leaky member that the old CV had rewarded. Removing it cost 0.02 of CV and *gained* 0.007 on the
   leaderboard. [docs/VALIDATION.md](docs/VALIDATION.md)
3. **An LLM examiner used as a corrector, not a member.** A zero-shot Qwen3.5-9B answers rubric questions about three
   transcripts. Its 21 numbers correct the stack's residual inside the folds. In a clean A/B it was worth 0.0055 on the
   leaderboard.
4. **Per-recording-batch stacking.** Even at equal length, a classifier tells the short and long batches apart with AUC 0.93–0.95,
   because the recordings themselves differ. The blend is fitted per batch, on predictions for prompts the model hasn't seen.
5. **Two pipelines averaged where they disagree.** The richer pipeline only helps long clips, a fact visible only on fresh
   fold seeds. So short clips average it with the simpler pipeline. Public score: 0.3344 → 0.3309 → 0.3279.
6. **An interpretable model you can read.** An additive Kolmogorov–Arnold network on eight understandable signals shows the
   shape of each effect (below). One additive KAN on the stack's inputs matches the 24-member stack on familiar prompts.

![Additive KAN: each signal's effect on the predicted grade](reports/kan/kan_curves.png)

## Where to read more

| document | what's in it |
|---|---|
| [docs/IDEAS.md](docs/IDEAS.md) | Amrita's ideas, one by one: the reasoning, what was built, and what happened |
| [docs/APPROACH.md](docs/APPROACH.md) | The final system and the approaches that are unusual for this task |
| [docs/VALIDATION.md](docs/VALIDATION.md) | How results were measured and decided, and why the leaderboard wasn't the yardstick |
| [docs/WHAT_FAILED.md](docs/WHAT_FAILED.md) | About 60 things that didn't work, with numbers and the lesson from each |
| [docs/RESULTS.md](docs/RESULTS.md) | The score timeline, from 0.3770 to 0.3264, and the final picks |
| [docs/REPRODUCING.md](docs/REPRODUCING.md) | How to run it: GPU jobs, the local loop, the final notebook |
| [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) | The full lab notebook, 198 runs in order, mistakes and corrections included |

## Repository map

```
kaggle/main/      the original notebook (Amrita): two ASR passes, handcrafted + grammar-aware features, speech encoders,
                  DeBERTa, 9 model families, report cards and findings
kaggle/final/     the final submission notebook: replays the 8 runs behind hybrid11 from their saved candidate matrices
kaggle/<job>/     22 Kaggle jobs (21 GPU, 1 CPU): grouped DeBERTa, Qwen2.5 / Qwen3.5 / Phi-4 examiners, Voxtral, verbatim ASR, Laya, ...
src/              the local experiment loop: run.py (one experiment = one flag), stacking, validation, every member built on CPU
scripts/          Kaggle helpers: pull the notebook's outputs, guarded submission, scores, GPU jobs
reports/          figures and aggregate tables (validation, rubric checks, error analysis, the KAN curves)
experiments.csv   the run ledger: every experiment's description, CV per seed and verdict
```

## How this was built

Amrita built the original model in a Kaggle notebook (versions 1–18), set the research directions and constraints, and
proposed many of the ideas tested afterwards ([docs/IDEAS.md](docs/IDEAS.md) lists them). The experiment loop that
followed was run with **Claude Code**, an AI coding agent, inside fixed guard rails:
- one change per experiment, kept only if a fixed CV rule accepts it;
- every result logged;
- GPU jobs and submissions only with her approval.

The notebook handled the slow GPU stages and the local loop the fast stacking experiments: 198 logged runs, 22 Kaggle jobs
and 57 board submissions over three weeks.

## Data and licences

**No competition data is in this repository**: no audio, labels, transcripts, features or predictions. Get the data from
the competition page on Kaggle, under its rules. A few job sources embedded per-clip tables when they ran; the public copies
say where those were removed.

Every model used allows commercial use; the competition required it. Whisper and Whisper-large-v3-turbo (MIT), wav2vec2-large-robust
(Apache-2.0), WavLM (CC BY-SA 3.0), DeBERTa-v3 (MIT), Qwen2.5-7B and Qwen3.5-9B (Apache-2.0), Voxtral-Mini-3B (Apache-2.0),
Phi-4 (MIT), Laya (Apache-2.0), all-mpnet-base-v2 (Apache-2.0), SpeechBrain SepFormer DNS-4 (Apache-2.0), spaCy (MIT). The
CoLA classifier's model card states no licence; [kaggle/final](kaggle/final) lists every stage. No external training data
was used: the two corpora considered (Speak & Improve, ICNALE) don't allow commercial use.

Code: MIT ([LICENSE](LICENSE)).
