# What failed, and what each failure taught

Changes are against the model of the time, out-of-fold, "better"/"worse" in RMSE. "Bar" means the acceptance rule of the
time: at least 0.002 better, on at least 2 of 3 fold seeds. Early entries (11–12 Oct) used random folds; later ones the
speaker-grouped, test-weighted CV described in [VALIDATION.md](VALIDATION.md). Run ids point into
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## 1. Refining the transcript features
| change | result | lesson |
|---|---|---|
| drop clip duration (the main train/test shift) | 0.0026 worse | duration carries real grade signal in training; CV can't say whether that transfers |
| spell out numbers before comparing the transcripts | 0.0003 better, below the bar | only 8% of clips contain numbers |
| split the Whisper/CTC edits into substitutions / insertions / deletions | ±0.0000 | the trees already use what the comparison offers |
| BERTScore between the two transcripts | 0.0001 worse | r −0.95 with `repair_wer`; a real but tiny extra signal |
| TF-IDF word and character n-grams → ridge | 0.707 alone, blend weight 0 | weaker than every DeBERTa column, and nothing they lack |

**Lesson:** after `repair_wer`, more ways of comparing the two transcripts add nothing. The next gains came from new
information (the DeBERTa pair model), not from refinement.

## 2. More capacity on the same inputs
| change | result | lesson |
|---|---|---|
| CatBoost as a 10th family | 0.0002 better | it replaced XGBoost in the blend rather than adding to it |
| Huber loss for the boosted trees (noisy extreme labels) | 0.0008 worse | RMSE rewards getting the extremes right |
| one RBF kernel-ridge head over three speech encoders | 0.0006 better | r 0.94 with the per-encoder ridges; inner CV chose the most linear kernel |
| non-negative ridge stacker on all candidates | ±0.0000 | the blend was already near-optimal; showed the reported CV's optimism (~0.005) |
| RBF-SVR heads on single encoder layers | blend weight 0 (later used in the combination) | the head type isn't the gap |
| additive KAN as a 10th family | 0.0002 / 0.0012 better | alone it equals the stack (0.5005), but r 0.993–0.995 with the forests |
| additive KAN replacing the 24-member blend | better on familiar prompts, 0.008 worse on unseen ones | rejected under a pre-set non-inferiority rule |
| KAN in the correction stage, squared error or the competition metric as the loss | 0.0008 / 0.0004 better, unseen prompts worse | the metric as the loss behaves like squared error: the least-squares scale is already optimal for it |

**Lesson:** tree diversity saturated early, and new heads re-express embeddings the stack already has. A model's standalone
accuracy says little about whether it adds to a blend; measure its correlation with the others first.

## 3. Rubric-structure features (Phase B)
| feature | result | lesson |
|---|---|---|
| clauses per sentence, dependency distance, subordinate-clause ratio | **wrong sign** | Whisper writes weak speakers' speech as unpunctuated run-on "sentences", so structure looks *more* complex as the grade falls |
| LanguageTool grammar errors per 100 words | signal (partial −0.12) but two monotonicity inversions | no separation of level 3 from 4 on short clips |
| CTC restart rate as "self-correction" | wrong sign | it measures disfluent restarts, which the rubric doesn't credit; excluded by Amrita's decision |
| lexical range + repetition as a rubric layer | 0.0006 better, below the bar | the earlier gain had come from the disfluency signal |

**Lesson:** grammar-structure counts computed on ASR text measure the recogniser's segmentation as much as the speaker. The
one rubric descriptor never captured, "range of complex structures", was later partly read by the LLM examiner.

## 4. Another model reading the same transcript
| change | result | lesson |
|---|---|---|
| DeBERTa-v3-large, first attempt | **5 of 15 folds collapsed** to the mean | ~585 training rows need stabilisers; fixed later (lr decay, accumulation) and then useful |
| Laya frozen heads / fine-tuned pair model / decision-head fine-tune | 0.0008 better / 0.0027 worse / 0.0040 worse | a second pair model with another backbone repeats the first (r 0.93) |
| Qwen2.5 heads on the verbatim transcript | best mean CV, but better on 1 of 3 seeds | — |
| delexicalised TF-IDF (content words masked) | 0.74–0.80 alone, topic dependence still there | masking words doesn't remove the prompt |
| Qwen3.5 examiner numbers as a blend *member* | 0.0009 worse | non-negative blend + collinearity; works only as a residual stage |
| examiner asked for an error profile (counts of 11 error types) | 0.0018 worse | itemised counts add noise; the overall grade carries the information |
| examiner self-consistency (6 paraphrased prompts) and few-shot anchors | no variant passed both checks | the paraphrased grades agree closely, so averaging removes little noise |
| pairwise judging against graded anchors, both answer orders | ±0.0000 | R² 0.81 from the existing examiner numbers; fixes the scale, not the ranking |
| Phi-4 (a different model family) as a second examiner | 0.0004 better at best, unseen prompts worse | its rubric answers often weren't digits (2–41% on that question); hidden-state heads lean on prompt content |

**Lesson:** the same model reading the same transcript gives the same information. The transcript side saturated around
the DeBERTa pair model and the Qwen3.5 examiner.

## 5. Audio additions that didn't add
| change | result | lesson |
|---|---|---|
| Voxtral's audio-encoder layers as a head | 0.0008 worse | — |
| Voxtral head | helps long clips, slightly hurts short ones (fresh seeds) | dropped from the base pipeline; its 3-view version helps the combination |
| Voxtral last-token state (audio + question) | 0.580 alone vs 0.538 | the audio-token mean is the better readout |
| heads on equal-length first-30-s crops | 0.0011 worse | the batches differ in recording, not just length |
| attention pooling over 640k Voxtral frames (trained on a T4) | neutral | weaker alone than the mean-pooled ridge |

## 6. Fixing the short-clip and recording-batch shift
| change | result | lesson |
|---|---|---|
| drop the raw amount-spoken counts | the targeted bias shrank 9% (needed 50%) | the learned members carry the "amount" effect themselves |
| treat failed CTC transcripts as missing | bias unchanged | under-scoring comes from the CTC-based members, not the feature |
| train on short views cut from long clips | 0.0003 worse | the short penalty sits in the batch's blend weights, not the features |
| per-batch linear calibration | 0.0023 worse | stretching helps the 14 strong short clips, hurts the many mid-level ones |
| test-mix sample weights in member training | 0.0007 worse | the per-batch blend already does this |
| separate blend group for the 45.06 s batch | no gain; unseen prompts worse | — |
| examiner slope specific to the 45.06 s batch | **best CV of the project; public 0.0031 worse** | with 86 clips, a batch-specific correction learns the batch's *prompts* |
| z-score audio embeddings within each recording batch | worse everywhere (public 0.3342 vs 0.3280) | part of the batch offset is real grade difference |
| recording-quality features (SNR, noise floor, bandwidth, clipping) | 0.0005 worse; unseen prompts worse | test short clips are noisier than train ones, so a noise→grade link can't transfer |
| speech-embedding components as family inputs | **CV 0.0037 / 0.0053 better, public ~0.003 worse** | the components carry the train→test recording shift (adversarial AUC 0.76) |
| keep only components without a train/test shift | rejected by its pre-set shift check | a per-component filter leaves combined shift |
| importance weights toward test-like clips | no gain on importance-weighted CV | down-weighting costs more variance than it removes shift |
| denoise the audio, re-transcribe, compare | **no-go**: test-short transcripts change *less* than train-short | noise doesn't distort the transcripts; the shift is in the audio inputs |

**Lesson:** CV shares training's recordings and prompts, and the test doesn't. Corrections fitted to a batch, and inputs that
carry recording conditions, look good on CV and fail on test. The only short-clip fixes that transferred were prompt- and
recording-independent: averaging two pipelines, and the examiner's overall grade.

## 7. Reshaping the output
| change | result | lesson |
|---|---|---|
| ordinal model + EM re-estimation of the test's grade distribution | 0.031 worse | EM on compressed predictions drifts toward the model's centre |
| ordinal model without EM | 0.0002 better | — |
| nested calibration of the final prediction | 0.0001 worse | the blend is already calibrated |
| stretching predictions around their mean (×0.9–1.2) | best at ×1.0 for the board metric | Pearson ignores scale; least squares already maximises it |

**Lesson:** gains came from what the members see, never from reshaping what they output.

## 8. Learning from the unlabelled test set
| change | result | lesson |
|---|---|---|
| self-training on the pipeline's own test predictions (simulated on held-out topics) | 0.0015 better, below the bar | refitting on its own outputs moves the model only slightly |
| erasing the prompt topic from text embeddings (LEACE) | better on unseen prompts, worse on familiar ones; public 0.3272 vs 0.3280 | topic alone explains only 4% of the grade, so erasing it removes little |

## 9. Mistakes in the process, and their fixes
- **The unseen-prompt check ran before the correction stage**, so corrections were never tested on new prompts. The
  batch-specific slope exploited exactly that gap. The check now runs after every stage.
- **A leaky column was rewarded for over a week** because the first CV used random folds. Speaker-grouped folds exposed it.
- **The combination pipeline's settings weren't saved with its runs.** They had to be reconstructed (and verified exactly)
  when a sanity check needed them. They're now in the log.
- **A summary stated an unmeasured overlap** (the KAN family). Amrita challenged it; the measurement corrected half of it.
- **Two experiment loops run at once exhausted the machine's process slots**, cutting some runs short. Now one loop at a time.
