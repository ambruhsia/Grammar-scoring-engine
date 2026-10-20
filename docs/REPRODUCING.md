# Reproducing the results

Everything is scripted, but **no competition data ships with this repository**. Download the data on Kaggle under the
competition's rules, then rebuild the intermediate caches (transcripts, embeddings, member predictions) with the jobs below.
The jobs behind the final model take about 15 GPU-hours on Kaggle T4s (7 of them for the original notebook from cold),
plus a few hours of local CPU. Every job tried, kept or not, adds up to about 25 GPU-hours.

## 1. Environment
- Python 3.11+, then `pip install -r requirements.txt`. `torch` and `transformers` (CPU builds are enough) are needed by the
  Rank-N-Contrast, attention-pooling and KAN modules; `pykan==0.2.8` only by the KAN experiments.
- The Kaggle API, authenticated (`kaggle auth login` or a token).
- Copy `config.env.example` to `config.env` and fill in the competition slug, your Kaggle user and the notebook slug.
- Developed on Windows with Git Bash; the scripts switch to `.venv` by themselves.

## 2. GPU stages on Kaggle
Each folder under `kaggle/` is a Kaggle notebook (`kernel-metadata.json` + source). Push it with
`kaggle kernels push -p kaggle/<job>`, or use `python scripts/gpu_job.py push|status|pull`. When you convert a `.py` source
with jupytext, set the notebook's kernelspec to `python3` yourself, or papermill fails at once.

| job | produces | T4 time | in the final model |
|---|---|---|---|
| `main` (the original notebook) | Whisper + CTC transcripts, 21 handcrafted + grammar-aware features, frozen speech-encoder layers, DeBERTa, x-vectors → `features.csv` + caches | ~7 h cold, ~45 min with caches | yes |
| `deberta_grouped` | DeBERTa-v3-base on Whisper text and on the (CTC raw, Whisper clean) pair, speaker-grouped folds | ~1.5 h | yes |
| `qwen` | Qwen2.5-7B rubric-prompt hidden states, zero-shot judge, minimal corrections | 53 min | yes (heads) |
| `verbatim` | Whisper-large-v3-turbo verbatim transcripts, Qwen states on them | ~1 h | yes |
| `voxtral`, `voxtral_frames` | Voxtral-Mini-3B states: audio-first view, then question-first views and frames | 17 + 52 min | yes (3 views) |
| `laya_checklist` | Laya yes/no checklist answers per sentence | 11 min | yes |
| `deberta_large_pair`, `deberta_large_pair_s2` | DeBERTa-v3-large on the pair, seeds 42/7/2024 and 11/22/33 (6-seed column) | 66 min + ~1 h | yes |
| `qwen35_judge`, `qwen35_judge2` | Qwen3.5-9B examiner: rubric / accuracy / complexity / completeness / CEFR × transcripts, plus answer states | 27 + 43 min | yes (correction) |
| `crop30`, `laya`, `laya_ft`, `laya_dh`, `deberta_large_vpair`, `vox_attn_gpu`, `qwen35_judge3`, `qwen35_judge4`, `qwen35_pairwise`, `phi4_judge`, `deberta_perseed`, `denoise_asr`, `audio_quality` (CPU) | experiments that were tested and not kept (see [WHAT_FAILED.md](WHAT_FAILED.md)) | 5 min – 3.5 h each | no |

A few job sources embedded per-clip tables when they ran: anchors with their grades, fold tables, a training table. The
public copies replace those with a note saying how the table is rebuilt. `kaggle/qwen35_pairwise/make_anchors.py`
regenerates its anchors from the local data. The denoise job needs its clip list rebuilt first (every short clip, plus
69 random long clips from each split) as `clips.json`; then `kaggle/denoise_asr/build.py` assembles the notebook.

## 3. The local experiment loop
```bash
python scripts/pull_cache.py               # the main notebook's outputs → cache/ (add --all for the large .npz files)
python src/speaker_groups.py               # x-vector clustering → cache/speaker_groups.csv (503 speakers)
python src/insight.py                      # prompt topics and other per-clip insight columns → cache/insight.csv
python src/run.py --desc "base, main seeds"                    # the base pipeline (experiment.py defaults)
python src/run.py --seeds 7 99 555 --desc "base, fresh seeds"
EXP_OVERRIDES='{"RNC_HEAD": true, "SVR_HEADS": true, "DEBERTA_LARGE": "replace", "VERBATIM_DEBERTA": true,
  "QWEN35_HEADS": ["qwen35_verbatim_pred", "qwen35_ctc_pred", "qwen35_whisper_pred"], "VOXTRAL": [], "VOXTRAL3": true,
  "BLENDER": "nnls_batch_shrink", "RESID_CORR": true, "POOL_S": 0.3}' python src/run.py --desc "combination"
```
- `run.py` writes `runs/<run_id>/`: the submission, out-of-fold predictions, candidate matrices and a code snapshot.
- It adds a row to `experiments.csv` and prints cv_grouped, cv_testmix, short / long, the unseen-prompt check after the
  correction stage, and a paired speaker-group bootstrap against the reference run.
- Every experiment is one flag in `src/experiment.py`, switched on with `EXP_OVERRIDES`, so any logged run can be
  re-created.
- The final model uses the base and combination pipelines on fold seeds 42/1337/2024, 7/99/555, 11/22/33 and 44/66/88.

`python src/variants.py` replays all eight runs from their saved candidate matrices. It checks each replay against the run's
files, rebuilds `hybrid11`, and evaluates post-processing variants without retraining any member.
`python src/paired_bootstrap.py <new_run> <old_run>` compares any two runs.

## 4. The final notebook
`kaggle/final/final_notebook.ipynb` (source `final_notebook.py`):
1. replays the eight runs behind `hybrid11` from their candidate matrices, collected by `build_artefacts.py` into a private
   Kaggle dataset;
2. checks each replay against the run's own files;
3. recomputes every metric;
4. writes `submission.csv` (pick 1) and `submission_pick2_offset.csv` (pick 2).

It runs on a CPU in under a minute. The GPU stages are ported into its appendix behind `RECOMPUTE_GPU = False`. The
artefacts dataset holds derived competition data, so it stays private. Rebuild it with `build_artefacts.py` after steps 2–3.
