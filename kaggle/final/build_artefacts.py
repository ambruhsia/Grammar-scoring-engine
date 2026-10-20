"""Collect what the final notebook reads into kaggle/final/artefacts/ (the private Kaggle dataset shl-grammar-final-artefacts).

    python kaggle/final/build_artefacts.py
The final pick (hybrid11) averages 8 runs of the experiment loop: the base pipeline and the combination pipeline, each on four
sets of speaker-grouped fold seeds. For each run the notebook replays the stacking from its candidate matrices (cands.npz) and
checks the result against the run's out-of-fold and submission files. The large audio embeddings are not copied: the notebook
gets them from the existing caches dataset (shl-grammar-caches-v18). The spaCy model the rubric layer used (en_core_web_sm 3.8.0,
MIT) is added as a zip, for sessions where it isn't installed.
"""
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "kaggle" / "final" / "artefacts"
BASE_RUNS = ("1015-070513", "1015-114531", "1015-165432", "1015-172136")
COMBO_RUNS = ("1016-100139", "1016-140135", "1016-140601", "1016-141007")
FINAL_FILE = ROOT / "runs" / "probes" / "hybrid11.csv"       # the final pick (pick 2 = hybrid11 - 0.041)


def main():
    if OUT.exists():                                  # rebuilt from scratch: no file of an earlier pick survives
        meta = OUT / "dataset-metadata.json"
        keep = meta.read_bytes() if meta.exists() else None
        shutil.rmtree(OUT)
        OUT.mkdir(parents=True)
        if keep is not None:
            meta.write_bytes(keep)
    OUT.mkdir(parents=True, exist_ok=True)
    files = {"chosen_submission.csv": FINAL_FILE, "pick2_submission.csv": ROOT / "runs" / "probes" / "pubmix_c.csv",
             "folds_all.csv": ROOT / "kaggle" / "final" / "folds_all.csv"}
    for r in BASE_RUNS + COMBO_RUNS:
        run = ROOT / "runs" / r
        files.update({f"cands_{r}.npz": run / "cands.npz", f"oof_{r}.csv": run / "oof.csv", f"sub_{r}.csv": run / "submission.csv",
                      f"summary_{r}.json": run / "summary.json"})
    files.update({
        "features.csv": ROOT / "cache" / "features.csv", "speaker_groups.csv": ROOT / "cache" / "speaker_groups.csv",
        "insight.csv": ROOT / "cache" / "insight.csv", "rubric_feats.csv": ROOT / "cache" / "rubric_feats.csv",
        "emb_mpnet.npz": ROOT / "cache" / "emb_mpnet.npz", "qwen_judge.csv": ROOT / "cache" / "qwen" / "qwen_judge.csv",
        "qwen_gen.jsonl": ROOT / "cache" / "qwen" / "qwen_gen.jsonl",
        "qwen35_judge.csv": ROOT / "cache" / "qwen35" / "qwen35_judge.csv",
        "qwen35_judge2.csv": ROOT / "cache" / "qwen35" / "qwen35_judge2.csv",
        "sentinel_guard.csv": ROOT / "reports" / "phaseA" / "sentinel_guard.csv",
        "selection.csv": ROOT / "reports" / "phaseB" / "selection.csv", "rubric_tests.csv": ROOT / "reports" / "phaseG" / "rubric_tests.csv",
        "asr_c.jsonl": ROOT / "cache" / "asr_c.jsonl", "ctc.jsonl": ROOT / "cache" / "ctc.jsonl",
        "spk_xvec.npz": ROOT / "cache" / "spk_xvec.npz", "train.csv": ROOT / "cache" / "train.csv", "test.csv": ROOT / "cache" / "test.csv",
    })
    total = 0
    for name, src in files.items():
        shutil.copy2(src, OUT / name)
        total += (OUT / name).stat().st_size
    import en_core_web_sm
    model = Path(en_core_web_sm.__file__).parent / f"en_core_web_sm-{en_core_web_sm.__version__}"
    zpath = OUT / f"spacy_{model.name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in model.rglob("*"):
            if f.is_file():
                zf.write(f, Path(model.name) / f.relative_to(model))
    total += zpath.stat().st_size
    print(f"{len(files) + 1} files, {total / 1e6:.1f} MB in {OUT}")


if __name__ == "__main__":
    main()
