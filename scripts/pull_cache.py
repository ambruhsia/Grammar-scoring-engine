"""Bring the notebook's saved outputs and the competition CSVs into cache/.

    python scripts/pull_cache.py                      # MAIN_KERNEL from config.env, latest version
    python scripts/pull_cache.py --slug ambruhsia/my-notebook/14   # a specific version
    python scripts/pull_cache.py --all                # also the *.npz embedding caches (large)

The notebook writes /kaggle/working/features.csv after every stage, so the saved version
already holds the final modelling table, every kept *_pred column included. No GPU re-run needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import _bootstrap  # noqa: E402,F401  — switches to .venv's python if needed

import argparse
import shutil
import zipfile

from common import CACHE, KaggleError, col, kaggle, load_config, parse_csv_output, unrecognized_flag  # noqa: E402

SMALL = r"features\.csv$|submission\.csv$|asr_c\.jsonl$|ctc\.jsonl$|\.log$"
LARGE = SMALL + r"|\.npz$"


def unzip_all(folder: Path) -> None:
    for z in folder.rglob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(z.parent)
        z.unlink()


def pull_version_outputs(owner: str, kernel: str, version: str, pattern: str, dest: Path) -> None:
    """The CLI parses owner/slug/N but never sends N (it always returns the latest run, even a failed
    one), so call the API directly. It wants the version label as 'v<N>'."""
    import re
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as k:
        req = ApiListKernelSessionOutputRequest()
        req.user_name, req.kernel_slug, req.version_label, req.page_size = owner, kernel, f"v{version}", 200
        try:
            files = list(k.kernels.kernels_api_client.list_kernel_session_output(req).files or [])
        except requests.HTTPError as e:
            raise KaggleError(f"no version {version} of {owner}/{kernel} ({e})")
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        if re.search(pattern, f.file_name):
            r = requests.get(f.url, timeout=1800)
            r.raise_for_status()
            (dest / f.file_name).parent.mkdir(parents=True, exist_ok=True)
            (dest / f.file_name).write_bytes(r.content)


def pull_kernel_outputs(slug: str, pattern: str, dest: Path) -> None:
    parts = slug.split("/")
    if len(parts) == 3:
        return pull_version_outputs(*parts, pattern, dest)
    proc = kaggle(["kernels", "output", slug, "-p", str(dest), "-o", "--file-pattern", pattern],
                  timeout=1800, check=False)
    if unrecognized_flag(proc):
        print("this kaggle CLI has no --file-pattern; downloading every output file "
              "(run `pip install -U kaggle` to avoid this)")
        proc = kaggle(["kernels", "output", slug, "-p", str(dest), "-o"], timeout=3600, check=False)
    if proc.returncode != 0:
        raise KaggleError(f"kernels output failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")


def pull_competition_csvs(comp: str, dest: Path) -> None:
    names = []
    proc = kaggle(["competitions", "files", comp, "-v", "-q"], timeout=120, check=False)
    if proc.returncode == 0:
        names = [col(r, "name", "ref") for r in parse_csv_output(proc.stdout)]
    for want in ("train.csv", "test.csv"):
        cands = [n for n in names if n and Path(n).name.lower() == want]
        cands += [f"Dataset_Final/{want}", want]
        for name in dict.fromkeys(cands):
            p = kaggle(["competitions", "download", comp, "-f", name, "-p", str(dest / "_comp"), "-o", "-q"],
                       timeout=300, check=False)
            if p.returncode == 0:
                break
        else:
            raise KaggleError(f"could not download {want} from {comp}. Have you joined the competition "
                              f"and accepted its rules on the website?")
    unzip_all(dest / "_comp")
    for want in ("train.csv", "test.csv"):
        found = sorted((dest / "_comp").rglob(want), key=lambda p: len(p.parts))
        shutil.copy(found[0], dest / want)
    shutil.rmtree(dest / "_comp", ignore_errors=True)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default=cfg["MAIN_KERNEL"], help="owner/kernel-slug[/version]")
    ap.add_argument("--all", action="store_true", help="also pull *.npz embedding caches")
    args = ap.parse_args()
    if not args.slug:
        sys.exit("set MAIN_KERNEL in config.env (find it with: kaggle kernels list --mine)")

    CACHE.mkdir(exist_ok=True)
    stage = CACHE / "_kernel"
    shutil.rmtree(stage, ignore_errors=True)
    print(f"pulling outputs of {args.slug} ...")
    pull_kernel_outputs(args.slug, LARGE if args.all else SMALL, stage)
    unzip_all(stage)
    got = {p.name: p for p in sorted(stage.rglob("*"), key=lambda p: len(p.parts)) if p.is_file()}
    if "features.csv" not in got:
        sys.exit(f"no features.csv in {args.slug}'s output (files: {sorted(got)[:20]}). Pick the saved "
                 f"version that ran the whole notebook (Save Version -> Save & Run All).")
    for name, p in got.items():
        target = "notebook_submission.csv" if name == "submission.csv" else name
        shutil.move(str(p), CACHE / target)
    shutil.rmtree(stage, ignore_errors=True)

    print(f"pulling train.csv / test.csv from {cfg['COMP']} ...")
    pull_competition_csvs(cfg["COMP"], CACHE)

    import pandas as pd
    F = pd.read_csv(CACHE / "features.csv")
    tr, te = pd.read_csv(CACHE / "train.csv"), pd.read_csv(CACHE / "test.csv")
    preds = [c for c in F.columns if c.endswith("_pred")]
    obj = [c for c in F.columns if F[c].dtype == object and c not in ("split", "filename")]
    print(f"\nfeatures.csv  {F.shape}   train rows {(F.split == 'train').sum()}  test rows {(F.split == 'test').sum()}")
    print(f"learned columns ({len(preds)}): {', '.join(preds)}")
    print(f"train.csv {len(tr)} rows ({(tr.label > 0).sum()} with label > 0)   test.csv {len(te)} rows")
    if obj:
        print(f"!! non-numeric feature columns will break the models: {obj}")
    print("files in cache/:", ", ".join(sorted(p.name for p in CACHE.iterdir() if p.is_file())))


if __name__ == "__main__":
    try:
        main()
    except KaggleError as e:
        sys.exit(f"kaggle error: {e}")
