"""Run a notebook on Kaggle's GPUs and bring its outputs back.

    python scripts/gpu_job.py list                             # your notebooks (find slugs)
    python scripts/gpu_job.py fetch ambruhsia/my-notebook      # notebook source + metadata -> kaggle/main/
    python scripts/gpu_job.py push kaggle/main                 # push a new version and start it
    python scripts/gpu_job.py push kaggle/main --accelerator NvidiaTeslaT4
    python scripts/gpu_job.py status ambruhsia/my-notebook
    python scripts/gpu_job.py wait ambruhsia/my-notebook       # blocks up to 9 min; exit 0 done, 3 still running
    python scripts/gpu_job.py pull ambruhsia/my-notebook       # outputs -> cache/ (= pull_cache.py --slug)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import _bootstrap  # noqa: E402,F401  — switches to .venv's python if needed

import argparse
import json
import subprocess
import time

from common import KaggleError, ROOT, kaggle, load_config  # noqa: E402

DONE, FAILED = ("complete",), ("error", "cancel", "fail")


def check_metadata(folder: Path, comp: str) -> str:
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    problems = []
    if str(meta.get("enable_gpu", "false")).lower() != "true":
        problems.append('enable_gpu is not "true"')
    if str(meta.get("enable_internet", "false")).lower() != "true":
        problems.append('enable_internet is not "true" (Hugging Face model downloads need it)')
    if comp not in meta.get("competition_sources", []):
        problems.append(f"competition_sources does not include {comp}")
    if not (folder / meta.get("code_file", "")).exists():
        problems.append(f"code_file {meta.get('code_file')} not found in {folder}")
    if problems:
        sys.exit("kernel-metadata.json needs fixing:\n  - " + "\n  - ".join(problems))
    if not meta.get("kernel_sources"):
        print("warning: no kernel_sources — the cached transcripts/embeddings won't be attached, "
              "so every stage recomputes (hours). Add the notebook whose output holds the caches.")
    return meta["id"]


def state(slug: str) -> str:
    out = kaggle(["kernels", "status", slug], timeout=60).stdout.lower().replace(slug.lower(), "")
    for word in DONE + FAILED + ("running", "queued"):
        if word in out:
            return word
    return out.strip()[-80:]


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push"); p.add_argument("folder"); p.add_argument("--accelerator")
    p.add_argument("--timeout", type=int, help="max run seconds")
    for name in ("status", "wait", "pull", "fetch"):
        q = sub.add_parser(name); q.add_argument("slug")
        if name == "wait":
            q.add_argument("--minutes", type=float, default=9.0)
        if name == "fetch":
            q.add_argument("dest", nargs="?", default="kaggle/main")
    sub.add_parser("list")
    args = ap.parse_args()

    if args.cmd == "list":
        print(kaggle(["kernels", "list", "--mine", "--page-size", "50"], timeout=120).stdout.strip())
    elif args.cmd == "fetch":
        dest = (ROOT / args.dest).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        print(kaggle(["kernels", "pull", args.slug, "-p", str(dest), "-m"], timeout=300).stdout.strip())
        print(f"notebook and kernel-metadata.json in {dest}. Check the metadata before pushing.")
    elif args.cmd == "push":
        folder = (ROOT / args.folder).resolve()
        slug = check_metadata(folder, cfg["COMP"])
        cmd = ["kernels", "push", "-p", str(folder)]
        if args.accelerator:
            cmd += ["--accelerator", args.accelerator]
        if args.timeout:
            cmd += ["-t", str(args.timeout)]
        print(kaggle(cmd, timeout=600).stdout.strip())
        print(f"pushed {slug}. Next: python scripts/gpu_job.py wait {slug}")
    elif args.cmd == "status":
        print(f"{args.slug}: {state(args.slug)}")
    elif args.cmd == "wait":
        end = time.time() + args.minutes * 60
        while True:
            s = state(args.slug)
            if s in DONE or any(f in s for f in FAILED):
                print(f"{args.slug}: {s}")
                sys.exit(0 if s in DONE else 1)
            if time.time() > end:
                print(f"{args.slug}: still {s} — call wait again (or /loop 15m on the status command)")
                sys.exit(3)
            time.sleep(60)
    elif args.cmd == "pull":
        sys.exit(subprocess.call([sys.executable, str(ROOT / "scripts" / "pull_cache.py"), "--slug", args.slug]))


if __name__ == "__main__":
    try:
        main()
    except KaggleError as e:
        sys.exit(f"kaggle error: {e}")
