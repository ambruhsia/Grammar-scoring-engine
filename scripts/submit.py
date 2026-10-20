"""Guarded submission. The ONLY way the agent submits (raw `kaggle competitions submit` is denied).

    python scripts/submit.py                 # best accepted full-CV run not yet submitted
    python scripts/submit.py 1011-142233     # a specific run
    python scripts/submit.py --dry-run       # run every check, print the command, submit nothing
    python scripts/submit.py --kernel-version 17   # SUBMIT_MODE=kernel (code competition)

Refuses unless:
  * the run was scored on the full 3 fold seeds and its verdict is accepted
    (a baseline only when named explicitly)
  * its CV beats every run already submitted from this ledger by at least MIN_GAIN
  * the submission file matches test.csv exactly (216 rows, same filenames, finite, within [1, 5])
  * fewer than MAX_SUBS_PER_DAY submissions were made today (UTC, counted from Kaggle itself)
Only a human can override the CV rule: ALLOW_FORCE=1 python scripts/submit.py RUN --force
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import _bootstrap  # noqa: E402,F401  — switches to .venv's python if needed

import argparse
import os
import re

import numpy as np
import pandas as pd

from common import (CACHE, FULL_SEEDS, RUNS, KaggleError, col, kaggle, load_config,  # noqa: E402
                    my_submissions, read_ledger, seeds_key, unrecognized_flag, utc_now)


def refuse(msg: str):
    print(f"REFUSED  {msg}")
    sys.exit(2)


def submissions_today(comp: str, rows: list[dict]) -> int:
    today = utc_now().strftime("%Y-%m-%d")
    try:
        subs = my_submissions(comp)
        return sum(1 for s in subs if col(s, "date")[:10] == today)
    except KaggleError as e:
        print(f"could not list Kaggle submissions ({str(e).splitlines()[0]}); counting from the ledger")
        return sum(1 for r in rows if r.get("submitted_utc", "")[:10] == today)


def score(row: dict) -> float:
    """The run's primary CV: cv_testmix when the ledger records it, otherwise cv."""
    return float(row.get("cv_testmix") or row["cv"])


def current_scheme(rows: list[dict], full: str) -> list[dict]:
    """Rows since the latest full-seed baseline. A rebaseline starts a new validation scheme
    (e.g. random → speaker-grouped folds), and scores from different schemes aren't comparable.
    If the ledger has a `scheme` column, the window is further limited to the baseline's scheme."""
    starts = [i for i, r in enumerate(rows) if r["seeds"] == full and r["verdict"] == "baseline"]
    window = rows[starts[-1]:] if starts else rows
    if starts and window[0].get("scheme"):
        window = [r for r in window if r.get("scheme") == window[0]["scheme"]]
    return window


def save_ledger(rows: list[dict]) -> None:
    """Like write_ledger, but keeps any columns run.py added (cv_testmix, scheme, …)."""
    import csv
    from common import LEDGER, LEDGER_COLS
    cols = list(LEDGER_COLS) + [k for r in rows for k in r if k not in LEDGER_COLS]
    cols = list(dict.fromkeys(cols))
    with LEDGER.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({c: r.get(c, "") for c in cols})


def validate(path: Path) -> pd.DataFrame:
    if not path.exists():
        refuse(f"{path} does not exist")
    sub, test = pd.read_csv(path), pd.read_csv(CACHE / "test.csv")
    if list(sub.columns) != ["filename", "label"]:
        refuse(f"columns are {list(sub.columns)}, expected ['filename', 'label']")
    if len(sub) != len(test) or set(sub.filename) != set(test.filename):
        refuse(f"{len(sub)} rows / filenames do not match test.csv ({len(test)} rows)")
    if not np.isfinite(sub.label).all() or sub.label.min() < 1 or sub.label.max() > 5:
        refuse("labels must be finite and within [1, 5]")
    return sub


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--message", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--kernel-version", type=int, help="code competitions: notebook version to submit")
    ap.add_argument("--force", action="store_true", help="human override of the CV rule (needs ALLOW_FORCE=1)")
    args = ap.parse_args()
    comp, min_gain, cap = cfg["COMP"], float(cfg["MIN_GAIN"]), int(cfg["MAX_SUBS_PER_DAY"])
    forced = args.force and os.environ.get("ALLOW_FORCE") == "1"
    if args.force and not forced:
        refuse("--force only works when a human runs it with ALLOW_FORCE=1")

    rows = read_ledger()
    full = seeds_key(FULL_SEEDS)
    scheme = current_scheme(rows, full)
    if args.run_id:
        match = [r for r in rows if r["run_id"] == args.run_id]
        if not match:
            refuse(f"run {args.run_id} not in experiments.csv")
        row = match[0]
    else:
        pool = [r for r in scheme if r["seeds"] == full and r["verdict"] == "accepted"
                and not r.get("submitted_utc")]
        if not pool:
            refuse("no unsubmitted accepted full-CV run under the current validation scheme")
        row = min(pool, key=score)

    cv = score(row)
    if not forced:
        if not any(r is row for r in scheme):
            refuse(f"run {row['run_id']} predates the current baseline ({scheme[0]['run_id']}); "
                   f"its CV is on a different validation scheme")
        if row["seeds"] != full:
            refuse(f"run {row['run_id']} was scored on seeds [{row['seeds']}]; submit full-CV runs only")
        if not (row["verdict"] == "accepted" or (row["verdict"] == "baseline" and args.run_id)):
            refuse(f"run {row['run_id']} verdict is {row['verdict']}; only accepted runs are submitted "
                   f"(a baseline only when named explicitly)")
        if row.get("submitted_utc"):
            refuse(f"run {row['run_id']} was already submitted ({row['public_lb'] or 'score pending'})")
        # Compare only with submissions scored on the same scheme. The first accepted run after a
        # rebaseline already beat that baseline by MIN_GAIN, so it may go out.
        done = [r for r in scheme if r.get("submitted_utc")]
        if done:
            best = min(done, key=score)
            if cv > score(best) - min_gain:
                refuse(f"CV {cv:.4f} does not beat best submitted run {best['run_id']} "
                       f"(CV {score(best):.4f}, same scheme) by {min_gain}")

    mode = cfg["SUBMIT_MODE"].lower()
    path = RUNS / row["run_id"] / "submission.csv"
    if mode == "csv":
        validate(path)
    elif not (args.kernel_version and cfg["MAIN_KERNEL"]):
        refuse("SUBMIT_MODE=kernel needs MAIN_KERNEL in config.env and --kernel-version N "
               "(the pushed notebook version that produced this run's predictions)")

    used = submissions_today(comp, rows)
    if used >= cap:
        refuse(f"{used} submissions today (UTC) — budget is {cap}. Resets 00:00 UTC.")

    msg = args.message or f"run {row['run_id']} cv {cv:.4f} | {row['desc'][:80]}"
    cmd = ["competitions", "submit", comp, "-f", "submission.csv" if mode == "kernel" else str(path), "-m", msg]
    if mode == "kernel":
        cmd += ["-k", cfg["MAIN_KERNEL"], "-v", str(args.kernel_version)]
    print(f"run {row['run_id']}  CV {cv:.4f}  — submission {used + 1}/{cap} today")
    if args.dry_run:
        print("DRY RUN  kaggle " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        return

    proc = kaggle(cmd + ["--wait", "540"], timeout=600, check=False)
    if unrecognized_flag(proc):
        proc = kaggle(cmd, timeout=300, check=False)
    print(proc.stdout[-800:])
    if proc.returncode != 0:
        print(proc.stderr[-800:])
        if re.search(r"code competition|notebook|kernel", proc.stderr + proc.stdout, re.I):
            print("hint: this may be a code competition — set SUBMIT_MODE=kernel in config.env")
        sys.exit(1)

    row["submitted_utc"] = utc_now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        mine = [s for s in my_submissions(comp) if col(s, "description").startswith(msg[:40])]
        public = col(mine[0], "publicscore", "public") if mine else ""
    except KaggleError:
        public = ""
    row["public_lb"] = public if re.fullmatch(r"\d+(\.\d+)?", public or "") else ""
    save_ledger(rows)
    print(f"SUBMITTED  run {row['run_id']}  CV {cv:.4f}  public {row['public_lb'] or 'pending — run scripts/scores.py later'}")


if __name__ == "__main__":
    try:
        main()
    except KaggleError as e:
        sys.exit(f"kaggle error: {e}")
