"""Where you stand: your submissions, the public leaderboard, and how public score tracks CV.

    python scripts/scores.py
Also fills in public scores for ledger runs that were still being scored at submit time.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import _bootstrap  # noqa: E402,F401  — switches to .venv's python if needed

import re
import shutil
import zipfile

import pandas as pd

from common import CACHE, KaggleError, col, kaggle, load_config, my_submissions, read_ledger, write_ledger  # noqa: E402

num = lambda s: float(s) if re.fullmatch(r"\d+(\.\d+)?", (s or "").strip()) else None


def leaderboard(comp: str) -> pd.DataFrame | None:
    dest = CACHE / "lb"
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    proc = kaggle(["competitions", "leaderboard", comp, "-d", "-p", str(dest), "-q"], timeout=300, check=False)
    if proc.returncode != 0:
        print(f"leaderboard download failed: {(proc.stderr or proc.stdout).strip()[-300:]}")
        return None
    for z in dest.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
    files = sorted(dest.rglob("*.csv"))
    return pd.read_csv(files[0]) if files else None


def main():
    cfg = load_config()
    comp = cfg["COMP"]
    try:
        subs = my_submissions(comp)
    except KaggleError as e:
        sys.exit(str(e))

    print(f"your submissions to {comp} (newest first):")
    for s in subs[:12]:
        print(f"  {col(s, 'date')[:16]:<17} {col(s, 'status').split('.')[-1].lower()[:10]:<11} "
              f"public {col(s, 'publicscore', 'public') or '-':<8} {col(s, 'description')[:70]}")
    scores = [v for v in (num(col(s, "publicscore", "public")) for s in subs) if v is not None]
    best = min(scores) if scores else None

    rows, changed = read_ledger(), False
    for r in rows:
        if r.get("submitted_utc") and not r.get("public_lb"):
            hit = [s for s in subs if col(s, "description").startswith(f"run {r['run_id']}")]
            if hit and num(col(hit[0], "publicscore", "public")) is not None:
                r["public_lb"] = col(hit[0], "publicscore", "public"); changed = True
    if changed:
        write_ledger(rows)
        print("ledger: filled in public scores for runs that finished scoring")

    lb = leaderboard(comp)
    if lb is not None and best is not None:
        score_col = next((c for c in lb.columns if c.lower() == "score"),
                         next(c for c in lb.columns if "score" in c.lower()))
        team_col = next((c for c in lb.columns if "teamname" in c.lower()), None)
        lb = lb.dropna(subset=[score_col]).sort_values(score_col).reset_index(drop=True)
        mine = None
        if cfg["TEAM_NAME"] and team_col:
            hit = lb.index[lb[team_col].astype(str).str.lower() == cfg["TEAM_NAME"].lower()]
            mine = int(hit[0]) if len(hit) else None
        rank = (mine + 1) if mine is not None else int((lb[score_col] < best).sum()) + 1
        print(f"\npublic leaderboard: {len(lb)} teams   your best {best:.4f}   rank ~{rank}")
        print(lb.head(5)[[c for c in (team_col, score_col) if c]].to_string(index=False))
        ahead = lb[lb[score_col] < best]
        if len(ahead):
            print(f"next team up scores {ahead[score_col].iloc[-1]:.4f} "
                  f"(gap {best - ahead[score_col].iloc[-1]:.4f})")

    done = [r for r in rows if r.get("public_lb")]
    if done:
        print("\nloop submissions — CV vs public:")
        for r in done:
            print(f"  {r['run_id']}  CV {float(r['cv']):.4f}  public {float(r['public_lb']):.4f}  "
                  f"ratio {float(r['public_lb']) / float(r['cv']):.3f}  {r['desc'][:50]}")
    print("\nreminder: the public board scores a slice of 216 clips; one clip landing a quarter point "
          "differently moves it by ~0.002-0.005. Decide on CV.")


if __name__ == "__main__":
    try:
        main()
    except KaggleError as e:
        sys.exit(f"kaggle error: {e}")
