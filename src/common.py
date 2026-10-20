"""Shared helpers: config, paths, metric, blend, experiment ledger, Kaggle CLI wrapper."""
from __future__ import annotations

import csv
import datetime as dt
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
RUNS = ROOT / "runs"
LEDGER = ROOT / "experiments.csv"
CLIP = (1.5, 5.0)            # only 4 training clips sit below 2.0
FULL_SEEDS = (42, 1337, 2024)

# cv / cv_per_seed hold the gate metric: cv_testmix on grouped folds, plain RMSE on the old random folds.
LEDGER_COLS = ["run_id", "utc", "desc", "seeds", "cv", "cv_per_seed", "cv_spread",
               "verdict", "commit", "submitted_utc", "public_lb",
               "folds", "cv_grouped", "cv_grouped_per_seed", "cv_short", "cv_long", "cv_newtopic"]

_DEFAULTS = {"COMP": "shl-hiring-assessment-2026", "KAGGLE_USER": "", "MAIN_KERNEL": "",
             "SUBMIT_MODE": "csv", "MAX_SUBS_PER_DAY": "3", "MIN_GAIN": "0.002", "TEAM_NAME": ""}


def load_config() -> dict:
    cfg = dict(_DEFAULTS)
    path = ROOT / "config.env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def fit_weighted(model, X, y, sw=None, **kw):
    """model.fit with sample weights routed to the final estimator (pipelines, GridSearchCV); unweighted fit if the
    estimator doesn't take sample_weight (e.g. kNN). sw None = plain fit."""
    if sw is None:
        return model.fit(X, y, **kw)
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    est = model.estimator if isinstance(model, GridSearchCV) else model
    key = f"{est.steps[-1][0]}__sample_weight" if isinstance(est, Pipeline) else "sample_weight"
    try:
        return model.fit(X, y, **{key: sw}, **kw)
    except TypeError:
        return model.fit(X, y, **kw)


def wrmse(a, b, w) -> float:
    """RMSE with per-clip weights (the test-mix metric when w are the batch weights)."""
    a, b, w = np.asarray(a, float), np.asarray(b, float), np.asarray(w, float)
    return float(np.sqrt(np.sum(w * (a - b) ** 2) / np.sum(w)))


def hill_climb(O, y, iters=200, steps=(.5, .35, .2, .1, .05, .02), w=None):
    """Greedy convex blend over candidate columns; same as the notebook's §9. w: optional clip weights."""
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    wn = w / w.sum()
    score = lambda P: np.sqrt((((P - y[:, None]) ** 2) * wn[:, None]).sum(0))
    single = score(O)
    j = int(single.argmin())
    cur, best = O[:, j].copy(), float(single[j])
    wt = np.zeros(O.shape[1]); wt[j] = 1.0
    for _ in range(iters):
        moved = False
        for a in steps:
            cand = (1 - a) * cur[:, None] + a * O
            sc = score(cand)
            k = int(sc.argmin())
            if sc[k] < best - 1e-7:
                cur, best = cand[:, k], float(sc[k])
                wt = (1 - a) * wt; wt[k] += a; moved = True
                break
        if not moved:
            break
    return wt, best


# ---- experiment ledger -------------------------------------------------------
def read_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    with LEDGER.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_ledger(rows: list[dict]) -> None:
    with LEDGER.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=LEDGER_COLS, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({c: r.get(c, "") for c in LEDGER_COLS})


def seeds_key(seeds) -> str:
    return " ".join(str(s) for s in seeds)


def per_seed(row: dict) -> list[float]:
    return [float(x) for x in row["cv_per_seed"].split("|")] if row.get("cv_per_seed") else []


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# ---- Kaggle CLI --------------------------------------------------------------
class KaggleError(RuntimeError):
    pass


def kaggle(args: list[str], timeout: int = 600, check: bool = True) -> subprocess.CompletedProcess:
    """Run the Kaggle CLI from this interpreter's environment, so it works without PATH setup."""
    if importlib.util.find_spec("kaggle") is not None:
        exe = [sys.executable, "-m", "kaggle"]
    elif shutil.which("kaggle"):
        exe = [shutil.which("kaggle")]
    else:
        raise KaggleError("kaggle CLI not found. Run `pip install -U kaggle` inside the venv, "
                          "then `kaggle auth login` (or set KAGGLE_API_TOKEN).")
    proc = subprocess.run([*exe, *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, env=os.environ.copy())
    if check and proc.returncode != 0:
        raise KaggleError(f"`kaggle {' '.join(args)}` failed ({proc.returncode}):\n"
                          f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
    return proc


def unrecognized_flag(proc: subprocess.CompletedProcess) -> bool:
    """argparse rejects unknown flags before doing anything, so a retry is safe."""
    return proc.returncode != 0 and "unrecognized arguments" in (proc.stderr + proc.stdout)


def parse_csv_output(text: str) -> list[dict]:
    """Parse `kaggle ... -v` CSV output, skipping any warning lines before the header."""
    lines = text.strip().splitlines()
    for i, ln in enumerate(lines):
        if "," in ln and not ln.lower().startswith(("warning", "next page")):
            return list(csv.DictReader(lines[i:]))
    return []


def col(row: dict, *needles: str) -> str:
    """Value of the first column whose name contains any needle (case-insensitive)."""
    for k, v in row.items():
        if k and any(n.lower() in k.lower() for n in needles):
            return (v or "").strip()
    return ""


def my_submissions(comp: str) -> list[dict]:
    proc = kaggle(["competitions", "submissions", comp, "-v", "-q"], timeout=120)
    return parse_csv_output(proc.stdout)
