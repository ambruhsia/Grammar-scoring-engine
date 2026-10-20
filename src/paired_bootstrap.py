"""Paired bootstrap over speaker groups for a CV comparison: is "new minus old" distinguishable from zero?

Speaker groups (cache/speaker_groups.csv) are resampled with replacement; on each draw both runs' out-of-fold predictions are
scored on the same resampled clips and the difference new - old is recorded. Clips of one speaker move together, because twin
clips are not independent evidence. Reports cv_testmix (each batch weighted to its test share) and the short / long batches:
the full-data difference, its 95% percentile interval and the share of draws where new is better.

The interval covers which clips and speakers we happened to get. It does not cover the fold assignment: compare runs on the
same fold seeds (or bags over several seed sets), and treat a seed-set-to-seed-set swing as a second, separate noise source.

usage: python src/paired_bootstrap.py <new run_id | oof csv> <old run_id | oof csv> [--draws 10000]
       (an oof csv needs columns filename, oof; a run_id reads runs/<id>/oof.csv)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, RUNS  # noqa: E402
from data import batch_weights  # noqa: E402


def paired(new, old, y, w, groups, draws=10000, seed=0, mask=None) -> dict:
    """Weighted-RMSE difference new - old with a speaker-group bootstrap. mask: score only these clips (e.g. one batch)."""
    m = np.ones(len(y), bool) if mask is None else np.asarray(mask, bool)
    g = pd.factorize(np.asarray(groups)[m])[0]
    wm = np.asarray(w, float)[m]
    en, eo = (np.asarray(new)[m] - y[m]) ** 2, (np.asarray(old)[m] - y[m]) ** 2
    k = g.max() + 1
    S = np.zeros((3, k))                                     # per group: sum w e_new^2, sum w e_old^2, sum w
    for i, v in enumerate((wm * en, wm * eo, wm)):
        np.add.at(S[i], g, v)
    full = np.sqrt(S[0].sum() / S[2].sum()) - np.sqrt(S[1].sum() / S[2].sum())
    rng = np.random.default_rng(seed)
    C = rng.multinomial(k, np.full(k, 1.0 / k), size=draws).astype(float)   # how often each group is drawn
    T = C @ S.T
    d = np.sqrt(T[:, 0] / T[:, 2]) - np.sqrt(T[:, 1] / T[:, 2])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": full, "lo": lo, "hi": hi, "p_better": float(np.mean(d < 0)), "sd": float(d.std()), "groups": k,
            "clips": int(m.sum())}


def table():
    """Train clips in the runs' row order (features.csv order) -> (filenames, y, w, groups, batch)."""
    F = pd.read_csv(CACHE / "features.csv", usecols=["split", "filename"])
    lab = pd.read_csv(CACHE / "train.csv")
    T = F[F.split == "train"].merge(lab[lab.label > 0], on="filename")
    TE = F[F.split == "test"]
    G = pd.read_csv(CACHE / "speaker_groups.csv")
    gtr, gte = G[G.split == "train"].set_index("filename"), G[G.split == "test"].set_index("filename")
    batch = gtr.loc[T.filename, "batch"].values
    w = batch_weights(batch, gte.loc[TE.filename, "batch"].values)
    return T.filename.values, T.label.values.astype(float), w, gtr.loc[T.filename, "group"].values, batch


def report(new, old, label="", draws=10000) -> dict:
    fn, y, w, groups, batch = table()
    out = {}
    for name, mask in (("testmix", None), ("short", batch == "short"), ("long", batch == "long")):
        out[name] = r = paired(new, old, y, w if name == "testmix" else np.ones(len(y)), groups, draws, mask=mask)
        print(f"{label:28s} {name:8s} new-old {r['diff']:+.4f}  95% [{r['lo']:+.4f}, {r['hi']:+.4f}]  "
              f"P(new better) {r['p_better']:.2f}")
    return out


def load_oof(ref: str) -> np.ndarray:
    p = Path(ref) if ref.endswith(".csv") else RUNS / ref / "oof.csv"
    fn = table()[0]
    return pd.read_csv(p).set_index("filename").loc[fn, "oof"].values


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("new"); ap.add_argument("old")
    ap.add_argument("--draws", type=int, default=10000)
    a = ap.parse_args()
    report(load_oof(a.new), load_oof(a.old), f"{a.new} vs {a.old}", a.draws)
