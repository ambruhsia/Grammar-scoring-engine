"""Bias metrics for the shortcut-removal gate (PLAN_RUBRIC G9-G11). Each is named before its run.

    G9  speech_slope_short   short clips: OLS slope of the OOF residual (label - prediction) on speech time, per 10 s
    G10 ctc_fail_residual    clips with repair_wer_nofiller > CTC_CUT (fixed from the G2 table): mean OOF residual among
                             good speakers (label >= 4.0), where the loss was measured; the all-label mean is reported too
Shortcut-removal gate: accept if cv_testmix is not worse by more than 0.002 AND the named metric shrinks by at least half.

    python src/bias_metrics.py <run_id> [<run_id> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CTC_CUT = 0.6          # G2: above it, good speakers (>= 4) were under-predicted by +0.70


def _frame(run_id: str) -> pd.DataFrame:
    O = pd.read_csv(ROOT / "runs" / run_id / "oof.csv")
    I = pd.read_csv(ROOT / "cache" / "insight.csv")
    I = I[I.split == "train"][["filename", "batch", "speech_time", "repair_wer_nofiller"]]
    D = O.merge(I, on="filename")
    D["residual"] = D.label - D.oof
    return D


def speech_slope_short(run_id: str) -> float:
    d = _frame(run_id).query("batch == 'short'")
    return float(np.polyfit(d.speech_time / 10.0, d.residual, 1)[0])


def ctc_fail_residual(run_id: str) -> tuple[float, float, int]:
    d = _frame(run_id)
    f = d[d.repair_wer_nofiller > CTC_CUT]
    good = f[f.label >= 4.0]
    return float(good.residual.mean()), float(f.residual.mean()), len(good)


if __name__ == "__main__":
    for rid in sys.argv[1:]:
        g, a, n = ctc_fail_residual(rid)
        print(f"{rid}: speech_slope_short {speech_slope_short(rid):+.4f} per 10 s | ctc_fail_residual (label>=4, n={n}) {g:+.4f}"
              f"  (all labels {a:+.4f})")
