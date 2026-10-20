"""Selection table for the rubric layer (PLAN_RUBRIC Phase B). -> reports/phaseB/selection.csv, boxplots.png

A feature joins the rubric layer only if it passes all four checks (on the 732 labelled train clips):
  1 sign       Spearman sign with the label matches the expected sign ("data decides" features: n/a)
  2 monotone   mean by label level 2.0 ... 5.0 moves the expected way, with at most one inversion
  3 length     |Spearman with duration| < 0.2 inside each recording batch (short and long)
  4 partial    |partial Spearman with the label, controlling for repair_wer_nofiller| >= 0.05

    python src/rubric_select.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

import rubric_feats as R

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
OUT = ROOT / "reports" / "phaseB"
EXISTING = {"cola_frac_bad": -1, "gpt2_nll_gap": -1, "repair_wer_nofiller": -1, "no_punct": -1}
# Phase C3 (Qwen2.5-7B-Instruct), joined when cache/qwen/ exists. The judge is for interpretation.
QWEN = {"llm_err_per100": -1, "corrected_error_ratio": +1, "llm_rubric_expected": +1}
LEVELS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
# Excluded by rule regardless of the checks (reported as a finding, never in the model).
EXCLUDED = {"self_repair_per100": "CTC restart rate: measures disfluent restarts, not self-correction"}
DUR_MAX, PARTIAL_MIN, MAX_INVERSIONS = 0.2, 0.05, 1


def partial_spearman(x, y, z) -> float:
    ok = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    rx, ry, rz = rankdata(x[ok]), rankdata(y[ok]), rankdata(z[ok])
    Z = np.c_[np.ones(ok.sum()), rz]
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def sp(x, y) -> float:
    ok = ~(np.isnan(x) | np.isnan(y))
    return float(spearmanr(x[ok], y[ok]).statistic)


def table(T: pd.DataFrame) -> pd.DataFrame:
    y, rw, dur = T.label.values, T.repair_wer_nofiller.values, T.duration.values
    short = (T.batch == "short").values
    rows = []
    extra = {f: e for f, e in QWEN.items() if f in T.columns}
    for f, exp in {**R.EXPECTED, **EXISTING, **extra}.items():
        x = T[f].values.astype(float)
        rho = sp(x, y)
        direction = exp if exp else np.sign(rho)
        means = [np.nanmean(x[y == lv]) for lv in LEVELS]
        inversions = int(sum(np.sign(b - a) == -direction for a, b in zip(means, means[1:])))
        d_short, d_long = sp(x[short], dur[short]), sp(x[~short], dur[~short])
        part = np.nan if f == "repair_wer_nofiller" else partial_spearman(x, y.astype(float), rw)
        c1 = True if exp == 0 else bool(np.sign(rho) == exp)
        c2 = inversions <= MAX_INVERSIONS
        c3 = max(abs(d_short), abs(d_long)) < DUR_MAX
        c4 = True if f == "repair_wer_nofiller" else abs(part) >= PARTIAL_MIN
        rows.append({"feature": f, "source": "new" if f in R.EXPECTED else ("qwen" if f in QWEN else "existing"),
                     "expected": {1: "+", -1: "-", 0: "data"}[exp], "spearman": rho,
                     "1_sign": c1, "inversions": inversions, "2_monotone": c2,
                     "dur_rho_short": d_short, "dur_rho_long": d_long, "3_length": c3,
                     "partial_vs_repair": part, "4_partial": c4,
                     "passes": c1 and c2 and c3 and c4 and f not in EXCLUDED, "note": EXCLUDED.get(f, ""),
                     "means_by_level": " ".join(f"{m:.3g}" for m in means)})
    return pd.DataFrame(rows)


def boxplots(T: pd.DataFrame, feats: list[str], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = 4
    rows = int(np.ceil(len(feats) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.5 * rows))
    for ax, f in zip(axes.ravel(), feats):
        data = [T.loc[T.label == lv, f].dropna().values for lv in LEVELS]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(LEVELS) + 1), [f"{lv:g}" for lv in LEVELS], fontsize=7)
        ax.set_title(f, fontsize=9)
    for ax in axes.ravel()[len(feats):]:
        ax.axis("off")
    fig.supxlabel("grammar score")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    F = pd.read_csv(CACHE / "features.csv")
    tr = pd.read_csv(CACHE / "train.csv")
    G = pd.read_csv(CACHE / "speaker_groups.csv")
    T = (F[F.split == "train"].merge(tr[tr.label > 0], on="filename")
         .merge(R.features().query("split == 'train'").drop(columns="split"), on="filename")
         .merge(G[G.split == "train"][["filename", "batch"]], on="filename"))
    if (CACHE / "qwen" / "qwen_judge.csv").exists():
        import qwen_feats
        Q = qwen_feats.features(CACHE, set(T.filename)).reset_index()
        T = T.merge(Q[Q.split == "train"].drop(columns="split"), on="filename")
    df = table(T)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "selection.csv", index=False)
    boxplots(T, list(R.EXPECTED) + list(EXISTING) + [f for f in QWEN if f in T.columns], OUT / "boxplots.png")
    print("qwen features passing 1-4:", df[(df.source == "qwen") & df.passes].feature.tolist())
    show = df.drop(columns="means_by_level").copy()
    with pd.option_context("display.width", 220):
        print(show.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print("\nrubric layer (new features passing 1-4):", df[(df.source == "new") & df.passes].feature.tolist())


if __name__ == "__main__":
    main()
