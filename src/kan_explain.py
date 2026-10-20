"""An interpretable grammar model: an additive KAN on 8 understandable signals, turned into a formula (2026-10-20).

Inputs (fixed in advance; all label-free):
    examiner    Qwen3.5-9B rubric grade on the verbatim transcript (expected value of its 1-5 answer)
    repair      repair_wer_nofiller: share of words the CTC recogniser and Whisper disagree on (errors Whisper smoothed over)
    lm_gap      gpt2_nll_gap: language-model surprise of the raw transcript minus that of the cleaned one
    artic_rate  articulation_rate: syllables per second while speaking
    sent_len    mean_sent_len: words per sentence
    n_words     words spoken
    ttr         text_ttr: vocabulary variety (type-token ratio)
    repetition  repetition_rate
Model: KAN width [8, 1] (pykan, MIT): one learned curve per input, summed. Inputs standardised (z-scores, clipped to [-3, 3]);
training length chosen on an inner 15% split, then refit (as src/kan_family.py). Symbolic step: pykan's auto_symbolic replaces
each curve by the best-fitting a*f(b*z + c) + d from a small library, then the affine parameters are refit.
Out-of-fold on the runs' speaker-grouped folds (seeds 42/1337/2024): spline KAN, its formula, a ridge on the same 8 inputs, and
the full stack for reference. Then one fit on all 732 clips gives the formula and the curves (reports/kan/).
usage: python src/kan_explain.py
"""
from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from paired_bootstrap import paired, table  # noqa: E402

LIB = ["x", "x^2", "x^3", "tanh", "sin", "exp", "abs"]
NAMES = {"examiner": None, "repair": "repair_wer_nofiller", "lm_gap": "gpt2_nll_gap", "artic_rate": "articulation_rate",
         "sent_len": "mean_sent_len", "n_words": "n_words", "ttr": "text_ttr", "repetition": "repetition_rate"}
SEEDS = (42, 1337, 2024)
OUT = ROOT / "reports" / "kan"


def inputs(fn) -> pd.DataFrame:
    F = pd.read_csv(ROOT / "cache" / "features.csv").set_index(["split", "filename"])
    J = pd.read_csv(ROOT / "cache" / "qwen35" / "qwen35_judge.csv").set_index(["split", "filename"])
    keys = [("train", f) for f in fn]
    P = J.loc[keys, [f"verbatim_holistic_p{i}" for i in range(1, 6)]].values
    X = pd.DataFrame({"examiner": (P / P.sum(1, keepdims=True)) @ np.arange(1, 6)}, index=fn)
    for k, c in NAMES.items():
        if c:
            X[k] = F.loc[keys, c].values
    return X.fillna(X.median())


class SymbolicKAN:
    def __init__(self, seed=0, lamb=1e-2, max_steps=20, val_frac=0.15):
        self.seed, self.lamb, self.max_steps, self.val_frac = seed, lamb, max_steps, val_frac

    def _net(self, n_in):
        from kan import KAN
        return KAN(width=[n_in, 1], grid=5, k=3, seed=self.seed, auto_save=False, device="cpu")

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-9
        self.ym_, self.ys_ = y.mean(), y.std() + 1e-9
        Z, t = self.z(X), (y - self.ym_) / self.ys_
        T = lambda a: torch.tensor(a, dtype=torch.float32)
        idx = np.random.default_rng(self.seed).permutation(len(y))
        nv = int(round(self.val_frac * len(y)))
        va, tr = idx[:nv], idx[nv:]
        ds_in = {"train_input": T(Z[tr]), "train_label": T(t[tr, None]), "test_input": T(Z[va]), "test_label": T(t[va, None])}
        res = self._net(Z.shape[1]).fit(ds_in, opt="LBFGS", steps=self.max_steps, lamb=self.lamb, log=10 ** 6)
        self.steps_ = int(np.nanargmin(res["test_loss"])) + 1
        ds = {"train_input": T(Z), "train_label": T(t[:, None]), "test_input": T(Z[va]), "test_label": T(t[va, None])}
        self.net_ = self._net(Z.shape[1])
        self.net_.fit(ds, opt="LBFGS", steps=self.steps_, lamb=self.lamb, log=10 ** 6)
        self.spline_pred_ = lambda Zn: self._forward(self.spline_copy_, Zn)
        self.spline_copy_ = self.net_.copy()
        self.net_.auto_symbolic(lib=LIB, verbose=0)
        self.net_.fit(ds, opt="LBFGS", steps=10, lamb=0.0, update_grid=False, log=10 ** 6)
        return self

    def z(self, X):
        return np.clip((np.asarray(X, float) - self.mu_) / self.sd_, -3, 3)

    def _forward(self, net, Zn):
        with torch.no_grad():
            return net(torch.tensor(Zn, dtype=torch.float32)).numpy()[:, 0] * self.ys_ + self.ym_

    def predict(self, X, symbolic=True):
        return self._forward(self.net_ if symbolic else self.spline_copy_, self.z(X))


def main():
    fn, y, w, groups, batch = table()
    X = inputs(fn)
    FD = pd.read_csv(ROOT / "kaggle" / "final" / "folds_all.csv")
    wr = lambda p: float(np.sqrt(np.sum(w * (y - np.clip(p, 1.5, 5)) ** 2) / np.sum(w)))
    oof = {k: [] for k in ("spline KAN", "formula (symbolic KAN)", "ridge, same 8 inputs")}
    for s in SEEDS:
        f = FD[FD.seed == s].set_index("filename").loc[fn, "fold"].values
        o = {k: np.zeros(len(y)) for k in oof}
        for k_ in range(5):
            a, b = np.where(f != k_)[0], np.where(f == k_)[0]
            m = SymbolicKAN(seed=s + k_).fit(X.values[a], y[a])
            o["spline KAN"][b] = m.predict(X.values[b], symbolic=False)
            o["formula (symbolic KAN)"][b] = m.predict(X.values[b], symbolic=True)
            r = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))).fit(X.values[a], y[a])
            o["ridge, same 8 inputs"][b] = r.predict(X.values[b])
        for k in oof:
            oof[k].append(np.clip(o[k], 1.5, 5))
        print(f"  seed {s}: " + ", ".join(f"{k} {wr(o[k]):.4f}" for k in oof), flush=True)
    stack = pd.read_csv(ROOT / "runs" / "1016-100139" / "oof.csv").set_index("filename").loc[fn, "oof"].values
    print("\nout-of-fold cv_testmix (seed mean of 3 fold-seed sets), speaker-grouped folds:")
    means = {k: np.mean(v, 0) for k, v in oof.items()}
    for k, p in means.items():
        r = paired(p, stack, y, w, groups)
        print(f"  {k:24s} {wr(p):.4f}   vs the full stack 0.5011: {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]")
    r = paired(means["formula (symbolic KAN)"], means["ridge, same 8 inputs"], y, w, groups)
    print(f"  formula vs linear ridge on the same inputs: {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]")

    # one fit on all clips: the formula and the curves
    m = SymbolicKAN(seed=0).fit(X.values, y)
    import sympy
    zs = sympy.symbols(" ".join(X.columns))
    f = m.net_.symbolic_formula(var=list(zs))[0][0]
    f = sympy.N(f * m.ys_ + m.ym_, 3)
    print(f"\nformula on all 732 clips (each input as a z-score; clipped to +-3): grade = {f}")
    OUT.mkdir(parents=True, exist_ok=True)
    Z = m.z(X.values)
    contrib = {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharey=True)
    for i, (name, ax) in enumerate(zip(X.columns, axes.ravel())):
        grid = np.linspace(-3, 3, 121)
        Zg = np.zeros((len(grid), Z.shape[1])); Zg[:, i] = grid
        base = np.zeros((1, Z.shape[1]))
        sp = m._forward(m.spline_copy_, Zg) - m._forward(m.spline_copy_, base)[0]
        sy = m._forward(m.net_, Zg) - m._forward(m.net_, base)[0]
        Zi = np.zeros_like(Z); Zi[:, i] = Z[:, i]
        vals = m._forward(m.net_, Zi) - m._forward(m.net_, np.zeros((1, Z.shape[1])))[0]
        contrib[name] = float(np.percentile(vals, 95) - np.percentile(vals, 5))
        ax.plot(grid, sp, label="learned curve"); ax.plot(grid, sy, "--", label="formula")
        lo, hi = np.percentile(Z[:, i], [5, 95]); ax.axvspan(lo, hi, color="0.92", zorder=0)
        ax.set_title(f"{name}  (5-95% effect {contrib[name]:.2f})"); ax.set_xlabel("z-score"); ax.axhline(0, color="0.6", lw=0.6)
    axes[0, 0].set_ylabel("effect on grade"); axes[0, 0].legend(fontsize=8)
    fig.suptitle("Additive KAN on 8 interpretable signals: each input's effect on the predicted grade (shaded: middle 90% of clips)")
    fig.tight_layout(); fig.savefig(OUT / "kan_curves.png", dpi=130)
    pd.Series(contrib).sort_values(ascending=False).to_csv(OUT / "kan_effects.csv", header=["effect_5_95"])
    (OUT / "kan_formula.txt").write_text(f"grade = {f}\n(inputs as z-scores over the 732 training clips, clipped to +-3)\n")
    print("effect of each input over its middle 90% (grade points):", {k: round(v, 2) for k, v in sorted(contrib.items(), key=lambda t: -t[1])})
    print(f"wrote {OUT / 'kan_curves.png'}, kan_effects.csv, kan_formula.txt")


if __name__ == "__main__":
    main()
