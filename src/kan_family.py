"""Kolmogorov-Arnold network (pykan, MIT) as a scikit-learn regressor: the 10th model family (2026-10-20).

A KAN puts a learned univariate function (B-spline plus a SiLU base) on every input edge and sums them. Width [n_inputs, 1] is
an additive model with one smooth learned curve per input: a different inductive bias from the other families (ridge: straight
lines; trees: steps; kNN / SVR: distances). Fixed before touching the competition data, from a synthetic smoke test (44 inputs,
585 rows): the additive form beat a hidden layer of 2, and validation error was lowest after ~3 LBFGS steps and rose after,
so the number of steps (<= MAX_STEPS) is chosen on an inner validation split of the training fold (VAL_FRAC), then the net is
refit on the whole training fold for that many steps. Inputs standardised and clipped to [-3, 3]; target standardised per fit.
Sample weights are not supported (TypeError, so common.fit_weighted falls back to an unweighted fit, as for kNN).
"""
from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.base import BaseEstimator, RegressorMixin  # noqa: E402

GRID, K, LAMB, MAX_STEPS, VAL_FRAC = 5, 3, 1e-2, 20, 0.15


class KANRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, seed: int = 0, grid: int = GRID, k: int = K, lamb: float = LAMB, max_steps: int = MAX_STEPS,
                 val_frac: float = VAL_FRAC):
        self.seed, self.grid, self.k, self.lamb, self.max_steps, self.val_frac = seed, grid, k, lamb, max_steps, val_frac

    def _scale(self, X):
        return np.clip((np.asarray(X, float) - self.mu_) / self.sd_, -3, 3)

    def _train(self, Xa, ya, Xb, yb, steps):
        from kan import KAN
        net = KAN(width=[Xa.shape[1], 1], grid=self.grid, k=self.k, seed=self.seed, auto_save=False,
                  symbolic_enabled=False, device="cpu")
        T = lambda a: torch.tensor(a, dtype=torch.float32)
        res = net.fit({"train_input": T(Xa), "train_label": T(ya[:, None]), "test_input": T(Xb), "test_label": T(yb[:, None])},
                      opt="LBFGS", steps=steps, lamb=self.lamb, log=10 ** 6)
        return net, np.asarray(res["test_loss"], float)

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise TypeError("KANRegressor does not take sample weights")
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-9
        self.ym_, self.ys_ = y.mean(), y.std() + 1e-9
        Z, t = self._scale(X), (y - self.ym_) / self.ys_
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(y))
        nv = max(1, int(round(self.val_frac * len(y))))
        va, tr = idx[:nv], idx[nv:]
        _, val_curve = self._train(Z[tr], t[tr], Z[va], t[va], self.max_steps)
        self.steps_ = int(np.nanargmin(val_curve)) + 1
        self.net_, _ = self._train(Z, t, Z[va], t[va], self.steps_)
        return self

    def predict(self, X):
        with torch.no_grad():
            out = self.net_(torch.tensor(self._scale(X), dtype=torch.float32)).numpy()[:, 0]
        return out * self.ys_ + self.ym_
