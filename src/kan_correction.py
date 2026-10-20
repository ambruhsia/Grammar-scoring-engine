"""KAN in the examiner-correction stage (2026-10-20): drop-in replacements for replay.ridge_correction.

The correction learns how the 21 examiner numbers (and the stack's own prediction) relate to what the stack gets wrong (its
residual). The ridge assumes straight lines; the additive KAN (one learned curve per input, pykan) can follow the curved
relations seen in the interpretable model (src/kan_explain.py: the examiner grade acts with a floor and a steep middle).
Two losses:
    "mse"    test-mix-weighted squared error of the residual (what the ridge minimises)
    "board"  the competition metric of the corrected prediction: 0.6 * weighted RMSE + 0.4 * (1 - weighted Pearson r)
Training length chosen on an inner 15% split of the fold's rows (<= MAX_STEPS LBFGS steps), then refit on all of them.
Weights, the uncorrected prediction and the label travel as extra label columns, so the same loss works on any subset.
"""
from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

GRID, K, LAMB, MAX_STEPS, VAL_FRAC, SEED = 5, 3, 1e-2, 20, 0.15, 0


def _loss(kind):
    def mse(pred, t):                                   # t columns: residual, weight, uncorrected prediction, label
        w = t[:, 1]
        return torch.sum(w * (pred[:, 0] - t[:, 0]) ** 2) / torch.sum(w)

    def board(pred, t):
        w = t[:, 1] / torch.sum(t[:, 1])
        p, y = t[:, 2] + pred[:, 0], t[:, 3]
        rmse = torch.sqrt(torch.sum(w * (y - p) ** 2) + 1e-12)
        pc, yc = p - torch.sum(w * p), y - torch.sum(w * y)
        r = torch.sum(w * pc * yc) / torch.sqrt(torch.sum(w * pc ** 2) * torch.sum(w * yc ** 2) + 1e-12)
        return 0.6 * rmse + 0.4 * (1 - r)
    return {"mse": mse, "board": board}[kind]


def kan_correction(Z, e, sw, loss="mse", seed=SEED):
    """Fit the residual e from inputs Z (rows of [extra features, prediction]) -> predict function, like ridge_correction."""
    from kan import KAN
    Z, e, sw = np.asarray(Z, float), np.asarray(e, float), np.asarray(sw, float)
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    zs = lambda A: np.clip((np.asarray(A, float) - mu) / sd, -3, 3)
    pred_col = Z[:, -1]                                  # the uncorrected prediction is the last input
    y = pred_col + e
    T = lambda a: torch.tensor(a, dtype=torch.float32)
    lab = T(np.c_[e, sw, pred_col, y])
    X = T(zs(Z))
    idx = np.random.default_rng(seed).permutation(len(e))
    nv = int(round(VAL_FRAC * len(e)))
    va, tr = idx[:nv], idx[nv:]
    fn = _loss(loss)

    def train(rows, steps):
        net = KAN(width=[Z.shape[1], 1], grid=GRID, k=K, seed=seed, auto_save=False, symbolic_enabled=False, device="cpu")
        res = net.fit({"train_input": X[rows], "train_label": lab[rows], "test_input": X[va], "test_label": lab[va]},
                      opt="LBFGS", steps=steps, lamb=LAMB, loss_fn=fn, log=10 ** 6)
        return net, np.asarray(res["test_loss"], float)
    _, curve = train(tr, MAX_STEPS)
    net, _ = train(np.arange(len(e)), int(np.nanargmin(curve)) + 1)

    def predict(Zn):
        with torch.no_grad():
            return net(T(zs(Zn))).numpy()[:, 0]
    return predict


def board_metric(y, p, w) -> float:
    w = np.asarray(w, float) / np.sum(w)
    rmse = np.sqrt(np.sum(w * (y - p) ** 2))
    pc, yc = p - np.sum(w * p), y - np.sum(w * y)
    r = np.sum(w * pc * yc) / np.sqrt(np.sum(w * pc ** 2) * np.sum(w * yc ** 2))
    return float(0.6 * rmse + 0.4 * (1 - r))
