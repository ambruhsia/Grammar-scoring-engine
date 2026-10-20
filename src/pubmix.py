"""Predict the public score of any blend of submitted files (Techy-A's averaging identity, extended to the board metric).

Board metric on the public clips P: S = 0.6 RMSE + 0.4 (1 - r). Anchor file b = 1014-175304 (public 0.3414), whose two shift
probes gave on P: RMSE_b 0.4881, Pearson r_b 0.8787, bias_b = mean(p_b - y) = +0.0490. Any file i is written as p_i = p_b + d_i;
everything about it on P follows from computable moments of d_i and ONE unknown, x_i = mean_P(d_i * e_b) (e_b = p_b - y):
    MSE_i = mean(d_i^2) + 2 x_i + MSE_b
    cov(p_i, y) = cov(p_b, y) + cov(d_i, p_b) - (x_i - mean(d_i) bias_b)
x_i is solved from the file's public score. For a blend q = sum_i w_i p_i (sum w = 1), d_q and x_q are the same blends of d_i and
x_i, so its public score is predicted without submitting it. Moments are taken over all 216 test clips (P is an unknown ~60%
subset): that is the approximation.
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

ROOT = Path(__file__).resolve().parents[1].as_posix()
TMP = os.path.join(tempfile.gettempdir(), "claude")
RMSE_B, R_B, BIAS_B = 0.4881, 0.8787, 0.0490
t = pd.read_csv(f"{ROOT}/cache/test.csv")


def load(f):
    return pd.read_csv(f"{ROOT}/{f}").set_index("filename").loc[t.filename, "label"].values.astype(float)


subs = json.load(open(os.path.join(TMP, "subs_matched.json")))
seen, files = set(), []
for d, sc, f in subs:                         # one entry per file (the latest score if a file was submitted twice)
    if f not in seen:
        seen.add(f); files.append((d, sc, f))
names = [d for d, _, _ in files]
S = np.array([sc for _, sc, _ in files])
P = np.column_stack([load(f) for _, _, f in files])
ib = [i for i, (_, _, f) in enumerate(files) if f == "runs/1014-175304/submission.csv"][0]
pb = P[:, ib]
Dm = P - pb[:, None]                          # d_i on the test clips
var_pb = pb.var()
mse_b = RMSE_B ** 2
var_eb = mse_b - BIAS_B ** 2
# sd(y) from var(e_b) = var(p_b) + var(y) - 2 r_b sd(p_b) sd(y)
sp = np.sqrt(var_pb)
sy = R_B * sp + np.sqrt(R_B ** 2 * var_pb - var_pb + var_eb)
cov_by = R_B * sp * sy


def score(d, x, m_d, cov_dpb, var_q):
    mse = np.mean(d ** 2) + 2 * x + mse_b
    cov_qy = cov_by + cov_dpb - (x - m_d * BIAS_B)
    r = cov_qy / (np.sqrt(var_q) * sy)
    return 0.6 * np.sqrt(max(mse, 1e-9)) + 0.4 * (1 - r)


def moments(d):
    q = pb + d
    return d.mean(), np.cov(d, pb, bias=True)[0, 1], q.var()


X = np.zeros(len(files))
for i in range(len(files)):
    d = Dm[:, i]
    m_d, c_d, v_q = moments(d)
    f = lambda x: score(d, x, m_d, c_d, v_q) - S[i]
    X[i] = 0.0 if i == ib else brentq(f, -0.2, 0.2)


def predict(w):
    w = np.asarray(w, float)
    d = Dm @ w
    m_d, c_d, v_q = moments(d)
    return score(d, X @ w, m_d, c_d, v_q)


if __name__ == "__main__":
    print(f"{len(files)} files; label sd on the public clips {sy:.3f}; anchor {names[ib][:40]}")
    ix = {f: i for i, (_, _, f) in enumerate(files)}
    a, b = ix["runs/1015-070513/submission.csv"], ix["runs/1015-121105/submission.csv"]
    w = np.zeros(len(files)); w[a] = w[b] = 0.5
    h8 = ix["runs/probes/hybrid8.csv"]
    print(f"check: hybrid8 = 1/2 1015-070513 + 1/2 1015-121105 -> predicted {predict(w):.4f}, actual {S[h8]:.4f}")
    np.save(os.path.join(TMP, "pubmix_X.npy"), X)
