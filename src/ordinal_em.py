"""Ordinal grade distribution + EM prior adaptation (new method, approved by Amrita 2026-10-13).

Turns the blend's point prediction into a calibrated distribution over the 8 grade levels, then re-estimates the grade prior of
an UNLABELLED target set (a validation fold in CV, the test clips for the submission) separately per recording batch:
    ordinal   proportional-odds model  P(y <= l_k | p) = sigmoid(theta_k - beta * p), fitted per batch by maximum likelihood
    EM        Saerens, Latinne & Decaestecker (2002): pi <- mean_i post_i,  post_i(l) ∝ p_i(l) * pi(l) / pi_train(l)
    decision  posterior mean sum_l l * post_i(l)   (the RMSE-optimal point under the adapted distribution)
No labels of the target set are used. Levels: 1.5 (with the single 1.0 merged in), 2.0, ..., 5.0.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

LEVELS = np.arange(1.5, 5.01, 0.5)
EM_ITERS, EM_TOL = 200, 1e-8


def level_index(y) -> np.ndarray:
    return np.clip(np.round((np.asarray(y, float) - 1.5) / 0.5).astype(int), 0, len(LEVELS) - 1)


def _probs(params, p):
    k = len(LEVELS) - 1
    theta = params[0] + np.concatenate([[0.0], np.cumsum(np.exp(params[1:k]))])
    beta = params[k]
    cum = expit(theta[None, :] - beta * np.asarray(p, float)[:, None])          # P(y <= l_j), j < K
    cum = np.hstack([np.zeros((len(p), 1)), cum, np.ones((len(p), 1))])
    return np.clip(np.diff(cum, axis=1), 1e-9, 1.0)


def fit(p, y):
    """MLE of the proportional-odds model on predictions p and labels y."""
    idx = level_index(y)
    k = len(LEVELS) - 1
    x0 = np.r_[1.5 * 2.0, np.full(k - 1, np.log(1.0)), 2.0]
    nll = lambda w: -np.log(_probs(w, p)[np.arange(len(idx)), idx]).sum() + 1e-3 * (w ** 2).sum()
    return minimize(nll, x0, method="L-BFGS-B").x


def em(post, pi_train):
    """Saerens EM: re-estimate the target prior from posteriors computed under pi_train."""
    pi = pi_train.copy()
    for _ in range(EM_ITERS):
        q = post * (pi / pi_train)[None, :]
        q /= q.sum(1, keepdims=True)
        new = q.mean(0)
        if np.abs(new - pi).max() < EM_TOL:
            pi = new
            break
        pi = new
    q = post * (pi / pi_train)[None, :]
    return q / q.sum(1, keepdims=True), pi


def adapt(p_src, y_src, p_tgt, use_em=True):
    """Fit on (p_src, y_src), return (posterior-mean predictions for p_tgt, adapted prior, train prior)."""
    w = fit(p_src, y_src)
    pi_train = np.bincount(level_index(y_src), minlength=len(LEVELS)) / len(y_src)
    pi_train = np.clip(pi_train, 1e-4, None); pi_train /= pi_train.sum()
    post = _probs(w, p_tgt)
    if use_em:
        post, pi = em(post, pi_train)
    else:
        pi = pi_train
    return post @ LEVELS, pi, pi_train
