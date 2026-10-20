"""Loads the cached modelling table the way the notebook's §9 builds it, with speaker-grouped folds.

Folds (PLAN_RUBRIC A3): StratifiedGroupKFold(5, shuffle=True, random_state=seed), stratified on the label
(1.0 merged into 1.5), grouped on speaker (cache/speaker_groups.csv, from src/speaker_groups.py), so a
speaker's twin clips never sit on both sides of a split. folds="random" keeps the old StratifiedKFold,
for comparison only. Clip weights w make each recording batch count in its test proportion.
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from common import CACHE, CLIP, FULL_SEEDS, fit_weighted

EXCLUDE = {"split", "filename", "label", "asr_avg_logprob"}
# label 1.5 (with the single 1.0 merged in) has 4 rows — fewer than 5 folds. Expected, as in the notebook.
warnings.filterwarnings("ignore", message="The least populated class")


def batch_weights(batch_train, batch_test) -> np.ndarray:
    """Per train clip p_test(batch) / p_train(batch), normalised to mean 1."""
    p_tr = pd.Series(batch_train).value_counts(normalize=True)
    p_te = pd.Series(batch_test).value_counts(normalize=True)
    w = np.array([p_te.get(b, 0.0) / p_tr[b] for b in batch_train])
    return w / w.mean()


def grouped_folds(T, strat, groups, seed):
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(T, strat, groups))


def load(seeds=(42, 1337, 2024), folds="grouped") -> SimpleNamespace:
    missing = [f for f in ("features.csv", "train.csv", "test.csv", "speaker_groups.csv")
               if not (CACHE / f).exists()]
    if missing:
        raise SystemExit(f"cache/ is missing {missing}. Run: python scripts/pull_cache.py, then "
                         f"python src/speaker_groups.py")

    F = pd.read_csv(CACHE / "features.csv")
    train_all = pd.read_csv(CACHE / "train.csv")
    test = pd.read_csv(CACHE / "test.csv")
    train = train_all[train_all.label > 0].reset_index(drop=True)   # 37 static clips are a 0 sentinel

    FEATS = [c for c in F.columns if c not in EXCLUDE]
    T = F[F.split == "train"].merge(train, on="filename").reset_index(drop=True)
    TE = F[F.split == "test"].reset_index(drop=True)
    y = T.label.values.astype(float)
    med = T[FEATS].median()
    strat = T.label.replace({1.0: 1.5}).astype(str)
    G = pd.read_csv(CACHE / "speaker_groups.csv")
    gtr = G[G.split == "train"].set_index("filename")
    gte = G[G.split == "test"].set_index("filename")
    groups = gtr.loc[T.filename, "group"].values
    batch, batch_test = gtr.loc[T.filename, "batch"].values, gte.loc[TE.filename, "batch"].values
    w = batch_weights(batch, batch_test)
    if folds == "grouped":
        fold_sets = {s: grouped_folds(T, strat, groups, s) for s in seeds}
        rows = [(f, s, k) for s in FULL_SEEDS for k, (_, va) in enumerate(grouped_folds(T, strat, groups, s))
                for f in T.filename.values[va]]
        pd.DataFrame(rows, columns=["filename", "seed", "fold"]).to_csv(CACHE / "folds_grouped.csv", index=False)
    elif folds == "random":
        fold_sets = {s: list(StratifiedKFold(5, shuffle=True, random_state=s).split(T, strat)) for s in seeds}
    else:
        raise ValueError(f"folds must be 'grouped' or 'random', not {folds!r}")

    def oof_test(model, cols, fold_list, y_fit=None, sample_weight=None):
        """-> (oof, test_pred), both clipped. Test prediction is the fold average. y_fit: training target
        (default y), e.g. denoised labels; evaluation is always on y."""
        yf = y if y_fit is None else np.asarray(y_fit, float)
        X, Xt = T[cols].fillna(med[cols]).values, TE[cols].fillna(med[cols]).values
        oof, te = np.zeros(len(y)), np.zeros(len(Xt))
        for tr_i, va_i in fold_list:
            sw = None if sample_weight is None else np.asarray(sample_weight, float)[tr_i]
            m = fit_weighted(clone(model), X[tr_i], yf[tr_i], sw)
            oof[va_i] = m.predict(X[va_i])
            te += m.predict(Xt) / len(fold_list)
        return np.clip(oof, *CLIP), np.clip(te, *CLIP)

    return SimpleNamespace(F=F, T=T, TE=TE, y=y, train=train, test=test, FEATS=FEATS, med=med,
                           seeds=tuple(seeds), folds=fold_sets, fold_scheme=folds, oof_test=oof_test,
                           groups=groups, batch=batch, batch_test=batch_test, w=w, cache=CACHE)
