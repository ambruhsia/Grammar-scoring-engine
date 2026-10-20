"""Same-speaker pooling (Phase I; the idea is from nakulgrover118's and Anurag Nayak's public write-ups).

Speakers recur: 53% of training clips have a same-voice twin in train, and 41 of the 216 test clips sit in 17 same-voice groups
inside test (sizes 2-3, adjacent file IDs, like the training twins). Twins' grades differ by 0.24 on average (1.16 for random
pairs) while the model's errors on them are only weakly shared (residual correlation 0.24), so pulling each prediction toward its
speaker group's mean prediction cancels part of the per-clip noise. No labels are used: OOF predictions are pooled within
training speaker groups (grouped folds keep a group inside one validation fold, so the group mean comes from one fold model,
as on test), test predictions within test speaker groups.

    p' = (1 - s) * p + s * mean of p over the clip's speaker group      (s = POOL_S in experiment.py; singletons unchanged)

Test groups: average-linkage clustering of the test x-vectors at the same cosine threshold t as the training groups
(src/speaker_groups.py: the 99th percentile of test->train nearest similarity).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def test_groups(cache: Path, train_files, test_files) -> np.ndarray:
    z = np.load(cache / "spk_xvec.npz")
    idx = {k: i for i, k in enumerate(zip(z["split"], z["filename"]))}
    E = z["E"].astype(np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    A = E[[idx[("train", f)] for f in train_files]]
    B = E[[idx[("test", f)] for f in test_files]]
    t = float(np.percentile((B @ A.T).max(1), 99))
    return AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                   distance_threshold=1.0 - t).fit_predict(B)


def pool(p: np.ndarray, groups: np.ndarray, s: float) -> np.ndarray:
    gm = pd.Series(p).groupby(np.asarray(groups)).transform("mean").values
    return (1 - s) * p + s * gm
