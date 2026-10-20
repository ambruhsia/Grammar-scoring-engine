"""Speaker groups and recording batch -> cache/speaker_groups.csv (PLAN_RUBRIC A1 + A2).

Speaker groups (train only). Test speakers are all new, so each test clip's nearest *train* clip is a
different speaker by construction. The 99th percentile of those test->train nearest cosine similarities
is the threshold t for "same voice". Train clips are grouped by average-linkage agglomerative clustering
at cosine distance 1 - t (average linkage, not connected components, so chains don't merge into giant
groups). Source: the WavLM x-vectors in cache/spk_xvec.npz.

Recording batch (train and test). Durations have two modes (~45 s and ~60 s). The cut is the minimum of
a smoothed density between the modes; batch = short below it, long above.

    python src/speaker_groups.py      # writes cache/speaker_groups.csv and reports/phaseA/*.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.cluster import AgglomerativeClustering

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
REPORT = ROOT / "reports" / "phaseA"
TWIN_PCT = 99            # percentile of test->train nearest similarity used as the same-voice threshold
MODES = (45.0, 60.0)     # the two recording-batch duration modes the cut lies between


def train_test(cache: Path = CACHE):
    F = pd.read_csv(cache / "features.csv")
    tr = pd.read_csv(cache / "train.csv")
    T = F[F.split == "train"].merge(tr[tr.label > 0], on="filename").reset_index(drop=True)
    TE = F[F.split == "test"].reset_index(drop=True)
    return T, TE


def xvectors(cache: Path, T: pd.DataFrame, TE: pd.DataFrame):
    z = np.load(cache / "spk_xvec.npz")
    idx = {k: i for i, k in enumerate(zip(z["split"], z["filename"]))}
    E = z["E"].astype(np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    return (E[[idx[("train", f)] for f in T.filename]], E[[idx[("test", f)] for f in TE.filename]])


def speaker_groups(A: np.ndarray, B: np.ndarray):
    """A: train x-vectors, B: test x-vectors (L2-normalised). -> (groups, t, stats dict, test->train nn sims)."""
    nn_test = (B @ A.T).max(1)
    t = float(np.percentile(nn_test, TWIN_PCT))
    groups = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=1.0 - t).fit_predict(A)
    sizes = pd.Series(groups).value_counts()
    S = A @ A.T
    np.fill_diagonal(S, -1)
    stats = {
        "threshold_t": t,
        "n_groups": int(sizes.size),
        "share_train_with_twin": float((pd.Series(groups).map(sizes) >= 2).mean()),
        "largest_group": int(sizes.max()),
        "largest_group_share": float(sizes.max() / len(A)),
        "groups_size_ge2": int((sizes >= 2).sum()),
        "test_above_t": int((nn_test > t).sum()),
        "train_nn_above_t": float((S.max(1) > t).mean()),
    }
    return groups, t, stats, nn_test, S.max(1)


def batch_cut(d_all: np.ndarray) -> float:
    kde = gaussian_kde(d_all, bw_method=0.05)
    xs = np.linspace(MODES[0] + 1, MODES[1] - 1, 1401)
    return float(xs[np.argmin(kde(xs))])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, TE = train_test()
    A, B = xvectors(CACHE, T, TE)
    groups, t, stats, nn_test, nn_train = speaker_groups(A, B)

    cut = batch_cut(np.r_[T.duration.values, TE.duration.values])
    bt = np.where(T.duration < cut, "short", "long")
    be = np.where(TE.duration < cut, "short", "long")

    out = pd.concat([
        pd.DataFrame({"split": "train", "filename": T.filename, "group": groups, "batch": bt,
                      "duration": T.duration}),
        pd.DataFrame({"split": "test", "filename": TE.filename, "group": -1, "batch": be,
                      "duration": TE.duration}),
    ], ignore_index=True)
    out.to_csv(CACHE / "speaker_groups.csv", index=False)

    REPORT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bins = np.arange(0, 63, 1.0)
    ax.hist(T.duration, bins=bins, alpha=0.6, density=True, label=f"train (n={len(T)})")
    ax.hist(TE.duration, bins=bins, alpha=0.6, density=True, label=f"test (n={len(TE)})")
    ax.axvline(cut, color="k", ls="--", lw=1, label=f"batch cut {cut:.1f} s")
    ax.set_xlabel("duration (s)"); ax.set_ylabel("density"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(REPORT / "duration_batches.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    b2 = np.linspace(min(nn_test.min(), nn_train.min()), 1, 60)
    ax.hist(nn_train, bins=b2, alpha=0.6, density=True, label="train -> nearest other train")
    ax.hist(nn_test, bins=b2, alpha=0.6, density=True, label="test -> nearest train (new speakers)")
    ax.axvline(t, color="k", ls="--", lw=1, label=f"t = p{TWIN_PCT} = {t:.3f}")
    ax.set_xlabel("cosine similarity of speaker x-vectors"); ax.set_ylabel("density"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(REPORT / "speaker_twins.png", dpi=130); plt.close(fig)

    print("speaker groups:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()})
    sizes = pd.Series(groups).value_counts()
    print("group size distribution:", sizes.value_counts().sort_index().to_dict())
    print(f"batch cut {cut:.2f} s")
    for sp, b in (("train", bt), ("test", be)):
        print(f"  {sp}: short {np.mean(b == 'short'):.1%} ({np.sum(b == 'short')})  long {np.mean(b == 'long'):.1%}")
    print("wrote", CACHE / "speaker_groups.csv")


if __name__ == "__main__":
    main()
