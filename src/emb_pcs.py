"""Principal components of the frozen speech-encoder embeddings as direct inputs to the model families (2026-10-19).

The families see each speech encoder only through its prediction (wavlm_pred, wavlm_large_pred, whisper_med_pred); with the
raw components they can combine how the voice sounds with the text features (audio x text interactions a per-encoder linear
head cannot express). Pooling as the leak-free heads (src/leakfree.py): each layer z-scored on the train clips, uniform mean
over the layers. Then PCA (k components per encoder) on all 948 clips: no labels involved, so transductive fitting is not a
leak; test clips get the same components.
shift filter (2026-10-19): test short clips are different recordings, and raw components carry that (within-batch adversarial
AUC 0.76 vs 0.64 for the handcrafted features). With shift_z_max > 0 a component is kept only if, in every recording batch
(exact 45.06 s / other short / long), its test-minus-train mean difference is within shift_z_max standard errors. Uses the
unlabelled test inputs only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ENCODERS = {"wavlm": "wavlm_layers.npz", "wavlm_large": "ssl_wavlm_large.npz", "whisper_med": "ssl_whisper_med.npz"}


def batches(cache, index) -> np.ndarray:
    G = pd.read_csv(cache / "speaker_groups.csv").set_index(["split", "filename"]).loc[index]
    return np.where(np.abs(G.duration.values - 45.06) < 0.005, "b45", G.batch.values)


def shift_z(P: pd.DataFrame, cache) -> pd.DataFrame:
    """Per component and recording batch: (test mean - train mean) / standard error of that difference."""
    b3, split = batches(cache, P.index), P.index.get_level_values(0)
    out = {}
    for b in ("b45", "short", "long"):
        tr, te = P[(b3 == b) & (split == "train")], P[(b3 == b) & (split == "test")]
        out[b] = (te.mean() - tr.mean()) / np.sqrt(tr.var() / len(tr) + te.var() / len(te))
    return pd.DataFrame(out)


def table(cache, k: int, shift_z_max: float = 0.0) -> pd.DataFrame:
    """-> (split, filename)-indexed table with columns pc_<encoder>_<i>, i = 1..k (minus shifted ones if shift_z_max > 0)."""
    out = []
    for enc, file in ENCODERS.items():
        z = np.load(cache / file)
        H = z["H"].astype(np.float64)
        tr = z["split"] == "train"
        mu, sd = H[tr].mean(0, keepdims=True), H[tr].std(0, keepdims=True) + 1e-6
        X = ((H - mu) / sd).mean(1)
        P = PCA(n_components=k, random_state=0).fit_transform(X - X.mean(0))
        idx = pd.MultiIndex.from_arrays([z["split"], z["filename"]], names=["split", "filename"])
        out.append(pd.DataFrame(P, index=idx, columns=[f"pc_{enc}_{i + 1}" for i in range(k)]))
    P = pd.concat(out, axis=1)
    if shift_z_max > 0:
        P = P.loc[:, shift_z(P, cache).abs().max(1) <= shift_z_max]
    return P
