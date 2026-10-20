"""Anchors for the pairwise examiner (shl-qwen35-pairwise): 8 graded training answers per recording batch.

Batches: "b45" (the exact 45.06 s recordings), "short" (other clips < 47.7 s), "long". Each clip is compared only with the anchors
of its own batch, so the two answers are of similar length and recording conditions.
Target grades per batch: 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.0 (the top twice: compressed extremes are a weak spot).
Candidates: training clips whose speaker group has a single clip (groups never cross train/test, so no clip shares a speaker with
an anchor) and whose verbatim transcript has >= 50 words. Among them, at each grade, the clip whose out-of-fold hybrid11
prediction is closest to its label (anchors whose grade the honest model agrees with: least likely to be label noise). The two
short batches share candidates at a 0.25 penalty for the other batch (4.5 / 5.0 are rare there; b45's only 5.0 is 1.28 off).
Writes anchors.json ({batch: [[filename, grade], ...]}) and batch_map.json ({"split/filename": batch}) next to this file.
usage: python kaggle/qwen35_pairwise/make_anchors.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ("1015-070513", "1015-114531", "1015-165432", "1015-172136")
COMBO = ("1016-100139", "1016-140135", "1016-140601", "1016-141007")
TARGETS = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.0)
BORROW = 0.25       # the two short batches share candidates; one from the other short batch counts 0.25 grade of error extra

G = pd.read_csv(ROOT / "cache" / "speaker_groups.csv")
G["b3"] = np.where(np.abs(G.duration - 45.06) < 0.005, "b45", G.batch)
lab = pd.read_csv(ROOT / "cache" / "train.csv")
lab = lab[lab.label > 0].set_index("filename").label
words = {(r["split"], r["filename"]): len((r.get("text") or "").split())
         for r in map(json.loads, (ROOT / "cache" / "verbatim" / "asr_v.jsonl").read_text(encoding="utf-8").splitlines())}
oof = lambda r: pd.read_csv(ROOT / "runs" / r / "oof.csv").set_index("filename").oof
b, c = np.mean([oof(r) for r in BASE], 0), np.mean([oof(r) for r in COMBO], 0)
fn = oof(BASE[0]).index
tr = G[G.split == "train"].set_index("filename").loc[fn].copy()
tr["label"] = lab.loc[fn].values
tr["hyb"] = np.clip(np.where(tr.batch == "short", (b + c) / 2, c), 1.5, 5.0)
tr["err"] = (tr.hyb - tr.label).abs()
tr["single"] = tr.group.map(G.groupby("group").size()).eq(1)
tr["words"] = [words.get(("train", f), 0) for f in tr.index]
pool = tr[tr.single & (tr.words >= 50)]

anchors = {}
for bt, other in (("b45", "short"), ("short", "b45"), ("long", None)):
    picked = []
    for g in TARGETS:
        cand = pool[pool.b3.isin([bt, other]) & (pool.label == g) & ~pool.index.isin(picked)]
        assert not cand.empty, (bt, g)
        picked.append((cand.err + BORROW * (cand.b3 != bt)).idxmin())
    anchors[bt] = [[f, float(tr.label[f])] for f in picked]
    print(bt, [(f, tr.label[f], tr.b3[f], round(tr.err[f], 2), tr.words[f]) for f in picked])
(HERE / "anchors.json").write_text(json.dumps(anchors))
(HERE / "batch_map.json").write_text(json.dumps({f"{s}/{f}": b3 for s, f, b3 in zip(G.split, G.filename, G.b3)}))
print({k: int(v) for k, v in G.b3.value_counts().items()})
