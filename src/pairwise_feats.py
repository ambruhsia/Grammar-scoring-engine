"""Pairwise anchored examiner (Kaggle job shl-qwen35-pairwise) -> a fold-aware Bradley-Terry grade, and its pre-registered test.

Input cache/qwen35/qwen35_pairwise.csv: one row per (clip, anchor, order) with p_a = P("A") / (P("A") + P("B")) of the next token.
P(clip better than anchor) = mean of p_a (clip shown as A) and 1 - p_a (clip shown as B), which cancels position bias.
pw_theta: the clip's grade on the anchors' scale, MAP of a Bradley-Terry model with a fixed slope,
    P(clip better than anchor k) = sigmoid(SLOPE * (theta - grade_k)),  prior theta ~ N(PRIOR_MU, PRIOR_SD^2),
fitted on a grid over the anchors that are *allowed*: an anchor's grade is a training label, so inside a fold only anchors among
that fold's training clips count (topic check: training clips of other topics; test: all anchors). An anchor is never compared
with itself, so its own grade never enters its own feature.
The feature joins the 21 judge numbers in the combination's examiner-correction ridge (nested in folds); the hybrid is rebuilt
over the 4 fold-seed sets of hybrid11 and judged by the rule pre-registered in LOG.md (paired speaker-group bootstrap interval
below 0 on cv_testmix, and the post-correction topic check not worse by more than 0.001).
usage: python src/pairwise_feats.py [comparisons csv]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "kaggle" / "final")]
import replay as R  # noqa: E402  (the final notebook's stacking code)
from paired_bootstrap import paired  # noqa: E402

SLOPE, PRIOR_MU, PRIOR_SD = 3.0, 3.5, 1.5
GRID = np.arange(0.5, 6.5001, 0.01)
BASE = ("1015-070513", "1015-114531", "1015-165432", "1015-172136")
COMBO = ("1016-100139", "1016-140135", "1016-140601", "1016-141007")
POOL_S = 0.3


class Pairwise:
    """P(clip better than anchor) for every train / test clip against its batch's anchors -> pw_theta for any allowed anchor set."""

    def __init__(self, fn, tfn, path=ROOT / "cache" / "qwen35" / "qwen35_pairwise.csv"):
        P = pd.read_csv(path)
        P["pc"] = np.where(P.order == 0, P.p_a, 1 - P.p_a)
        self.raw = P
        Q = P.groupby(["split", "filename", "anchor"]).pc.mean().unstack("anchor")
        self.anchors = list(Q.columns)
        g = P.drop_duplicates("anchor").set_index("anchor").anchor_grade
        self.grade = g.loc[self.anchors].values.astype(float)
        self.M = Q.reindex([("train", f) for f in fn]).values            # train clips x anchors (NaN = not compared)
        self.Mt = Q.reindex([("test", f) for f in tfn]).values
        assert not np.isnan(self.Mt).all(1).any() and not np.isnan(self.M).all(1).any(), "a clip has no comparisons"
        pos = {f: i for i, f in enumerate(fn)}
        self.anchor_row = np.array([pos[a] for a in self.anchors])        # each anchor's row among the train clips
        z = SLOPE * (GRID[None, :] - self.grade[:, None])                # anchors x grid
        self.LS, self.L1S = -np.logaddexp(0, -z), -np.logaddexp(0, z)    # log sigmoid(z), log(1 - sigmoid(z))
        self.prior = -0.5 * ((GRID - PRIOR_MU) / PRIOR_SD) ** 2

    def theta(self, M, allowed):
        W = ~np.isnan(M) & allowed[None, :]
        Pm = np.where(W, M, 0.0)
        L = Pm @ self.LS + (W - Pm) @ self.L1S + self.prior
        return GRID[L.argmax(1)]

    def allowed(self, train_rows):
        return np.isin(self.anchor_row, train_rows)

    def train(self, train_rows):
        return self.theta(self.M, self.allowed(train_rows))

    def test(self):
        return self.theta(self.Mt, np.ones(len(self.anchors), bool))


def replay_combo(c, folds, XJ, groups, extra=None):
    """The combination's stacking (as replay.replay with shrink, correction, pooling), with an optional fold-dependent extra
    correction feature: extra(train_rows) -> (n_train,) values for all train clips. -> out-of-fold prediction (seed mean)."""
    CO_s, TO, y, w, bt = c["oof"], c["topic"], c["y"], c["w"], c["batch"]
    X = XJ[0]
    out = []
    for i, s in enumerate(int(v) for v in c["seeds"]):
        o = np.zeros(len(y))
        for a, b in folds[s]:
            o[b] = R.apply_blend(CO_s[i][b], bt[b], R.blend_weights(TO[a], y[a], w[a], bt[a], True))
        o = np.clip(o, *R.CLIP)
        oc = o.copy()
        for a, b in folds[s]:
            Xf = X if extra is None else np.c_[X, extra(a)]
            f = R.ridge_correction(np.c_[Xf[a], o[a]], y[a] - o[a], w[a])
            oc[b] = o[b] + f(np.c_[Xf[b], o[b]])
        out.append(np.clip(R.pool(np.clip(oc, *R.CLIP), groups, POOL_S), *R.CLIP))
    return np.clip(np.mean(out, 0), *R.CLIP)


def main(path=ROOT / "cache" / "qwen35" / "qwen35_pairwise.csv"):
    C = {r: dict(np.load(ROOT / "runs" / r / "cands.npz", allow_pickle=True)) for r in BASE + COMBO}
    c0 = C[BASE[0]]
    fn, tfn, y, w, bt = c0["filename"], c0["test_filename"], c0["y"], c0["w"], c0["batch"]
    FD = pd.read_csv(ROOT / "kaggle" / "final" / "folds_all.csv")
    folds = {}
    for s in FD.seed.unique():
        f = FD[FD.seed == s].set_index("filename").loc[fn, "fold"].values
        folds[int(s)] = [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(5)]
    G = pd.read_csv(ROOT / "cache" / "speaker_groups.csv")
    gtr = G[G.split == "train"].set_index("filename")
    groups = gtr.loc[fn, "group"].values
    b3 = np.where(np.abs(gtr.loc[fn, "duration"].values - 45.06) < 0.005, "b45", bt)
    J1 = pd.read_csv(ROOT / "cache" / "qwen35" / "qwen35_judge.csv").set_index(["split", "filename"])
    J2 = pd.read_csv(ROOT / "cache" / "qwen35" / "qwen35_judge2.csv").set_index(["split", "filename"])
    XJ = (R.judge_features(J1, J2, [("train", f) for f in fn]), R.judge_features(J1, J2, [("test", f) for f in tfn]))
    PW = Pairwise(fn, tfn, path)
    oof = lambda r: pd.read_csv(ROOT / "runs" / r / "oof.csv").set_index("filename").loc[fn, "oof"].values
    short = bt == "short"
    hybrid = lambda bb, cb: np.clip(np.where(short, (bb + cb) / 2, cb), *R.CLIP)
    wr = lambda p: float(np.sqrt(np.sum(w * (y - p) ** 2) / np.sum(w)))

    # ---- diagnostics (not part of the decision)
    P = PW.raw
    print(f"comparisons {len(P)}, clips {P[['split', 'filename']].drop_duplicates().shape[0]}, anchors {len(PW.anchors)}; "
          f"top token A/B {P.top.mean():.1%}; mean P(A) clip-first {P[P.order == 0].p_a.mean():.3f}, anchor-first "
          f"{P[P.order == 1].p_a.mean():.3f}; order agreement corr {np.corrcoef(P[P.order == 0].sort_values(['split', 'filename', 'anchor']).p_a, 1 - P[P.order == 1].sort_values(['split', 'filename', 'anchor']).p_a)[0, 1]:.3f}")
    th_all = PW.train(np.arange(len(y)))
    b_base, b_combo = np.mean([oof(r) for r in BASE], 0), np.mean([oof(r) for r in COMBO], 0)
    h11 = hybrid(b_base, b_combo)
    res = y - h11
    jv = XJ[0][:, 0]                                                       # judge35_verbatim_holistic_ev, for reference
    part = lambda v: np.corrcoef(v - np.polyval(np.polyfit(h11, v, 1), h11), res)[0, 1]
    for name, m in (("all", np.ones(len(y), bool)), ("b45", b3 == "b45"), ("short", b3 == "short"), ("long", b3 == "long")):
        print(f"  {name:5s} n={m.sum():3d}  corr(pw_theta, grade) {np.corrcoef(th_all[m], y[m])[0, 1]:+.3f}   "
              f"(round-1 verbatim judge {np.corrcoef(jv[m], y[m])[0, 1]:+.3f})")
    print(f"  partial corr with hybrid11's out-of-fold residual: pw_theta {part(th_all):+.3f}, round-1 verbatim judge "
          f"{part(jv):+.3f}; pw_theta test mean {PW.test().mean():.3f} vs train {th_all.mean():.3f}")

    # ---- the replay without the extra feature reproduces the runs
    for r in COMBO:
        d = np.abs(replay_combo(C[r], folds, XJ, groups) - oof(r)).max()
        assert d < 1e-9, f"{r}: replay differs from oof.csv by {d:.2e}"
    print("replay of the 4 combination runs = their oof.csv (max |diff| < 1e-9)")

    # ---- the pre-registered test
    new_combo = np.mean([replay_combo(C[r], folds, XJ, groups, extra=PW.train) for r in COMBO], 0)
    h_new = hybrid(b_base, new_combo)
    print(f"\nhybrid11 cv_testmix {wr(h11):.4f} -> with pw_theta {wr(h_new):.4f}   (combo bag {wr(b_combo):.4f} -> {wr(new_combo):.4f})")
    rb = {}
    for name, m in (("testmix", None), ("short", short), ("long", ~short)):
        rb[name] = r_ = paired(h_new, h11, y, w if m is None else np.ones(len(y)), groups, mask=m)
        print(f"  bootstrap {name:8s} new-old {r_['diff']:+.4f}  95% [{r_['lo']:+.4f}, {r_['hi']:+.4f}]  P(new better) {r_['p_better']:.2f}")

    # topic check after the correction, seed-42 combination run (as in the final notebook)
    I = pd.read_csv(ROOT / "cache" / "insight.csv").set_index(["split", "filename"])
    tp = I.loc[[("train", f) for f in fn], "topic"].values
    tfolds = [(np.where((tp != t) & ~np.isin(groups, np.unique(groups[tp == t])))[0], np.where(tp == t)[0]) for t in sorted(set(tp))]
    c42 = C[COMBO[0]]
    nt = np.zeros(len(y))
    for _, b in tfolds:
        rows = np.setdiff1d(np.arange(len(y)), b)
        nt[b] = R.apply_blend(c42["topic"][b], bt[b], R.blend_weights(c42["topic"][rows], y[rows], w[rows], bt[rows], True))
    nt = np.clip(nt, *R.CLIP)

    def corrected(extra):
        out = nt.copy()
        for a, b in tfolds:
            Xf = XJ[0] if extra is None else np.c_[XJ[0], extra(a)]
            f = R.ridge_correction(np.c_[Xf[a], nt[a]], y[a] - nt[a], w[a])
            out[b] = nt[b] + f(np.c_[Xf[b], nt[b]])
        return np.clip(out, *R.CLIP)
    t_old, t_new = wr(corrected(None)), wr(corrected(PW.train))
    print(f"  topic check after the correction (seed-42 combination run): {t_old:.4f} -> {t_new:.4f} ({t_new - t_old:+.4f}); "
          f"before the correction {wr(nt):.4f}")
    ok = rb["testmix"]["hi"] < 0 and t_new - t_old <= 0.001
    print(f"\nDECISION (pre-registered): {'ADOPT' if ok else 'NOT ADOPTED'}  — interval upper {rb['testmix']['hi']:+.4f} "
          f"(needs < 0), topic check {t_new - t_old:+.4f} (needs <= +0.001)")
    return dict(h11=h11, h_new=h_new, new_combo=new_combo, PW=PW, ok=ok)


if __name__ == "__main__":
    main(*sys.argv[1:2])
