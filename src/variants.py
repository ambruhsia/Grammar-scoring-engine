"""Replay-based variants of hybrid11 (2026-10-17): the members stay as they are, the post-member stages change.

hybrid11 = short clips: mean of the base bag and the combination bag; long clips: the combination bag (4 fold-seed sets each).
Base pipeline: per-batch NNLS blend. Combination: blend with shrink, examiner correction (ridge on 21 Qwen3.5 judge numbers + the
prediction, nested in folds), same-speaker pooling 0.3. A variant changes the correction's inputs of one or both pipelines; an
extra feature may be fold-aware (a function of the fold's training rows, e.g. the pairwise examiner's anchors). Every replay of
the unchanged pipelines is checked against each run's oof.csv and submission.csv, and hybrid11 against runs/probes/hybrid11.csv.
Prints cv_testmix, the paired speaker-group bootstrap against hybrid11 and the post-correction topic check (seed-42 runs);
writes runs/probes/<variant>.csv (test.csv order).
usage: python src/variants.py [variant ...]      (default: all; see VARIANTS)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "kaggle" / "final")]
import replay as R  # noqa: E402
from paired_bootstrap import paired  # noqa: E402
from pooling import test_groups  # noqa: E402

CACHE = ROOT / "cache"
BASE = ("1015-070513", "1015-114531", "1015-165432", "1015-172136")
COMBO = ("1016-100139", "1016-140135", "1016-140601", "1016-141007")
STAGES = {"base": dict(shrink=False, pool_s=0.0, corr=False), "combo": dict(shrink=True, pool_s=0.3, corr=True)}
PARAS = ("para_bands", "para_reversed", "para_raters", "para_only", "para_short")


def judge_table(keys) -> pd.DataFrame:
    """The correction's 21 judge numbers with their names (same values and order as replay.judge_features)."""
    J1 = pd.read_csv(CACHE / "qwen35" / "qwen35_judge.csv").set_index(["split", "filename"])
    J2 = pd.read_csv(CACHE / "qwen35" / "qwen35_judge2.csv").set_index(["split", "filename"])
    cols = {}
    for v in R.JUDGE_VIEWS:
        for p in R.JUDGE_PROMPTS:
            P = J1.loc[keys, [f"{v}_{p}_p{i}" for i in range(1, 6)]].values
            P = P / P.sum(1, keepdims=True)
            cols[f"judge35_{v}_{p}_ev"] = P @ np.arange(1, 6)
            cols[f"judge35_{v}_{p}_lo"] = np.log(P[:, 3:].sum(1) + 1e-9) - np.log(P[:, :3].sum(1) + 1e-9)
    for c in sorted({c.rsplit("_p", 1)[0] for c in J2.columns if "_p" in c and not c.endswith("_top")}):
        nd = 6 if c.endswith("cefr") else 5
        P = J2.loc[keys, [f"{c}_p{i}" for i in range(1, nd + 1)]].values
        cols[f"j2_{c}_ev"] = (P / P.sum(1, keepdims=True)) @ np.arange(1, nd + 1)
    return pd.DataFrame(cols, index=pd.MultiIndex.from_tuples(keys))


def examiner(J: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Round-4 examiner variants (as experiment.py EXAMINER): sc = mean + spread of 6 rubric paraphrases (Whisper, verbatim),
    sc_fs = sc with the few-shot grade in the Whisper mean, fs = the Whisper rubric grade replaced by the few-shot grade."""
    P4 = pd.read_csv(CACHE / "qwen35" / "qwen35_judge4.csv").set_index(["split", "filename"]).loc[J.index]
    P1 = pd.read_csv(CACHE / "qwen35" / "qwen35_judge.csv").set_index(["split", "filename"]).loc[J.index]
    ev = lambda D, c: (lambda M: (M / M.sum(1, keepdims=True)) @ np.arange(1, 6))(D[[f"{c}_p{i}" for i in range(1, 6)]].values)
    J = J.copy()
    if kind in ("sc", "sc_fs"):
        J = J.drop(columns=[f"judge35_{v}_holistic_{k}" for v in ("whisper", "verbatim") for k in ("ev", "lo")])
        for v in ("whisper", "verbatim"):
            E = np.c_[ev(P1, f"{v}_holistic"), np.column_stack([ev(P4, f"{v}_{q}") for q in PARAS])]
            if kind == "sc_fs" and v == "whisper":
                E = np.c_[E, ev(P4, "whisper_fewshot")]
            J[f"sc_{v}_ev"], J[f"sc_{v}_sd"] = E.mean(1), E.std(1)
    elif kind == "fs":
        J = J.drop(columns=["judge35_whisper_holistic_ev", "judge35_whisper_holistic_lo"])
        J["fs_whisper_ev"] = ev(P4, "whisper_fewshot")
    else:
        raise ValueError(kind)
    return J


class Ctx:
    def __init__(self):
        self.C = {r: dict(np.load(ROOT / "runs" / r / "cands.npz", allow_pickle=True)) for r in BASE + COMBO}
        c0 = self.C[BASE[0]]
        self.fn, self.tfn, self.y, self.w = c0["filename"], c0["test_filename"], c0["y"], c0["w"]
        self.bt, self.btt = c0["batch"], c0["batch_test"]
        FD = pd.read_csv(ROOT / "kaggle" / "final" / "folds_all.csv")
        self.folds = {}
        for s in FD.seed.unique():
            f = FD[FD.seed == s].set_index("filename").loc[self.fn, "fold"].values
            self.folds[int(s)] = [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(5)]
        G = pd.read_csv(CACHE / "speaker_groups.csv")
        self.groups = G[G.split == "train"].set_index("filename").loc[self.fn, "group"].values
        self.groups_test = test_groups(CACHE, self.fn, self.tfn)
        self.keys = [("train", f) for f in self.fn]
        self.tkeys = [("test", f) for f in self.tfn]
        Jall = judge_table(self.keys + self.tkeys)
        self.J, self.Jt = Jall.loc[self.keys], Jall.loc[self.tkeys]
        I = pd.read_csv(CACHE / "insight.csv").set_index(["split", "filename"])
        tp = I.loc[self.keys, "topic"].values
        g = self.groups
        self.tfolds = [(np.where((tp != t) & ~np.isin(g, np.unique(g[tp == t])))[0], np.where(tp == t)[0]) for t in sorted(set(tp))]
        self.test_order = pd.read_csv(CACHE / "test.csv").filename.values

    def wr(self, p):
        return float(np.sqrt(np.sum(self.w * (self.y - p) ** 2) / np.sum(self.w)))


def run_pipe(cx: Ctx, c, *, shrink, pool_s, corr, X=None, Xt=None, extra=None, extra_test=None, corr_fn=None):
    """One run's stacking -> (out-of-fold prediction, test prediction). corr: apply the examiner correction with inputs X / Xt
    (default: the 21 judge numbers) plus the optional fold-aware extra(train_rows) / extra_test."""
    CO_s, CT_s, TO, y, w, bt = c["oof"], c["test"], c["topic"], cx.y, cx.w, cx.bt
    test = np.clip(R.apply_blend(CT_s.mean(0), cx.btt, R.blend_weights(TO, y, w, bt, shrink)), *R.CLIP)
    seed_oofs = []
    for i, s in enumerate(int(v) for v in c["seeds"]):
        o = np.zeros(len(y))
        for a, b in cx.folds[s]:
            o[b] = R.apply_blend(CO_s[i][b], bt[b], R.blend_weights(TO[a], y[a], w[a], bt[a], shrink))
        seed_oofs.append(np.clip(o, *R.CLIP))
    oof = np.clip(np.mean(seed_oofs, 0), *R.CLIP)
    if corr:
        X = cx.J.values if X is None else X
        Xt = cx.Jt.values if Xt is None else Xt
        full = lambda rows: X if extra is None else np.c_[X, extra(rows)]
        new = []
        for i, s in enumerate(int(v) for v in c["seeds"]):
            o, oc = seed_oofs[i], seed_oofs[i].copy()
            for a, b in cx.folds[s]:
                Xf = full(a)
                f = (corr_fn or R.ridge_correction)(np.c_[Xf[a], o[a]], y[a] - o[a], w[a])
                oc[b] = o[b] + f(np.c_[Xf[b], o[b]])
            new.append(np.clip(oc, *R.CLIP))
        Xf = full(np.arange(len(y)))
        Xtf = Xt if extra is None else np.c_[Xt, extra_test]
        f = (corr_fn or R.ridge_correction)(np.c_[Xf, oof], y - oof, w)
        test = np.clip(test + f(np.c_[Xtf, test]), *R.CLIP)
        seed_oofs = new
        oof = np.clip(np.mean(seed_oofs, 0), *R.CLIP)
    if pool_s > 0:
        seed_oofs = [np.clip(R.pool(o, cx.groups, pool_s), *R.CLIP) for o in seed_oofs]
        oof = np.clip(np.mean(seed_oofs, 0), *R.CLIP)
        test = np.clip(R.pool(test, cx.groups_test, pool_s), *R.CLIP)
    return oof, test


def topic_check(cx: Ctx, c, *, shrink, corr, X=None, extra=None, corr_fn=None):
    """Topic-held-out OOF of one run's candidates (fit outside one prompt cluster), after its correction stage (as the final
    notebook)."""
    y, w, bt = cx.y, cx.w, cx.bt
    nt = np.zeros(len(y))
    for _, b in cx.tfolds:
        rows = np.setdiff1d(np.arange(len(y)), b)
        nt[b] = R.apply_blend(c["topic"][b], bt[b], R.blend_weights(c["topic"][rows], y[rows], w[rows], bt[rows], shrink))
    nt = np.clip(nt, *R.CLIP)
    if not corr:
        return cx.wr(nt)
    X = cx.J.values if X is None else X
    out = nt.copy()
    for a, b in cx.tfolds:
        Xf = X if extra is None else np.c_[X, extra(a)]
        f = (corr_fn or R.ridge_correction)(np.c_[Xf[a], nt[a]], y[a] - nt[a], w[a])
        out[b] = nt[b] + f(np.c_[Xf[b], nt[b]])
    return cx.wr(np.clip(out, *R.CLIP))


def augment(cx: Ctx, run, add):
    """Append candidate columns to a run: add(run, seeds) -> [(oof per seed, test per seed, topic-held-out oof), ...]."""
    c = dict(cx.C[run])
    seeds = [int(s) for s in c["seeds"]]
    for o_s, t_s, tp in add(run, seeds):
        c["oof"] = np.concatenate([c["oof"], np.clip(np.stack(o_s), *R.CLIP)[..., None]], axis=2)
        c["test"] = np.concatenate([c["test"], np.clip(np.stack(t_s), *R.CLIP)[..., None]], axis=2)
        c["topic"] = np.concatenate([c["topic"], np.clip(tp, *R.CLIP)[:, None]], axis=1)
    return c


def stage(spec, pipe):
    st = STAGES[pipe] | spec.get(pipe, {})
    return {k: v for k, v in st.items() if k != "add"}, st.get("add")


def hybrid(cx: Ctx, spec: dict):
    """spec: {"base": stage overrides, "combo": stage overrides} -> (oof, test, per-pipeline bags)."""
    bags = {}
    for pipe, runs in (("base", BASE), ("combo", COMBO)):
        st, add = stage(spec, pipe)
        outs = [run_pipe(cx, augment(cx, r, add) if add else cx.C[r], **st) for r in runs]
        bags[pipe] = (np.mean([o for o, _ in outs], 0), np.mean([t for _, t in outs], 0))
    sh, sht = cx.bt == "short", cx.btt == "short"
    comb = lambda b, c, m: np.clip(np.where(m, (b + c) / 2, c), *R.CLIP)
    return comb(bags["base"][0], bags["combo"][0], sh), comb(bags["base"][1], bags["combo"][1], sht), bags


def verify(cx: Ctx):
    for pipe, runs in (("base", BASE), ("combo", COMBO)):
        for r in runs:
            o, t = run_pipe(cx, cx.C[r], **STAGES[pipe])
            O = pd.read_csv(ROOT / "runs" / r / "oof.csv").set_index("filename").loc[cx.fn, "oof"].values
            S = pd.read_csv(ROOT / "runs" / r / "submission.csv").set_index("filename").loc[cx.tfn, "label"].values
            assert np.abs(o - O).max() < 1e-9 and np.abs(t - S).max() < 1e-9, (r, np.abs(o - O).max(), np.abs(t - S).max())
    o, t, _ = hybrid(cx, {})
    H = pd.read_csv(ROOT / "runs" / "probes" / "hybrid11.csv").set_index("filename").loc[cx.tfn, "label"].values
    assert np.abs(t - H).max() < 1e-9, np.abs(t - H).max()
    print(f"verified: 8 runs' oof.csv + submission.csv and hybrid11.csv reproduced (cv_testmix {cx.wr(o):.4f})")
    return o, t


def variants(cx: Ctx) -> dict:
    """name -> spec. Built lazily so a variant's inputs load only when asked for."""
    from pairwise_feats import Pairwise
    V = {}
    pw = None

    def pwf():
        nonlocal pw
        pw = pw or Pairwise(cx.fn, cx.tfn)
        return dict(extra=pw.train, extra_test=pw.test())
    for kind in ("sc", "sc_fs", "fs"):
        V[f"exam_{kind}"] = lambda k=kind: {"combo": dict(X=examiner(cx.J, k).values, Xt=examiner(cx.Jt, k).values)}
    V["pairwise"] = lambda: {"combo": pwf()}
    V["base_corr"] = lambda: {"base": dict(corr=True)}
    V["base_corr_pairwise"] = lambda: {"base": dict(corr=True) | pwf(), "combo": pwf()}

    # Phi-4 (kaggle/phi4_judge): examiner numbers next to the Qwen ones in the correction; hidden-state heads as new candidates
    def phi4_X():
        import phi4_feats
        P = phi4_feats.judge_numbers(cx.keys + cx.tkeys)
        return dict(X=np.c_[cx.J.values, P.loc[cx.keys].values], Xt=np.c_[cx.Jt.values, P.loc[cx.tkeys].values])
    hd = {}

    def phi4_add(run, seeds):
        import phi4_feats
        if not hd:
            hd.update(phi4_feats.heads(cx.fn, cx.tfn, cx.y, cx.groups, cx.folds, cx.tfolds))
        return [([h["oof"][s] for s in seeds], [h["test"][s] for s in seeds], h["topic"]) for h in hd.values()]
    V["phi4_corr"] = lambda: {"combo": phi4_X()}
    V["phi4_corr_both"] = lambda: {"base": dict(corr=True) | phi4_X(), "combo": phi4_X()}
    V["phi4_heads"] = lambda: {"base": dict(add=phi4_add), "combo": dict(add=phi4_add)}
    V["phi4_all"] = lambda: {"base": dict(corr=True, add=phi4_add) | phi4_X(), "combo": dict(add=phi4_add) | phi4_X()}

    # KAN in the correction stage (src/kan_correction.py): squared-error loss, or the competition metric as the loss
    from functools import partial
    from kan_correction import kan_correction
    for loss in ("mse", "board"):
        sfx = "" if loss == "mse" else "_board"
        V[f"kan_corr{sfx}"] = lambda l=loss: {"combo": dict(corr_fn=partial(kan_correction, loss=l))}
        V[f"base_corr_kan{sfx}"] = lambda l=loss: {"base": dict(corr=True, corr_fn=partial(kan_correction, loss=l)),
                                                  "combo": dict(corr_fn=partial(kan_correction, loss=l))}
    return V


def main(names):
    cx = Ctx()
    h_oof, h_test = verify(cx)
    V = variants(cx)
    rows = []
    for name in names or list(V):
        spec = V[name]()
        o, t, bags = hybrid(cx, spec)
        r = {k: paired(o, h_oof, cx.y, cx.w if m is None else np.ones(len(cx.y)), cx.groups, mask=m)
             for k, m in (("testmix", None), ("short", cx.bt == "short"), ("long", cx.bt == "long"))}
        topics = []
        for pipe, runs in (("base", BASE), ("combo", COMBO)):
            st, add = stage(spec, pipe)
            if pipe in spec:
                kw = dict(shrink=st["shrink"], corr=st["corr"], X=st.get("X"), extra=st.get("extra"), corr_fn=st.get("corr_fn"))
                old = topic_check(cx, cx.C[runs[0]], shrink=st["shrink"], corr=STAGES[pipe]["corr"])
                new = topic_check(cx, augment(cx, runs[0], add) if add else cx.C[runs[0]], **kw)
                topics.append(f"{pipe} {old:.4f}->{new:.4f}")
        sub = pd.DataFrame({"filename": cx.tfn, "label": t}).set_index("filename").loc[cx.test_order].reset_index()
        path = ROOT / "runs" / "probes" / f"{name}.csv"
        sub.to_csv(path, index=False)
        print(f"{name:20s} cv_testmix {cx.wr(o):.4f} ({cx.wr(o) - cx.wr(h_oof):+.4f})  95% [{r['testmix']['lo']:+.4f}, "
              f"{r['testmix']['hi']:+.4f}]  short {r['short']['diff']:+.4f}  long {r['long']['diff']:+.4f}  | topic {'; '.join(topics)}"
              f"  | test mean {t.mean():.3f} (hybrid11 {h_test.mean():.3f}), max |diff| {np.abs(t - h_test).max():.3f}")
        rows.append(dict(variant=name, cv_testmix=cx.wr(o), d=cx.wr(o) - cx.wr(h_oof), lo=r["testmix"]["lo"], hi=r["testmix"]["hi"],
                         d_short=r["short"]["diff"], d_long=r["long"]["diff"], topic="; ".join(topics), file=str(path)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main(sys.argv[1:])
