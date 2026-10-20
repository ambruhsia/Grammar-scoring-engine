"""One experiment: build candidates from src/experiment.py, blend, score with repeated-seed CV,
write runs/<id>/ (submission, OOF, weights, code snapshot) and append a verdict to experiments.csv.

    python src/run.py --desc "what changed and why"              # full: fold seeds 42 1337 2024
    python src/run.py --fast --desc "..."                         # screen: seed 42 only
    python src/run.py --seeds 7 99 555 --desc "confirm"           # fresh seeds, guards against CV overfit
    python src/run.py --rebaseline --desc "new cache from <kernel>"  # restart comparisons
    python src/run.py --folds random --desc "..."                 # old random folds, comparison only

Folds are speaker-grouped (see data.py). Metrics per seed and averaged: cv_grouped (plain RMSE),
cv_testmix (primary: each recording batch weighted to its test share), cv_short, cv_long.
Gate: cv_testmix gains >= MIN_GAIN, better on >= 2/3 seeds, cv_grouped not worse by more than 0.002.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  — switches to .venv's python if needed

import argparse
import importlib
import inspect
import json
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

from common import (CLIP, FULL_SEEDS, ROOT, RUNS, hill_climb, load_config, per_seed,  # noqa: E402
                    read_ledger, rmse, seeds_key, utc_now, wrmse, write_ledger)

GROUPED_SLACK = 0.002     # cv_grouped may get worse by at most this much on an accepted change
from data import load  # noqa: E402


def git_rev() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return (rev or "nogit") + ("+dirty" if dirty else "")
    except OSError:
        return "nogit"


def reference(rows, key, folds):
    """Best kept run since the latest baseline on the same fold seeds and fold scheme."""
    same = [r for r in rows if r["seeds"] == key and (r.get("folds") or "random") == folds]
    starts = [i for i, r in enumerate(same) if r["verdict"] == "baseline"]
    window = same[starts[-1]:] if starts else []
    kept = [r for r in window if r["verdict"] in ("baseline", "accepted")]
    return min(kept, key=lambda r: float(r["cv"])) if kept else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc", required=True, help="one line: the change and the reason for it")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(FULL_SEEDS))
    ap.add_argument("--fast", action="store_true", help="seed 42 only, for screening")
    ap.add_argument("--folds", choices=["grouped", "random"], default="grouped",
                    help="speaker-grouped folds (default) or the old random folds (comparison only)")
    ap.add_argument("--rebaseline", action="store_true",
                    help="record this run as the new baseline (after the cache changed)")
    args = ap.parse_args()
    seeds = [42] if args.fast else args.seeds
    cfg = load_config()
    min_gain = float(cfg["MIN_GAIN"])

    t0 = time.time()
    ctx = load(seeds, folds=args.folds)
    import experiment
    importlib.reload(experiment)
    print(f"train {len(ctx.T)}  test {len(ctx.TE)}  features {len(ctx.FEATS)}  fold seeds {seeds}  "
          f"folds {args.folds}")

    cands = experiment.make_candidates(ctx)
    names = list(cands)
    for n, (o, t) in cands.items():
        assert len(o) == len(seeds) and len(t) == len(seeds), f"{n}: need one array per seed"
        assert all(len(a) == len(ctx.y) for a in o), f"{n}: OOF length != train rows"
        assert all(len(a) == len(ctx.TE) for a in t), f"{n}: test length != test rows"
        assert all(np.isfinite(a).all() for a in o + t), f"{n}: NaN/inf in predictions"

    CO_s = [np.column_stack([np.clip(cands[n][0][i], *CLIP) for n in names]) for i in range(len(seeds))]
    CT_s = [np.column_stack([np.clip(cands[n][1][i], *CLIP) for n in names]) for i in range(len(seeds))]
    CO, CT = np.mean(CO_s, 0), np.mean(CT_s, 0)

    # contract: blend(CO, y) / blend(CO, y, clip_weights) / blend(CO, y, clip_weights, batch) -> weights (K,)
    # or {"short": (K,), "long": (K,)} for per-batch weights
    n_par = len(inspect.signature(experiment.blend).parameters) if hasattr(experiment, "blend") else 0

    def fit_blend(C, m):
        if not n_par:
            return hill_climb(C, ctx.y[m])[0]
        bb = getattr(ctx, "blend_batch", ctx.batch)     # Phase I: blend groups (default = the recording batch)
        return experiment.blend(*(C, ctx.y[m], ctx.w[m], bb[m])[:max(2, min(n_par, 4))])

    def apply_w(C, batch, wf):                          # per-group weights {group: (K,)} or one shared (K,)
        if not isinstance(wf, dict):
            return C @ np.asarray(wf, float)
        out = np.zeros(len(C))
        for k, v in wf.items():
            m = batch == k
            out[m] = C[m] @ np.asarray(v, float)
        return out

    TO = None                                       # G11: topic-held-out OOF matrix, when the experiment builds it
    if getattr(ctx, "topic_cands", None) is not None:
        TO = np.column_stack([np.clip(ctx.topic_cands[n], *CLIP) for n in names])
    on_topic = TO is not None and getattr(experiment, "BLEND_ON_TOPIC", False)
    w = fit_blend(TO if on_topic else CO, np.arange(len(ctx.y)))
    per_batch = isinstance(w, dict)
    w = {k: np.asarray(v, float) for k, v in w.items()} if per_batch else np.asarray(w, float)

    BB, BBT = getattr(ctx, "blend_batch", ctx.batch), getattr(ctx, "blend_batch_test", ctx.batch_test)

    def apply(C, batch):
        batch = BB if batch is ctx.batch else BBT if batch is ctx.batch_test else batch
        return apply_w(C, batch, w)

    blend_oof, blend_test = np.clip(apply(CO, ctx.batch), *CLIP), np.clip(apply(CT, ctx.batch_test), *CLIP)
    seed_oofs = [np.clip(apply(C, ctx.batch), *CLIP) for C in CO_s]
    nested = getattr(experiment, "NESTED_BLEND", False)
    if nested:                                      # blend weights refit inside each fold: honest blend OOF
        seed_oofs = []
        for i, C in enumerate(CO_s):
            o = np.zeros(len(ctx.y))
            for a, b in ctx.folds[seeds[i]]:
                wf = fit_blend(TO[a] if on_topic else C[a], a)
                o[b] = apply_w(C[b], BB[b], wf)
            seed_oofs.append(np.clip(o, *CLIP))
        blend_oof = np.clip(np.mean(seed_oofs, 0), *CLIP)
    cal_mode = getattr(experiment, "CALIBRATE", False)
    if cal_mode:     # D4: weighted linear fit of OOF to label; G5(c) "batch": a_batch + c_batch * pred; nested in folds
        def lin(o_, y_, w_):
            A = np.c_[np.ones(len(o_)), o_] * np.sqrt(w_)[:, None]
            return np.linalg.lstsq(A, y_ * np.sqrt(w_), rcond=None)[0]

        groups_tr = [ctx.batch == "short", ctx.batch == "long"] if cal_mode == "batch" else [np.ones(len(ctx.y), bool)]
        groups_te = ([ctx.batch_test == "short", ctx.batch_test == "long"] if cal_mode == "batch"
                     else [np.ones(len(ctx.batch_test), bool)])
        cal_oofs, cal_tests = [], []
        for i, (o, Ct) in enumerate(zip(seed_oofs, CT_s)):
            t = np.clip(apply(Ct, ctx.batch_test), *CLIP)
            oc, tc = np.zeros_like(o), np.zeros_like(t)
            for a, b in ctx.folds[seeds[i]]:
                for g in groups_tr:
                    ga, gb = a[g[a]], b[g[b]]
                    coef = lin(o[ga], ctx.y[ga], ctx.w[ga])
                    oc[gb] = coef[0] + coef[1] * o[gb]
            for g, gt in zip(groups_tr, groups_te):
                coef = lin(o[g], ctx.y[g], ctx.w[g])
                tc[gt] = coef[0] + coef[1] * t[gt]
            cal_oofs.append(np.clip(oc, *CLIP)); cal_tests.append(np.clip(tc, *CLIP))
        seed_oofs = cal_oofs
        blend_oof, blend_test = np.clip(np.mean(cal_oofs, 0), *CLIP), np.clip(np.mean(cal_tests, 0), *CLIP)
    post_mode = getattr(experiment, "POSTPROC", None)   # "ordinal" | "ordinal_em": grade distribution (+ EM prior adaptation)
    em_report = {}
    if post_mode:
        import ordinal_em as OE
        use_em = post_mode == "ordinal_em"
        pp_oofs, pp_tests = [], []
        for i, (o, Ct) in enumerate(zip(seed_oofs, CT_s)):
            t = np.clip(apply(Ct, ctx.batch_test), *CLIP)
            oc, tc = np.zeros_like(o), np.zeros_like(t)
            for a, b in ctx.folds[seeds[i]]:
                for bt in ("short", "long"):
                    ga, gb = a[ctx.batch[a] == bt], b[ctx.batch[b] == bt]
                    oc[gb] = OE.adapt(o[ga], ctx.y[ga], o[gb], use_em)[0]
            for bt in ("short", "long"):
                g, gt = ctx.batch == bt, ctx.batch_test == bt
                tc[gt], pi, pit = OE.adapt(o[g], ctx.y[g], t[gt], use_em)
                em_report.setdefault(bt, []).append((pi @ OE.LEVELS, pit @ OE.LEVELS, pi[OE.LEVELS >= 4].sum(), pit[OE.LEVELS >= 4].sum()))
            pp_oofs.append(np.clip(oc, *CLIP)); pp_tests.append(np.clip(tc, *CLIP))
        seed_oofs = pp_oofs
        blend_oof, blend_test = np.clip(np.mean(pp_oofs, 0), *CLIP), np.clip(np.mean(pp_tests, 0), *CLIP)
    if getattr(experiment, "RESID_CORR", False):   # Phase I: ridge on [extra features, prediction] -> residual, nested in folds
        from sklearn.linear_model import RidgeCV
        Xr, Xrt = experiment.resid_features(ctx)

        def corr_fit(Z, e, sw):
            mu, sd = Z.mean(0), Z.std(0) + 1e-9
            m = RidgeCV(alphas=np.logspace(-1, 4, 30)).fit((Z - mu) / sd, e, sample_weight=sw)
            return lambda Zn: m.predict((Zn - mu) / sd)

        new_oofs = []
        for i, o in enumerate(seed_oofs):
            oc = o.copy()
            for a, b in ctx.folds[seeds[i]]:
                f = corr_fit(np.c_[Xr[a], o[a]], ctx.y[a] - o[a], ctx.w[a])
                oc[b] = o[b] + f(np.c_[Xr[b], o[b]])
            new_oofs.append(np.clip(oc, *CLIP))
        f = corr_fit(np.c_[Xr, blend_oof], ctx.y - blend_oof, ctx.w)
        blend_test = np.clip(blend_test + f(np.c_[Xrt, blend_test]), *CLIP)
        seed_oofs = new_oofs
        blend_oof = np.clip(np.mean(seed_oofs, 0), *CLIP)
        print(f"residual correction on {Xr.shape[1]} extra features (nested in folds)")
    pool_s = float(getattr(experiment, "POOL_S", 0) or 0)     # Phase I: same-speaker pooling (src/pooling.py)
    if pool_s > 0:
        import pooling
        g_test = pooling.test_groups(ctx.cache, ctx.T.filename, ctx.TE.filename)
        seed_oofs = [np.clip(pooling.pool(o, ctx.groups, pool_s), *CLIP) for o in seed_oofs]
        blend_oof = np.clip(np.mean(seed_oofs, 0), *CLIP)
        blend_test = np.clip(pooling.pool(blend_test, g_test, pool_s), *CLIP)
        n_te = int((pd.Series(g_test).map(pd.Series(g_test).value_counts()) >= 2).sum())
        print(f"pooling s={pool_s}: {n_te} test clips in same-voice groups")
    short = ctx.batch == "short"

    def metrics(o):
        return {"grouped": rmse(ctx.y, o), "testmix": wrmse(ctx.y, o, ctx.w),
                "short": rmse(ctx.y[short], o[short]), "long": rmse(ctx.y[~short], o[~short])}

    M = metrics(blend_oof)
    if TO is not None:                              # cv_newtopic: blend fitted outside topic T, scored on T's topic OOF
        nt = np.zeros(len(ctx.y))
        FM = TO if on_topic else CO
        for _, b_t in ctx.topic_folds:
            rows = np.setdiff1d(np.arange(len(ctx.y)), b_t)
            wf = fit_blend(FM[rows], rows)
            nt[b_t] = apply_w(TO[b_t], BB[b_t], wf)
        nt = np.clip(nt, *CLIP)
        M["newtopic"], M["newtopic_plain"] = wrmse(ctx.y, nt, ctx.w), rmse(ctx.y, nt)
        M["newtopic_short"] = rmse(ctx.y[short], nt[short])
        if getattr(experiment, "RESID_CORR", False):  # Phase I fix (2026-10-17): the correction stage under prompt shift too:
            ntc = nt.copy()                            # fitted outside topic T (and T's speaker groups), applied to T
            for a_t, b_t in ctx.topic_folds:
                f = corr_fit(np.c_[Xr[a_t], nt[a_t]], ctx.y[a_t] - nt[a_t], ctx.w[a_t])
                ntc[b_t] = nt[b_t] + f(np.c_[Xr[b_t], nt[b_t]])
            ntc = np.clip(ntc, *CLIP)
            M["newtopic_post"], M["newtopic_post_short"] = wrmse(ctx.y, ntc, ctx.w), rmse(ctx.y[short], ntc[short])
        if post_mode:                               # EM simulation under a real shift: hold a topic out, adapt on its clips
            import ordinal_em as OE
            sim = {"ordinal": np.zeros(len(ctx.y)), "ordinal_em": np.zeros(len(ctx.y))}
            for _, b_t in ctx.topic_folds:
                rows = np.setdiff1d(np.arange(len(ctx.y)), b_t)
                for bt in ("short", "long"):
                    ra, rb = rows[ctx.batch[rows] == bt], b_t[ctx.batch[b_t] == bt]
                    if len(rb):
                        sim["ordinal"][rb] = OE.adapt(nt[ra], ctx.y[ra], nt[rb], False)[0]
                        sim["ordinal_em"][rb] = OE.adapt(nt[ra], ctx.y[ra], nt[rb], True)[0]
            em_report["sim"] = {"raw": wrmse(ctx.y, nt, ctx.w), **{k: wrmse(ctx.y, np.clip(v, *CLIP), ctx.w) for k, v in sim.items()}}
    Ms = [metrics(o) for o in seed_oofs]
    primary = "testmix" if args.folds == "grouped" else "grouped"
    cv, cv_seed = M[primary], [m[primary] for m in Ms]
    spread = max(cv_seed) - min(cv_seed)

    board = pd.DataFrame({"candidate": names, "oof_rmse": [rmse(ctx.y, CO[:, i]) for i in range(len(names))],
                          "oof_testmix": [wrmse(ctx.y, CO[:, i], ctx.w) for i in range(len(names))],
                          **({"oof_topic": [rmse(ctx.y, TO[:, i]) for i in range(len(names))]} if TO is not None else {}),
                          **({"weight": w["long"], "weight_short": w["short"]} if per_batch else {"weight": w})}
                         ).sort_values("oof_rmse")
    print("\ncandidates (seed-averaged OOF RMSE, blend weight):")
    print(board.head(15).to_string(index=False, float_format="%.4f"))

    # ---- write the run -------------------------------------------------------
    stamp = utc_now()
    run_id = stamp.strftime("%m%d-%H%M%S")
    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    sub = ctx.test[["filename"]].merge(pd.DataFrame({"filename": ctx.TE.filename, "label": blend_test}),
                                       on="filename", how="left")
    if sub.label.isna().any():
        print(f"!! {sub.label.isna().sum()} test files had no prediction — filled with train mean")
        sub.label = sub.label.fillna(ctx.y.mean())
    sub.to_csv(out / "submission.csv", index=False)
    pd.DataFrame({"filename": ctx.T.filename, "label": ctx.y, "oof": blend_oof,
                  **{f"oof_seed{s_}": o for s_, o in zip(seeds, seed_oofs)}}).to_csv(out / "oof.csv", index=False)
    board.to_csv(out / "board.csv", index=False)
    train_fit = None
    if getattr(ctx, "train_fit", None) is not None:     # Phase F: TRAINING RMSE of the final pipeline (in-sample refit)
        TF = np.column_stack([np.clip(ctx.train_fit[n], *CLIP) for n in names])
        train_fit = np.clip(apply(TF, ctx.batch), *CLIP)
        print(f"TRAINING RMSE {rmse(ctx.y, train_fit):.4f} (in-sample refit; OOF kept for {', '.join(ctx.train_fit_oofcols)})")
    np.savez(out / "cands.npz", names=np.array(names), oof=np.stack(CO_s), test=np.stack(CT_s), seeds=np.array(seeds),
             **({"train_fit": TF, "train_fit_pred": train_fit, "train_fit_oofcols": np.array(ctx.train_fit_oofcols)}
                if train_fit is not None else {}),
             filename=ctx.T.filename.values, test_filename=ctx.TE.filename.values, y=ctx.y, batch=ctx.batch,
             batch_test=ctx.batch_test, w=ctx.w, blend_oof=blend_oof, blend_test=blend_test,
             seed_oofs=np.stack(seed_oofs), **({"w_short": w["short"], "w_long": w["long"]} if per_batch else {"w": w}),
             **({"topic": TO} if TO is not None else {}))
    shutil.copy(Path(experiment.__file__), out / "experiment.py")

    # ---- verdict -------------------------------------------------------------
    rows = read_ledger()
    key = seeds_key(seeds)
    ref = None if args.rebaseline else reference(rows, key, args.folds)
    if ref is None:
        verdict, why = "baseline", ("rebaseline requested" if args.rebaseline
                                    else "first run on these fold seeds")
    else:
        gain = float(ref["cv"]) - cv
        rps = per_seed(ref)
        wins = sum(a < b for a, b in zip(cv_seed, rps))
        need = len(seeds) // 2 + 1
        ok = gain >= min_gain and wins >= need
        why = (f"vs {ref['run_id']} ({float(ref['cv']):.4f}): {primary} gain {gain:+.4f} "
               f"(need >= {min_gain}), better on {wins}/{len(seeds)} seeds (need {need})")
        if args.folds == "grouped" and ref.get("cv_grouped"):
            worse = M["grouped"] - float(ref["cv_grouped"])
            ok = ok and worse <= GROUPED_SLACK
            why += f", cv_grouped {worse:+.4f} (allowed <= +{GROUPED_SLACK})"
        verdict = "accepted" if ok else "rejected"

    rows.append({"run_id": run_id, "utc": stamp.strftime("%Y-%m-%d %H:%M:%S"), "desc": args.desc,
                 "seeds": key, "cv": f"{cv:.5f}", "cv_per_seed": "|".join(f"{v:.5f}" for v in cv_seed),
                 "cv_spread": f"{spread:.5f}", "verdict": verdict, "commit": git_rev(),
                 "folds": args.folds, "cv_grouped": f"{M['grouped']:.5f}",
                 "cv_grouped_per_seed": "|".join(f"{m['grouped']:.5f}" for m in Ms),
                 "cv_short": f"{M['short']:.5f}", "cv_long": f"{M['long']:.5f}",
                 "cv_newtopic": f"{M['newtopic']:.5f}" if "newtopic" in M else ""})
    write_ledger(rows)
    json.dump({"run_id": run_id, "desc": args.desc, "seeds": seeds, "folds": args.folds, "metric": primary,
               "nested_blend": nested, "calibrate": getattr(experiment, "CALIBRATE", False),
               "cv": cv, "cv_per_seed": cv_seed, "metrics": M, "metrics_per_seed": Ms,
               "verdict": verdict, "why": why,
               "weights": ({k: dict(zip(names, v.round(4).tolist())) for k, v in w.items()} if per_batch
                           else dict(zip(names, w.round(4).tolist()))),
               "minutes": round((time.time() - t0) / 60, 2)}, open(out / "summary.json", "w"), indent=2)

    for label, wv in (w.items() if per_batch else [("", w)]):
        print(f"\nblend weights{(' ' + label) if label else ''}: " + ", ".join(
            f"{n} {wt:.3f}" for n, wt in sorted(zip(names, wv), key=lambda p: -p[1]) if wt > 1e-3))
    print(f"\nRESULT   run {run_id}   {primary} {cv:.4f}   per seed {' '.join(f'{v:.4f}' for v in cv_seed)}"
          f"   spread {spread:.4f}   ({(time.time() - t0) / 60:.1f} min)")
    print(f"METRICS  cv_grouped {M['grouped']:.4f}   cv_testmix {M['testmix']:.4f}   "
          f"cv_short {M['short']:.4f}   cv_long {M['long']:.4f}   [folds {args.folds}"
          f"{', nested blend' if nested else ''}{', calibrated' if getattr(experiment, 'CALIBRATE', False) else ''}]")
    if getattr(ctx, "view_cands", None) is not None:   # G4: short views of long clips, blended with the short weights
        VC = np.column_stack([np.clip(ctx.view_cands[n], *CLIP) for n in names])
        wv = w["short"] if per_batch else w
        vp = np.clip(VC @ wv, *CLIP)
        op = blend_oof[ctx.view_src]
        yv = ctx.y[ctx.view_src]
        pd.DataFrame({"filename": ctx.T.filename.values[ctx.view_src], "label": yv, "orig_pred": op, "view_pred": vp}
                     ).to_csv(out / "views.csv", index=False)
        print(f"VIEWS    n {len(vp)}  mean shift {np.mean(vp - op):+.4f}  mean |shift| {np.mean(np.abs(vp - op)):.4f}  "
              f"by label: " + " ".join(f"{lv:g}:{np.mean(vp[yv == lv] - op[yv == lv]):+.3f}" for lv in sorted(set(yv)) if lv >= 2))
        print("VIEWS    feature mean change (view - clip): " + ", ".join(f"{k} {v:+.3g}" for k, v in ctx.view_shift.items()))
    short_hi = (ctx.batch == "short") & (ctx.y >= 4.5)
    print(f"SHORT>=4.5 mean residual {np.mean(ctx.y[short_hi] - blend_oof[short_hi]):+.4f} (n {int(short_hi.sum())})")
    if em_report:
        for bt in ("short", "long"):
            if bt in em_report:
                m = np.mean(em_report[bt], 0)
                print(f"EMPRIOR  test-{bt}: adapted mean grade {m[0]:.3f} (train {m[1]:.3f}), share >= 4.0 {m[2]:.3f} (train {m[3]:.3f})")
        if "sim" in em_report:
            print("EMSIM    topic-held-out (testmix): " + "  ".join(f"{k} {v:.4f}" for k, v in em_report["sim"].items()))
    if "newtopic" in M:
        print(f"NEWTOPIC cv_newtopic {M['newtopic']:.4f} (plain {M['newtopic_plain']:.4f})  blend fitted on "
              f"{'topic-held-out' if on_topic else 'speaker-grouped'} OOF")
        if "newtopic_post" in M:
            print(f"NEWTOPIC after the residual correction {M['newtopic_post']:.4f} (short {M['newtopic_post_short']:.4f}; "
                  f"before it {M['newtopic']:.4f}, short {M['newtopic_short']:.4f})")
    for s_, m in zip(seeds, Ms):
        print(f"  seed {s_:>4}: grouped {m['grouped']:.4f}  testmix {m['testmix']:.4f}  "
              f"short {m['short']:.4f}  long {m['long']:.4f}")
    print(f"VERDICT  {verdict.upper()}  {why}")
    ref_oof = RUNS / ref["run_id"] / "oof.csv" if ref is not None else None
    if ref_oof is not None and ref_oof.exists():       # 2026-10-17: interval for "new minus reference" (src/paired_bootstrap.py)
        from paired_bootstrap import paired
        old = pd.read_csv(ref_oof).set_index("filename").loc[ctx.T.filename, "oof"].values
        for name, msk in (("testmix", None), ("short", ctx.batch == "short"), ("long", ctx.batch == "long")):
            r_ = paired(blend_oof, old, ctx.y, ctx.w if msk is None else np.ones(len(ctx.y)), ctx.groups, mask=msk)
            print(f"BOOTSTRAP vs {ref['run_id']} {name:8s} new-old {r_['diff']:+.4f}  95% [{r_['lo']:+.4f}, {r_['hi']:+.4f}]  "
                  f"P(new better) {r_['p_better']:.2f}  (speaker groups resampled)")
    nb = ctx.cache / "notebook_submission.csv"
    if verdict == "baseline" and nb.exists():
        m = sub.merge(pd.read_csv(nb), on="filename", suffixes=("", "_nb"))
        print(f"vs the notebook's own submission: {len(m)} shared rows, mean |diff| "
              f"{(m.label - m.label_nb).abs().mean():.4f}, max {(m.label - m.label_nb).abs().max():.4f}, "
              f"corr {m.label.corr(m.label_nb):.4f}")
    subs = [r for r in rows if r.get("public_lb")]
    if subs:
        ratios = [float(r["public_lb"]) / float(r["cv"]) for r in subs]
        print(f"public/CV ratio of past loop submissions: {' '.join(f'{x:.3f}' for x in ratios)}")
    if verdict == "accepted" and len(seeds) < 3:
        print("note: a --fast pass is a screen only; run the full 3-seed version before keeping it")


if __name__ == "__main__":
    main()
