"""G12: transductive pseudo-labelling, simulated on held-out topics (pre-registered by Amrita 2026-10-14).

For each held-out topic T (G11 topic folds: train outside T, speaker-safe) and each fold seed:
  before  the current pipeline (G11 state: leak-free members, Qwen heads, families on denoised labels, per-batch NNLS blend fitted
          outside T on topic-held-out OOF) predicts T;
  after   T's clips, labelled with those predictions (sample_weight 0.5), join the training data of the locally refittable members
          (audio / emb ridge heads, Qwen ridge heads, the 9 families on the updated head values); refit once; re-predict T with
          the same blend weights.
Pass rule: cv_newtopic improves by >= MIN_GAIN, on >= 2 of 3 seeds, and in at least half the topics. Simplification: the Qwen
heads' ridge alpha is the one their inner GroupKFold picked in the "before" fit (a single refit, not a new search); kNN ignores
sample weights (it has none). DeBERTa and ft_wavlm can't be refit here and keep their values.

    python src/pseudo_label.py      # writes reports/phaseG/g12_pseudo_sim.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bootstrap  # noqa: E402,F401  — switches to .venv's python if needed
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import experiment as E
import leakfree
import qwen_feats
from common import CLIP, FULL_SEEDS, fit_weighted, load_config, wrmse, rmse
from data import load
from topic_cv import topic_folds

ROOT = Path(__file__).resolve().parents[1]
PSEUDO_W = 0.5


def main():
    min_gain = float(load_config()["MIN_GAIN"])
    ctx = load(FULL_SEEDS)
    ctx.y_fit = E.denoised(ctx, E.DENOISE_LAMBDA)
    _, fixed, dropped = leakfree.build(ctx, E.DROP_FT_WAVLM)
    for c, (tr, te) in fixed.items():
        ctx.T[c], ctx.TE[c] = tr, te
    tf, topic = topic_folds(ctx)
    n = len(ctx.y)
    clip = lambda v: np.clip(v, *CLIP)

    # ---- level-1 refittable members: inputs, targets, topic-held-out "before" fits -----------------------------------
    members = {}
    for c, (X, Xt, model) in leakfree.member_inputs(ctx).items():
        members[c] = {"X": X, "target": ctx.y_fit, "model": model, "groups": None}
    for c, (X, Xt, model) in qwen_feats.member_inputs(ctx).items():
        members[c] = {"X": X, "target": ctx.y_fit, "model": model, "groups": ctx.groups}
    before_mem = {c: np.zeros(n) for c in members}
    alpha = {}
    for c, m in members.items():
        for k, (a, b) in enumerate(tf):
            if m["groups"] is not None:
                f = fit_weighted(clone(m["model"]), m["X"][a], m["target"][a], None, groups=m["groups"][a])
                alpha[(c, k)] = f.best_params_["ridge__alpha"]
            else:
                f = clone(m["model"]).fit(m["X"][a], m["target"][a])
            before_mem[c][b] = f.predict(m["X"][b])
        print(f"  member {c}: topic-held-out fits done", flush=True)
        ctx.T[c] = clip(before_mem[c])
        if c not in ctx.FEATS:
            ctx.FEATS.append(c)
            ctx.med[c] = 3.5
    use = [c for c in ctx.FEATS if c not in dropped]
    fam_names = list(E.zoo(0))
    names = fam_names + [c for c in ctx.T.columns if c.endswith("_pred") and c not in dropped]
    Xf = ctx.T[use].fillna(ctx.med[use]).values

    def fam_fit_predict(model, Xa, ya, wa, Xb):
        return clip(fit_weighted(clone(model), Xa, ya, wa).predict(Xb))

    rows_all = np.arange(n)
    res = []
    for s in FULL_SEEDS:
        zoo = E.zoo(s)
        # before: families' topic-held-out OOF for this seed
        fam_before = {f: np.zeros(n) for f in fam_names}
        for a, b in tf:
            for f in fam_names:
                fam_before[f][b] = fam_fit_predict(zoo[f], Xf[a], ctx.y_fit[a], None, Xf[b])
        TO = np.column_stack([fam_before[c] if c in fam_before else clip(ctx.T[c].values) for c in names])
        before, after = np.zeros(n), np.zeros(n)
        for k, (a, b) in enumerate(tf):
            rows = np.setdiff1d(rows_all, b)
            wb = E.blend(TO[rows], ctx.y[rows], ctx.w[rows], ctx.batch[rows])
            apply = lambda M, idx: np.where(ctx.batch[idx] == "short", M @ wb["short"], M @ wb["long"])
            before[b] = clip(apply(TO[b], b))
            pseudo = before[b]
            # after: members refit with T's pseudo-labelled clips at weight 0.5
            sw = np.r_[np.ones(len(a)), np.full(len(b), PSEUDO_W)]
            ab = np.r_[a, b]
            Xb_f = Xf[b].copy()
            mem_after = {}
            for c, m in members.items():
                tgt = np.r_[m["target"][a], pseudo]
                if m["groups"] is not None:
                    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha[(c, k)]))
                else:
                    model = m["model"]
                f = fit_weighted(clone(model), m["X"][ab], tgt, sw)
                mem_after[c] = clip(f.predict(m["X"][b]))
                Xb_f[:, use.index(c)] = mem_after[c]
            Xab = np.vstack([Xf[a], Xb_f])
            yab = np.r_[ctx.y_fit[a], pseudo]
            fam_after = {f: fam_fit_predict(zoo[f], Xab, yab, sw, Xb_f) for f in fam_names}
            TA = np.column_stack([fam_after[c] if c in fam_after else mem_after[c] if c in mem_after else clip(ctx.T[c].values[b])
                                  for c in names])
            after[b] = clip(apply(TA, b))
            res.append({"seed": s, "topic": int(topic[b][0]), "n": len(b), "rmse_before": rmse(ctx.y[b], before[b]),
                        "rmse_after": rmse(ctx.y[b], after[b])})
        nb, na = wrmse(ctx.y, before, ctx.w), wrmse(ctx.y, after, ctx.w)
        res.append({"seed": s, "topic": "ALL", "n": n, "rmse_before": nb, "rmse_after": na})
        print(f"seed {s}: cv_newtopic before {nb:.4f}  after {na:.4f}  (gain {nb - na:+.4f})", flush=True)

    R = pd.DataFrame(res)
    out = ROOT / "reports" / "phaseG"
    out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "g12_pseudo_sim.csv", index=False)
    tot = R[R.topic == "ALL"]
    per_topic = R[R.topic != "ALL"].groupby("topic")[["n", "rmse_before", "rmse_after"]].mean()
    gain = float(tot.rmse_before.mean() - tot.rmse_after.mean())
    seeds_better = int((tot.rmse_after < tot.rmse_before).sum())
    topics_better = int((per_topic.rmse_after < per_topic.rmse_before).sum())
    print("\nper topic (mean over seeds):\n", per_topic.round(4).to_string())
    print(f"\ncv_newtopic mean gain {gain:+.4f} (need >= {min_gain}); better on {seeds_better}/3 seeds (need 2); "
          f"better in {topics_better}/{len(per_topic)} topics (need {int(np.ceil(len(per_topic) / 2))})")
    ok = gain >= min_gain and seeds_better >= 2 and topics_better >= np.ceil(len(per_topic) / 2)
    print(f"G12 simulation: {'PASS -> apply to test' if ok else 'FAIL -> do not apply'}")


if __name__ == "__main__":
    main()
