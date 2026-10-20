import json, numpy as np, pandas as pd
def jl(p):
    return {(r["split"], r["filename"]): r for r in map(json.loads, open(p, encoding="utf-8")) }
W, C, V = jl("cache/asr_c.jsonl"), jl("cache/ctc.jsonl"), jl("cache/verbatim/asr_v.jsonl")
def seg_feats(r, pre):
    S = r.get("segments") or []
    d = np.array([max(s["end"] - s["start"], 0.01) for s in S]) if S else np.array([1.0])
    lp = np.array([s.get("avg_logprob", np.nan) for s in S]) if S else np.array([np.nan])
    ns = np.array([s.get("no_speech_prob", np.nan) for s in S]) if S else np.array([np.nan])
    fb = np.array([s.get("temperature", 0) > 0 for s in S]) if S else np.array([False])
    wp = np.array([w[3] for w in (r.get("words") or []) if len(w) > 3 and w[3] is not None], float)
    return {f"{pre}_logprob": np.nansum(lp * d) / d.sum(), f"{pre}_nospeech": np.nanmean(ns), f"{pre}_fallback": fb.mean(),
            f"{pre}_wprob": wp.mean() if len(wp) else np.nan, f"{pre}_wlow": (wp < 0.5).mean() if len(wp) else np.nan}
rows = []
for k in W:
    f = {"split": k[0], "filename": k[1]}
    f.update(seg_feats(W[k], "w")); f.update(seg_feats(V.get(k, {}), "v"))
    cs = np.array([w[3] for w in (C.get(k, {}).get("words") or [])], float)
    f["c_score"] = cs.mean() if len(cs) else np.nan; f["c_low"] = (cs < -1.0).mean() if len(cs) else np.nan
    rows.append(f)
R = pd.DataFrame(rows).set_index(["split", "filename"])
R.to_csv("cache/asr_reliability.csv")
print(R.describe().T[["mean", "std", "min", "max"]].round(3).to_string())
