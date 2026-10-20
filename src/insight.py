"""Insight table and error analysis (PLAN_RUBRIC G1 + G2).

    python src/insight.py <run_id>     # run_id with per-seed OOF in runs/<id>/oof.csv (the best run's reproduction)

G1 -> cache/insight.csv, one row per clip (train and test; label-free columns for test):
    identity  split, filename, batch, speaker group (train) and group size
    length    duration, speech_time (sum of Whisper word durations), n_words
    topic     k-means, k = 10 fixed in advance, on L2-normalised mpnet embeddings of the Whisper text, fitted on
              train + test together (no labels); top TF-IDF words per topic in reports/phaseG/topics.csv
    ASR       asr_avg_logprob, repair_wer_nofiller
    rubric    the Phase B features + the Qwen-derived ones (interpretation)
    model     label, oof (the run's grouped, nested-blend OOF), residual = label - oof, seed_spread (std over fold seeds);
              test: the run's test prediction
G2 -> reports/phaseG/*.csv, *.png and reports/worst_clips.md (the 25 largest |residuals|).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import qwen_feats
import rubric_feats

ROOT = Path(__file__).resolve().parents[1]
CACHE, RUNS = ROOT / "cache", ROOT / "runs"
OUT = ROOT / "reports" / "phaseG"
K_TOPICS, SEED = 10, 0
LEVELS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
KEY_FEATS = ["duration", "speech_time", "n_words", "repair_wer_nofiller", "asr_avg_logprob", "mattr50",
             "llm_err_per100", "llm_rubric_expected", "cola_frac_bad", "no_punct", "seed_spread"]


def load_jsonl(p: Path) -> dict:
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[(r["split"], r["filename"])] = r
    return out


def rmse(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else np.nan


def build(run_id: str) -> pd.DataFrame:
    F = pd.read_csv(CACHE / "features.csv")
    tr = pd.read_csv(CACHE / "train.csv")
    G = pd.read_csv(CACHE / "speaker_groups.csv")
    W, C = load_jsonl(CACHE / "asr_c.jsonl"), load_jsonl(CACHE / "ctc.jsonl")
    keys = list(zip(F.split, F.filename))
    D = F[["split", "filename", "duration", "n_words", "asr_avg_logprob", "repair_wer_nofiller", "no_punct",
           "cola_frac_bad", "gpt2_nll_gap"]].copy()
    D["speech_time"] = [sum(max(0.0, float(w[2]) - float(w[1])) for w in W[k]["words"]) for k in keys]
    D = D.merge(G[["split", "filename", "group", "batch"]], on=["split", "filename"], how="left")
    sizes = D[D.split == "train"].group.value_counts()
    D["group_size"] = np.where(D.split == "train", D.group.map(sizes), np.nan)
    D.loc[D.split == "test", "group"] = np.nan

    z = np.load(CACHE / "emb_mpnet.npz")
    idx = {k: i for i, k in enumerate(zip(z["split"], z["filename"]))}
    E = z["E"][[idx[k] for k in keys]].astype(np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    D["topic"] = KMeans(K_TOPICS, n_init=10, random_state=SEED).fit_predict(E)
    texts = [W[k]["text"] for k in keys]
    tf = TfidfVectorizer(stop_words="english", min_df=3, max_df=0.5)
    X = tf.fit_transform(texts)
    vocab = np.array(tf.get_feature_names_out())
    top = {t: " ".join(vocab[np.asarray(X[D.topic.values == t].mean(0)).ravel().argsort()[::-1][:6]]) for t in range(K_TOPICS)}
    D["topic_words"] = D.topic.map(top)

    R = rubric_feats.features()
    D = D.merge(R, on=["split", "filename"], how="left")
    Q = qwen_feats.features(CACHE, set(tr[tr.label > 0].filename)).reset_index()
    D = D.merge(Q, on=["split", "filename"], how="left")

    O = pd.read_csv(RUNS / run_id / "oof.csv")
    seed_cols = [c for c in O.columns if c.startswith("oof_seed")]
    O["seed_spread"] = O[seed_cols].std(axis=1) if seed_cols else np.nan
    O = O.rename(columns={"oof": "pred"})[["filename", "label", "pred", "seed_spread"]]
    O["split"] = "train"
    S = pd.read_csv(RUNS / run_id / "submission.csv").rename(columns={"label": "pred"})
    S["split"] = "test"
    D = D.merge(pd.concat([O, S], ignore_index=True), on=["split", "filename"], how="left")
    D["residual"] = D.label - D.pred
    D["ctc_text"] = [C[k]["raw_text"] for k in keys]
    D["whisper_text"] = texts
    D.drop(columns=["ctc_text", "whisper_text"]).to_csv(CACHE / "insight.csv", index=False)
    return D


def table_rmse(df, by):
    g = df.assign(se=df.residual ** 2).groupby(by, observed=True)
    return pd.DataFrame({"n": g.size(), "rmse": np.sqrt(g.se.mean()), "mean_residual": g.residual.mean()}).round(4)


def analyse(D: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    T = D[D.split == "train"].copy()

    t1 = table_rmse(T[T.label >= 2.0], ["batch", "label"])
    t1.to_csv(OUT / "error_by_batch_level.csv")

    shares = pd.DataFrame({"train_share": T.topic.value_counts(normalize=True),
                           "test_share": D[D.split == "test"].topic.value_counts(normalize=True),
                           "test_short_share": D[(D.split == "test") & (D.batch == "short")].topic.value_counts(normalize=True),
                           "train_short_share": T[T.batch == "short"].topic.value_counts(normalize=True)}).fillna(0).round(3)
    shares["words"] = shares.index.map(D.drop_duplicates("topic").set_index("topic").topic_words)
    t2 = table_rmse(T, ["topic", "batch"]).unstack("batch")
    t2.columns = [f"{a}_{b}" for a, b in t2.columns]
    t2 = shares.join(t2)
    t2.to_csv(OUT / "error_by_topic.csv")

    T["asr_disagreement_q"] = pd.qcut(T.repair_wer_nofiller, 5, labels=[f"Q{i}" for i in range(1, 6)])
    t3 = table_rmse(T, ["asr_disagreement_q"])
    t3["repair_wer_range"] = T.groupby("asr_disagreement_q", observed=True).repair_wer_nofiller.agg(
        lambda s: f"{s.min():.2f}-{s.max():.2f}")
    t3["mean_label"] = T.groupby("asr_disagreement_q", observed=True).label.mean().round(3)
    t3.to_csv(OUT / "error_by_asr_disagreement.csv")

    T["speech_time_q"] = T.groupby("batch").speech_time.transform(lambda s: pd.qcut(s, 5, labels=[f"Q{i}" for i in range(1, 6)]))
    t4 = table_rmse(T, ["batch", "speech_time_q"])
    t4["speech_time_range"] = T.groupby(["batch", "speech_time_q"], observed=True).speech_time.agg(
        lambda s: f"{s.min():.0f}-{s.max():.0f}s")
    t4.to_csv(OUT / "error_by_speech_time.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, b in zip(axes, ["short", "long"]):
        d = t1.loc[b] if b in t1.index.get_level_values(0) else None
        if d is not None:
            ax.bar([f"{x:g}" for x in d.index], d.mean_residual, color=["#c0392b" if v > 0 else "#2e86c1" for v in d.mean_residual])
            for x, (n, r) in enumerate(zip(d.n, d.mean_residual)):
                ax.text(x, r, f"n={n}", ha="center", va="bottom" if r > 0 else "top", fontsize=7)
        ax.axhline(0, color="k", lw=0.8); ax.set_title(f"{b} clips: mean residual (label - prediction)")
        ax.set_xlabel("true grammar score")
    fig.tight_layout(); fig.savefig(OUT / "residual_by_level.png", dpi=130); plt.close(fig)

    worst = T.reindex(T.residual.abs().sort_values(ascending=False).index).head(25)
    lines = ["# 25 largest errors (grouped OOF, best run)", "",
             "Residual = label − prediction. Positive: the model scored the speaker too low. Read each against the rubric.",
             "To listen to one (needs Amrita's approval):",
             "`kaggle competitions download shl-hiring-assessment-2026 -f Dataset_Final/train/<file> -p cache/audio`", ""]
    for i, r in enumerate(worst.itertuples(), 1):
        feats = ", ".join(f"{f} {getattr(r, f):.3g}" for f in KEY_FEATS if pd.notna(getattr(r, f, np.nan)))
        lines += [f"## {i}. {r.filename}: label {r.label:g}, predicted {r.pred:.2f} (residual {r.residual:+.2f})",
                  f"- batch **{r.batch}**, topic {r.topic} ({r.topic_words}), speaker group size {int(r.group_size)}",
                  f"- {feats}",
                  f"- Whisper: {r.whisper_text[:300]!r}",
                  f"- CTC: {r.ctc_text[:300]!r}", ""]
    (ROOT / "reports" / "worst_clips.md").write_text("\n".join(lines), encoding="utf-8")
    return t1, t2, t3, t4, worst


if __name__ == "__main__":
    D = build(sys.argv[1])
    t1, t2, t3, t4, worst = analyse(D)
    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        print("== by batch x label level\n", t1.to_string())
        print("\n== by topic\n", t2.to_string())
        print("\n== by ASR disagreement quintile\n", t3.to_string())
        print("\n== by speech-time quintile within batch\n", t4.to_string())
        print("\n== worst 25: batch", worst.batch.value_counts().to_dict(), " label", worst.label.value_counts().sort_index().to_dict(),
              " residual sign", (worst.residual > 0).map({True: "under", False: "over"}).value_counts().to_dict())
