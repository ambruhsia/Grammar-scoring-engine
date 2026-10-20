"""Delexicalised text members (Phase I): grammar without the topic.

About 40% of test answers prompts the training set barely has, and the text members lean on familiar prompt content (G11). Here
every content word is replaced by its fine-grained part-of-speech tag (spaCy en_core_web_sm, MIT), which keeps the inflection
("he go" -> "he VB", "two dog" -> "two NN"), while function words (determiners, pronouns, prepositions, auxiliaries,
conjunctions, particles) and fillers stay as words. Agreement, article, preposition, tense and fragment patterns stay visible;
topic vocabulary is gone. TF-IDF over 1-4-grams of that sequence -> RidgeCV.

Views (one member each): delex_v_pred (verbatim Whisper-large-v3-turbo text: keeps errors and fillers, spelled correctly),
delex_w_pred (Whisper clean text), delex_c_pred (CTC raw text: keeps errors, misspells content words).
Sequences are cached in cache/delex.jsonl (no labels involved).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
KEEP_POS = {"DET", "PRON", "ADP", "AUX", "CCONJ", "SCONJ", "PART", "INTJ"}
SOURCES = {"delex_v_pred": ("verbatim/asr_v.jsonl", "text"), "delex_w_pred": ("asr_c.jsonl", "text"),
           "delex_c_pred": ("ctc.jsonl", "raw_text")}


def delexicalise(doc) -> str:
    out = []
    for t in doc:
        if t.is_space or t.is_punct:
            if t.text in {".", "?", "!"}:
                out.append("EOS")
            continue
        out.append(t.lower_ if t.pos_ in KEEP_POS else t.tag_)
    return " ".join(out)


def build(cache: Path = CACHE) -> dict:
    path = cache / "delex.jsonl"
    if path.exists():
        return {(r["split"], r["filename"]): r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    out = {}
    for col, (file, field) in SOURCES.items():
        rows = [json.loads(l) for l in (cache / file).read_text(encoding="utf-8").splitlines() if l.strip()]
        keys = [(r["split"], r["filename"]) for r in rows]
        for k, doc in zip(keys, nlp.pipe([r.get(field) or "" for r in rows], batch_size=64)):
            out.setdefault(k, {"split": k[0], "filename": k[1]})[col] = delexicalise(doc)
    path.write_text("\n".join(json.dumps(r) for r in out.values()), encoding="utf-8")
    return out


def model():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 4), min_df=3, sublinear_tf=True, token_pattern=r"\S+", lowercase=False),
                         RidgeCV(alphas=np.logspace(-1, 3, 25)))


def member_inputs(ctx, which=tuple(SOURCES)) -> dict:
    """-> {col: (texts train, texts test, model)}; the texts go through TF-IDF inside the fold."""
    D = build(ctx.cache)
    return {col: (np.array([D[("train", f)][col] for f in ctx.T.filename], dtype=object),
                  np.array([D[("test", f)][col] for f in ctx.TE.filename], dtype=object), model()) for col in which}


if __name__ == "__main__":
    D = build()
    k = ("train", "audio_192.wav")
    for col in SOURCES:
        print(col, "|", D[k][col][:300])
