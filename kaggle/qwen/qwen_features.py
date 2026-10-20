# %% [markdown]
# # Qwen2.5-7B-Instruct features for SHL grammar scoring (PLAN_RUBRIC C3)
# Reads only the cached transcripts (asr_c.jsonl = Whisper, ctc.jsonl = CTC raw) from the attached caches dataset.
# Writes to /kaggle/working, one file per stage, so a timeout keeps the finished stages:
# - `qwen_hidden.npz`: rubric-prompt hidden states (last token + mean over transcript tokens, layers 16/20/24),
#   for the CTC text and the Whisper text. Ridge heads are fitted locally on speaker-grouped folds.
# - `qwen_judge.csv`: zero-shot rubric judge, P(next token = "1".."5") for each text. No labels used.
# - `qwen_gen.jsonl`: minimal grammar correction of the CTC text, and the self-correction counts as JSON.
# Model licence: Apache-2.0 (Hugging Face model card). Competition data stays on Kaggle.

# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("bitsandbytes") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=True)

# %%
import json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

OUT = Path("/kaggle/working")
MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = (16, 20, 24)                  # upper-middle of 28 decoder layers (hidden_states[0] = embeddings)
MAX_TEXT_TOKENS = 600
BATCH_FWD, BATCH_GEN = 8, 8
SEED = 42
RUBRIC = """1: The person's speech struggles with proper sentence structure and syntax, displaying limited control over simple grammatical structures and memorized sentence patterns.
2: The person has a limited understanding of sentence structure and syntax. Although they use simple structures, they consistently make basic sentence structure and grammatical mistakes. They might leave sentences incomplete.
3: The person demonstrates a decent grasp of sentence structure but makes errors in grammatical structure, or they show a decent grasp of grammatical structure but make errors in sentence syntax and structure.
4: The person displays a strong understanding of sentence structure and syntax. They consistently show good control of grammar. While occasional errors may occur, they are generally minor and do not lead to misunderstandings; the person can correct most of them.
5: Overall, the person showcases high grammatical accuracy and adept control of complex grammar. They use grammar accurately and effectively, seldom making noticeable mistakes. Additionally, they handle complex language structures well and correct themselves when necessary."""

assert torch.cuda.is_available(), "no CUDA device: the accelerator must be a GPU"
torch.manual_seed(SEED)


def find(name):
    for d, _, files in os.walk("/kaggle/input", followlinks=True):
        if name in files:
            return Path(d) / name
    raise FileNotFoundError(f"{name} not found under /kaggle/input: attach the caches dataset")


def load_jsonl(p):
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            out[(r["split"], r["filename"])] = r
        except json.JSONDecodeError:
            pass
    return out


W, C = load_jsonl(find("asr_c.jsonl")), load_jsonl(find("ctc.jsonl"))
KEYS = sorted(set(W) & set(C))
TEXTS = {"ctc": [C[k]["raw_text"] or "" for k in KEYS], "whisper": [W[k]["text"] or "" for k in KEYS]}
print(f"{len(KEYS)} clips, GPU {torch.cuda.get_device_name(0)}")

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map={"": 0}, torch_dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
).eval()
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
DIGITS = [tok.convert_tokens_to_ids(str(d)) for d in range(1, 6)]
assert all(isinstance(i, int) and i >= 0 for i in DIGITS), DIGITS
print("model loaded", torch.cuda.memory_allocated() / 1e9, "GB")


def chat_pieces(system, user_before, text, user_after):
    """Token ids of a chat prompt, split so the transcript's token span is known exactly."""
    marker = "⁣TRANSCRIPT⁣"
    full = tok.apply_chat_template([{"role": "system", "content": system},
                                    {"role": "user", "content": user_before + marker + user_after}],
                                   tokenize=False, add_generation_prompt=True)
    pre, post = full.split(marker)
    a = tok(pre, add_special_tokens=False).input_ids
    b = tok(text, add_special_tokens=False).input_ids[:MAX_TEXT_TOKENS]
    c = tok(post, add_special_tokens=False).input_ids
    return a + b + c, (len(a), len(a) + len(b))


def left_pad(seqs):
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), PAD, dtype=torch.long)
    att = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, n - len(s):] = torch.tensor(s)
        att[i, n - len(s):] = 1
    return ids.cuda(), att.cuda(), [n - len(s) for s in seqs]


# %% [markdown]
# ## Stage 1: rubric prompt, hidden states and zero-shot judge

# %%
SYSTEM_JUDGE = "You are an expert assessor of spoken English grammar."
BEFORE = f"Grammar rubric (score levels 1 to 5):\n{RUBRIC}\n\nAutomatic transcript of a spoken response:\n\"\"\""
AFTER = "\"\"\"\n\nRate the grammar of this response from 1 to 5 using the rubric. Answer with a single digit."

t0, judge = time.time(), {"split": [k[0] for k in KEYS], "filename": [k[1] for k in KEYS]}
hidden = {"split": np.array([k[0] for k in KEYS]), "filename": np.array([k[1] for k in KEYS])}
for name, texts in TEXTS.items():
    built = [chat_pieces(SYSTEM_JUDGE, BEFORE, t, AFTER) for t in texts]
    order = np.argsort([len(s) for s, _ in built])
    last = np.zeros((len(texts), len(LAYERS), model.config.hidden_size), np.float16)
    mean = np.zeros_like(last)
    probs = np.zeros((len(texts), 5), np.float32)
    for b0 in range(0, len(order), BATCH_FWD):
        idx = order[b0:b0 + BATCH_FWD]
        ids, att, pads = left_pad([built[i][0] for i in idx])
        with torch.inference_mode():           # base model + lm_head on the last position only (no full logits)
            out = model.model(input_ids=ids, attention_mask=att, output_hidden_states=True)
            logits = model.lm_head(out.last_hidden_state[:, -1, :]).float()
        p = torch.softmax(logits[:, DIGITS], dim=-1).cpu().numpy()
        for j, i in enumerate(idx):
            lo, hi = built[i][1]
            lo, hi = lo + pads[j], hi + pads[j]
            for li, layer in enumerate(LAYERS):
                h = out.hidden_states[layer][j]
                last[i, li] = h[-1].float().cpu().numpy()
                mean[i, li] = (h[lo:hi].float().mean(0) if hi > lo else h[-1].float()).cpu().numpy()
            probs[i] = p[j]
        del out
        if b0 % (BATCH_FWD * 25) == 0:
            print(f"  {name} {b0}/{len(order)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    hidden[f"{name}_last"], hidden[f"{name}_mean"] = last, mean
    for d in range(5):
        judge[f"{name}_p{d + 1}"] = probs[:, d]
np.savez_compressed(OUT / "qwen_hidden.npz", layers=np.array(LAYERS), **hidden)
pd.DataFrame(judge).to_csv(OUT / "qwen_judge.csv", index=False)
print(f"stage 1 done {(time.time() - t0) / 60:.1f} min")

# %% [markdown]
# ## Stages 2 and 3: minimal grammar correction and self-correction counts (CTC text)

# %%
SYSTEM_GEN = "You are a careful English grammar editor for transcripts of non-native spoken English."
CORR_BEFORE = ("This is an automatic transcript of spoken English. It has no punctuation and some words are "
               "misrecognized.\n\nTranscript:\n\"\"\"")
CORR_AFTER = ("\"\"\"\n\nRewrite it with the MINIMUM changes needed to make it grammatical. Fix only grammar: verb "
              "forms, agreement, articles, prepositions, word order and missing words. Do not fix spelling, do not "
              "replace words you cannot understand, do not paraphrase, do not add punctuation. Output only the "
              "corrected transcript.")
SC_BEFORE = CORR_BEFORE
SC_AFTER = ("\"\"\"\n\nCount the grammatical errors the speaker made. (a) corrected: errors the speaker corrected "
            "themselves, e.g. said a wrong form and then restated it correctly. (b) uncorrected: errors left "
            "uncorrected. Ignore fillers, pauses, repetitions without a change, and misrecognized words. Answer "
            "only with JSON: {\"corrected\": <int>, \"uncorrected\": <int>}")


def generate(before, after, max_new):
    built = [chat_pieces(SYSTEM_GEN, before, t, after) for t in TEXTS["ctc"]]
    order = np.argsort([len(s) for s, _ in built])
    outs, t0 = [""] * len(built), time.time()
    for b0 in range(0, len(order), BATCH_GEN):
        idx = order[b0:b0 + BATCH_GEN]
        ids, att, _ = left_pad([built[i][0] for i in idx])
        n_new = max_new(max(built[i][1][1] - built[i][1][0] for i in idx))
        with torch.inference_mode():
            g = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=n_new, do_sample=False,
                               pad_token_id=PAD)
        for j, i in enumerate(idx):
            outs[i] = tok.decode(g[j, ids.shape[1]:], skip_special_tokens=True).strip()
        if b0 % (BATCH_GEN * 25) == 0:
            print(f"  gen {b0}/{len(order)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    return outs


t0 = time.time()
corrected = generate(CORR_BEFORE, CORR_AFTER, lambda n: min(int(1.3 * n) + 24, 800))
with open(OUT / "qwen_gen.jsonl", "w", encoding="utf-8") as fh:
    for k, c in zip(KEYS, corrected):
        fh.write(json.dumps({"split": k[0], "filename": k[1], "correction": c}) + "\n")
print(f"stage 2 done {(time.time() - t0) / 60:.1f} min")

t0 = time.time()
counts = generate(SC_BEFORE, SC_AFTER, lambda n: 32)
with open(OUT / "qwen_gen.jsonl", "w", encoding="utf-8") as fh:
    for k, c, s in zip(KEYS, corrected, counts):
        fh.write(json.dumps({"split": k[0], "filename": k[1], "correction": c, "selfcorr_json": s}) + "\n")
print(f"stage 3 done {(time.time() - t0) / 60:.1f} min")
print(pd.Series([len(c.split()) for c in corrected]).describe())
print(counts[:5])
