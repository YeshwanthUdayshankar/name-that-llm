# Name That LLM: blend of TF-IDF LR + linear probes on frozen Qwen3 Base hidden states.
# Local prompt-grouped 5-fold CV: tfidf 0.338, qwen0.6b[4,20] 0.393, qwen4b[4,8,20] 0.436, blend 0.458.
# v1 (LB pending) used 0.6b + 1.7b[4,16], CV 0.440.
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

D = Path(os.environ.get("NTL_DATA", "/kaggle/input/competitions/name-that-llm"))
# (kaggle slug, HF id, layers, blend weight, max_layers, dtype); weights tuned on local OOF.
# max_layers>0 keeps only the first N layers (no LM head, no log-prob stats) to save memory/time.
# 4B runs in fp16 because T4 has no bf16; checked locally: same CV as bf16, no overflow.
QWEN = [("0.6b-base", "Qwen/Qwen3-0.6B-Base", [4, 20], 0.1, 0, "float32"),
        ("4b-base", "Qwen/Qwen3-4B-Base", [4, 8, 20], 0.4, 24, "float16")]
W_TFIDF = 0.5

train = pd.read_csv(D / "train.csv")
test = pd.read_csv(D / "test.csv")
labels = sorted(train.model.unique())
y = train.model.values


def model_dir(slug, hf_id):
    hits = glob.glob(f"/kaggle/input/**/{slug}/**/config.json", recursive=True)
    return str(Path(sorted(hits)[0]).parent) if hits else hf_id


def tfidf_probs():
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=False,
                           token_pattern=r"(?u)\b\w+\b|[^\w\s]", min_df=2,
                           sublinear_tf=True, max_features=200_000)
    char = TfidfVectorizer(analyzer="char", ngram_range=(1, 5), lowercase=False,
                           min_df=2, sublinear_tf=True, max_features=400_000)
    X = hstack([word.fit_transform(train.text), char.fit_transform(train.text)]).tocsr()
    clf = LogisticRegression(C=20, max_iter=3000).fit(X, y)
    assert list(clf.classes_) == labels
    return clf.predict_proba(hstack([word.transform(test.text), char.transform(test.text)]).tocsr())


@torch.no_grad()
def qwen_feats(path, texts, layers, max_layers, dtype, bs=4):
    """Mean-pooled hidden states of `layers` + 6 token log-prob stats (zeros if truncated), per passage."""
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, os.environ.get("NTL_DTYPE", dtype))  # NTL_DTYPE=bfloat16 for local 8GB tests
    if max_layers:
        model = AutoModel.from_pretrained(path, dtype=dtype)
        model.layers = model.layers[:max_layers]
        model.config.num_hidden_layers = max_layers
        model = model.cuda().eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).cuda().eval()
    order = np.argsort([len(t) for t in texts])
    out = [None] * len(texts)
    for s in range(0, len(texts), bs):
        ix = order[s:s + bs]
        enc = tok([texts[i] for i in ix], return_tensors="pt", padding=True).to("cuda")
        o = model(**enc, output_hidden_states=True)
        mask = enc.attention_mask.unsqueeze(-1).float()
        pooled = [(o.hidden_states[l].float() * mask).sum(1) / mask.sum(1) for l in layers]
        if max_layers:
            for j, i in enumerate(ix):
                out[i] = np.concatenate([p[j].cpu().numpy() for p in pooled] + [np.zeros(6, dtype=np.float32)])
            continue
        m = enc.attention_mask[:, 1:].bool()
        for j, i in enumerate(ix):
            # one passage at a time: full-vocab fp32 log-probs for a whole batch don't fit in memory
            logp = o.logits[j, :-1][m[j]].float().log_softmax(-1)
            tgt = enc.input_ids[j, 1:][m[j]]
            v = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            lp = [v.mean().item(), v.std().item(), v.min().item(),
                  (logp.argmax(-1) == tgt).float().mean().item(),
                  (-(logp.exp() * logp).sum(-1)).mean().item(), len(tgt)]
            del logp
            out[i] = np.concatenate([p[j].cpu().numpy() for p in pooled] + [np.array(lp, dtype=np.float32)])
        if s % 800 == 0:
            print(f"  {s}/{len(texts)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return np.stack(out)


blend = W_TFIDF * np.log(np.clip(tfidf_probs(), 1e-6, 1))
print("tfidf done", flush=True)
texts = train.text.tolist() + test.text.tolist()
for slug, hf_id, layers, w, max_layers, dtype in QWEN:
    path = model_dir(slug, hf_id)
    print("qwen", slug, "from", path, flush=True)
    F = qwen_feats(path, texts, layers, max_layers, dtype)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)).fit(F[:len(train)], y)
    assert list(clf.classes_) == labels
    blend += w * np.log(np.clip(clf.predict_proba(F[len(train):]), 1e-6, 1))

sub = pd.DataFrame({"id": test["id"], "model": np.array(labels)[blend.argmax(1)]})
sub.to_parquet("submission.parquet", index=False)
print(sub.shape, sub.model.value_counts().head())
