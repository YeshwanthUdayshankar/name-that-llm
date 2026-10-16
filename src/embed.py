"""Extract frozen-LLM features for every passage in a CSV.

Per passage: mean-pooled hidden state of every layer, plus token log-prob stats
(how predictable the text is to the model).

Usage: python src/embed.py --model Qwen/Qwen3-0.6B-Base [--csv data/train.csv]
Writes feats/<tag>_<csvstem>.npz with arrays: hidden [N, L+1, D] (fp16), lp [N, 6], ids [N].
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--csv", default=str(ROOT / "data/train.csv"))
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dtype", default="bfloat16", help="float16 to mimic Kaggle T4 (no bf16)")
    ap.add_argument("--max_layers", type=int, default=0,
                    help="keep only the first N transformer layers (fits bigger models in 8GB); skips log-prob stats")
    ap.add_argument("--pool_layers", default="",
                    help="e.g. 4,8,20: also save mean/max/std/first-8/last-8 token pooling for these layers")
    args = ap.parse_args()
    pool_layers = [int(x) for x in args.pool_layers.split(",") if x]
    tag = args.tag or args.model.split("/")[-1]

    df = pd.read_csv(args.csv)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.max_layers:
        # Load on CPU, drop the upper layers and LM head, then move to GPU. Layers < N are
        # identical to the full model; layer N is returned after the final norm.
        model = AutoModel.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
        model.layers = model.layers[:args.max_layers]
        model.config.num_hidden_layers = args.max_layers
        model = model.cuda().eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).cuda().eval()

    # Sort by length so batches pad little; restore order at the end.
    order = np.argsort(df.text.str.len().values)
    hidden, lp, pools = [None] * len(df), [None] * len(df), [None] * len(df)
    t0 = time.time()
    for s in range(0, len(df), args.bs):
        ix = order[s:s + args.bs]
        enc = tok([df.text.iloc[i] for i in ix], return_tensors="pt", padding=True).to("cuda")
        out = model(**enc, output_hidden_states=True)
        mask = enc.attention_mask.unsqueeze(-1).float()
        hs = torch.stack(out.hidden_states, 1).float()  # [B, L+1, T, D]
        pooled = (hs * mask.unsqueeze(1)).sum(2) / mask.sum(1).unsqueeze(1)
        if pool_layers:
            for j, i in enumerate(ix):
                n = int(enc.attention_mask[j].sum())
                h = hs[j, pool_layers, :n]  # [nL, n, D]
                pools[i] = torch.stack([h.mean(1), h.max(1).values, h.std(1),
                                        h[:, :8].mean(1), h[:, -8:].mean(1)], 1).half().cpu().numpy()
        if args.max_layers:
            for j, i in enumerate(ix):
                hidden[i] = pooled[j].half().cpu().numpy()
                lp[i] = np.zeros(6, dtype=np.float32)
            continue
        # next-token log-probs of the actual text (positions 1..T-1)
        logp = out.logits[:, :-1].float().log_softmax(-1)
        tgt = enc.input_ids[:, 1:]
        tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        top1 = (logp.argmax(-1) == tgt).float()
        ent = -(logp.exp() * logp).sum(-1)
        m = enc.attention_mask[:, 1:].float()
        for j, i in enumerate(ix):
            n = m[j].bool()
            v = tok_lp[j][n]
            hidden[i] = pooled[j].half().cpu().numpy()
            lp[i] = np.array([v.mean().item(), v.std().item(), v.min().item(),
                              top1[j][n].mean().item(), ent[j][n].mean().item(), n.sum().item()], dtype=np.float32)
        if s % (args.bs * 40) == 0:
            print(f"{s}/{len(df)} ({time.time()-t0:.0f}s)", flush=True)

    out_dir = ROOT / "feats"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{tag}_{Path(args.csv).stem}.npz"
    extra = {"pools": np.stack(pools), "pool_layers": np.array(pool_layers)} if pool_layers else {}
    np.savez(path, hidden=np.stack(hidden), lp=np.stack(lp), ids=df.id.values, **extra)
    print(f"saved {path} hidden={np.stack(hidden).shape} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
