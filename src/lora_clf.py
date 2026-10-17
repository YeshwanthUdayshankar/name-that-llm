"""LoRA-tuned Qwen3 classifier (Idiosyncrasies recipe), same prompt-grouped folds as baseline.py.

Base model frozen in bf16 with LoRA adapters on all linear layers; mean-pool the hidden states of layer
`--layer` (default: last kept layer) and classify with a linear head in fp32.

Usage: python src/lora_clf.py --model Qwen/Qwen3-0.6B-Base [--folds 0] [--epochs 3] [--max_layers 0]
Writes OOF probs to submissions/oof_lora_<tag>.npy (rows aligned with train.csv; untouched folds are NaN).
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]


class Clf(nn.Module):
    def __init__(self, name, n_labels, max_layers, r, alpha, dropout):
        super().__init__()
        base = AutoModel.from_pretrained(name, dtype=torch.bfloat16)
        if max_layers:
            base.layers = base.layers[:max_layers]
            base.config.num_hidden_layers = max_layers
        cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        # recompute activations in backward instead of storing them: fits 8 GB, ~30% slower
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
        self.base = get_peft_model(base, cfg)
        self.head = nn.Linear(base.config.hidden_size, n_labels)  # fp32

    def forward(self, input_ids, attention_mask):
        h = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()
        m = attention_mask.unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1))


def loader(tok, texts, labels, bs, shuffle, max_len=384):
    def collate(ix):
        enc = tok([texts[i] for i in ix], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        if labels is not None:
            enc["labels"] = torch.tensor([labels[i] for i in ix])
        return enc
    return torch.utils.data.DataLoader(list(range(len(texts))), batch_size=bs, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, dl):
    model.eval()
    out = []
    for b in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(b["input_ids"].cuda(), b["attention_mask"].cuda()).float().softmax(-1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4, help="LoRA params; head uses 10x")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max_layers", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"{args.model.split('/')[-1]}_L{args.max_layers or 'all'}_lr{args.lr}_e{args.epochs}"

    tr = pd.read_csv(ROOT / "data/train.csv")
    labels = sorted(tr.model.unique())
    y = tr.model.map({m: i for i, m in enumerate(labels)}).values
    texts = tr.text.tolist()
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    out_path = ROOT / f"submissions/oof_lora_{tag}.npy"
    oof = np.load(out_path) if out_path.exists() else np.full((len(tr), len(labels)), np.nan, dtype=np.float32)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(tr, tr.model, groups=tr.prompt_id))
    for k in map(int, args.folds.split(",")):
        a, b = folds[k]
        torch.manual_seed(k)
        model = Clf(args.model, len(labels), args.max_layers, args.r, args.alpha, args.dropout).cuda()
        params = [{"params": [p for p in model.base.parameters() if p.requires_grad], "lr": args.lr},
                  {"params": model.head.parameters(), "lr": args.lr * 10}]
        opt = torch.optim.AdamW(params, weight_decay=0.001)
        tdl = loader(tok, [texts[i] for i in a], [y[i] for i in a], args.bs, True)
        vdl = loader(tok, [texts[i] for i in b], None, args.bs * 2, False)
        steps = args.epochs * len(tdl)
        sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
        t0 = time.time()
        for ep in range(args.epochs):
            model.train()
            tot = 0.0
            for batch in tdl:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(batch["input_ids"].cuda(), batch["attention_mask"].cuda())
                loss = nn.functional.cross_entropy(logits.float(), batch["labels"].cuda())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                opt.step(), sched.step(), opt.zero_grad()
                tot += loss.item()
            p = predict(model, vdl)
            print(f"fold {k} ep {ep}: train_loss={tot/len(tdl):.3f} bal_acc={balanced_accuracy_score(y[b], p.argmax(1)):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        oof[b] = p
        np.save(out_path, oof)
        del model, opt
        torch.cuda.empty_cache()

    done = ~np.isnan(oof[:, 0])
    print(f"\nOOF balanced accuracy over {done.sum()} rows: {balanced_accuracy_score(y[done], oof[done].argmax(1)):.4f}")
    # same rows, frozen-probe reference
    for ref in ["oof_probe_Qwen3-0.6B-Base_train_4-20.npy", "oof_probe_Qwen3-4B-Base-L24_train_4-8-20.npy"]:
        r = np.load(ROOT / "submissions" / ref)
        print(f"  reference {ref}: {balanced_accuracy_score(y[done], r[done].argmax(1)):.4f}")


if __name__ == "__main__":
    main()
