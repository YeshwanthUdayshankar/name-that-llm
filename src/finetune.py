"""Fine-tune a sequence classifier with the same prompt-grouped folds as baseline.py.

Usage: python src/finetune.py --model microsoft/deberta-v3-base [--folds 0,1,2,3,4] [--epochs 5]
Writes OOF probabilities to submissions/oof_<tag>.npy (rows aligned with train.csv).
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]


def batches(tok, texts, labels, bs, max_len, shuffle):
    idx = list(range(len(texts)))

    def collate(ix):
        enc = tok([texts[i] for i in ix], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        if labels is not None:
            enc["labels"] = torch.tensor([labels[i] for i in ix])
        return enc

    return DataLoader(idx, batch_size=bs, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    out = []
    for b in loader:
        b = {k: v.to(dev) for k, v in b.items() if k != "labels"}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(**b).logits.float().softmax(-1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps (effective batch = bs*accum)")
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1]

    tr = pd.read_csv(ROOT / "data/train.csv")
    labels = sorted(tr.model.unique())
    y = tr.model.map({m: i for i, m in enumerate(labels)}).values
    texts = tr.text.tolist()
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)

    out_path = ROOT / f"submissions/oof_{tag}.npy"
    oof = np.load(out_path) if out_path.exists() else np.full((len(tr), len(labels)), np.nan, dtype=np.float32)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    folds = list(cv.split(tr, tr.model.values, groups=tr.prompt_id))
    for k in map(int, args.folds.split(",")):
        a, b = folds[k]
        torch.manual_seed(k)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(labels), dtype=torch.float32).to(dev)
        train_dl = batches(tok, [texts[i] for i in a], [y[i] for i in a], args.bs, args.max_len, True)
        val_dl = batches(tok, [texts[i] for i in b], None, args.bs * 2, args.max_len, False)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        steps = args.epochs * math.ceil(len(a) / (args.bs * args.accum))
        sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
        t0 = time.time()
        for ep in range(args.epochs):
            model.train()
            for i, batch in enumerate(train_dl):
                batch = {k_: v.to(dev) for k_, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**batch).loss
                (loss / args.accum).backward()
                if (i + 1) % args.accum == 0 or i + 1 == len(train_dl):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(), sched.step(), opt.zero_grad()
            p = predict(model, val_dl, dev)
            print(f"fold {k} ep {ep}: loss={loss.item():.3f} bal_acc={balanced_accuracy_score(y[b], p.argmax(1)):.4f} ({time.time()-t0:.0f}s)", flush=True)
        oof[b] = p  # last epoch, no early-stopping peeking
        np.save(out_path, oof)
        del model, opt
        torch.cuda.empty_cache()

    done = ~np.isnan(oof[:, 0])
    print(f"\nOOF balanced accuracy over {done.sum()} rows: {balanced_accuracy_score(y[done], oof[done].argmax(1)):.4f}")


if __name__ == "__main__":
    main()
