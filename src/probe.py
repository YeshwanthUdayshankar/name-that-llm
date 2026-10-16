"""Linear probe on frozen-LLM features, same prompt-grouped folds as baseline.py.

Usage: python src/probe.py feats/Qwen3-0.6B-Base_train.npz [--layers sweep|sweep4|12,16,20] [--C 1.0]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def cv_probs(X, tr, C):
    y = tr.model.values
    labels = sorted(set(y))
    oof = np.zeros((len(tr), len(labels)), dtype=np.float32)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for a, b in cv.split(tr, y, groups=tr.prompt_id):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        clf.fit(X[a], y[a])
        oof[b] = clf.predict_proba(X[b])
    return oof, labels


def score(oof, labels, y):
    return balanced_accuracy_score(y, np.array(labels)[oof.argmax(1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feats")
    ap.add_argument("--layers", default="sweep4")
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--save", action="store_true", help="write OOF probs for the given --layers to submissions/")
    args = ap.parse_args()

    tr = pd.read_csv(ROOT / "data/train.csv")
    f = np.load(args.feats, allow_pickle=True)
    assert (f["ids"] == tr.id.values).all()
    H, lp = f["hidden"].astype(np.float32), f["lp"]
    y = tr.model.values

    oof, labels = cv_probs(lp, tr, 1.0)
    print(f"logprob stats only: {score(oof, labels, y):.4f}")

    if args.layers.startswith("sweep"):
        # "sweep" = every layer, "sweep4" = every 4th layer; layers run in parallel processes
        step = int(args.layers[5:] or 1)
        ls = list(range(step, H.shape[1], step))
        res = Parallel(n_jobs=min(len(ls), 8))(delayed(cv_probs)(H[:, l], tr, args.C) for l in ls)
        for l, (oof, labels) in zip(ls, res):
            print(f"layer {l:2d}: {score(oof, labels, y):.4f}")
        return
    ls = [int(x) for x in args.layers.split(",")]
    X = np.concatenate([H[:, l] for l in ls] + [lp], 1)
    oof, labels = cv_probs(X, tr, args.C)
    print(f"layers {ls} + lp, C={args.C}: {score(oof, labels, y):.4f}")
    if args.save:
        out = ROOT / f"submissions/oof_probe_{Path(args.feats).stem}_{'-'.join(map(str, ls))}.npy"
        np.save(out, oof)
        print("saved", out)


if __name__ == "__main__":
    sys.exit(main())
