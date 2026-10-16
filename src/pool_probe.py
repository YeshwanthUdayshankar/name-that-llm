"""Compare token-pooling variants (from embed.py --pool_layers) with the grouped-CV linear probe.

Usage: python src/pool_probe.py feats/Qwen3-4B-Base-L24-pool_train.npz
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
from probe import cv_probs, score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POOLS = ["mean", "max", "std", "first8", "last8"]
COMBOS = [["mean"], ["mean", "max"], ["mean", "std"], ["mean", "first8"], ["mean", "last8"],
          ["mean", "first8", "last8"], POOLS]


def main():
    tr = pd.read_csv(ROOT / "data/train.csv")
    f = np.load(sys.argv[1], allow_pickle=True)  # ids are stored as an object array
    assert (f["ids"] == tr.id.values).all()
    P = f["pools"].astype(np.float32)  # [N, nL, 5, D]
    y = tr.model.values

    def run(combo):
        X = np.concatenate([P[:, :, POOLS.index(c)].reshape(len(P), -1) for c in combo], 1)
        oof, labels = cv_probs(X, tr, 0.05)
        return combo, oof, labels

    for combo, oof, labels in Parallel(n_jobs=len(COMBOS))(delayed(run)(c) for c in COMBOS):
        print(f"{'+'.join(combo):32s} {score(oof, labels, y):.4f}")
        np.save(ROOT / f"submissions/oof_pool_{Path(sys.argv[1]).stem}_{'-'.join(combo)}.npy", oof)


if __name__ == "__main__":
    main()
