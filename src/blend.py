"""Blend OOF probability files (log-prob weighted average) and report balanced accuracy.

The TF-IDF logistic-regression OOF probs are computed and cached on first use.
Usage: python src/blend.py submissions/oof_probe_X.npy [more.npy ...]
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from baseline import Features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def tfidf_oof(tr):
    path = ROOT / "submissions/oof_tfidf_lr.npy"
    if path.exists():
        return np.load(path)
    y = tr.model.values
    oof = np.zeros((len(tr), 18), dtype=np.float32)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for a, b in cv.split(tr, y, groups=tr.prompt_id):
        f = Features()
        clf = LogisticRegression(C=20, max_iter=3000).fit(f.fit_transform(tr.text.values[a]), y[a])
        oof[b] = clf.predict_proba(f.transform(tr.text.values[b]))
    np.save(path, oof)
    return oof


def main():
    tr = pd.read_csv(ROOT / "data/train.csv")
    labels = np.array(sorted(tr.model.unique()))
    y = tr.model.values
    probs = [tfidf_oof(tr)] + [np.load(p) for p in sys.argv[1:]]
    names = ["tfidf"] + [Path(p).stem for p in sys.argv[1:]]
    logs = [np.log(np.clip(p, 1e-6, 1)) for p in probs]
    for n, p in zip(names, probs):
        print(f"{n:45s} {balanced_accuracy_score(y, labels[p.argmax(1)]):.4f}")

    grid = np.arange(0, 1.01, 0.1)
    best = (0, None)
    for w in product(grid, repeat=len(logs)):
        if abs(sum(w) - 1) > 1e-6:
            continue
        s = balanced_accuracy_score(y, labels[sum(wi * l for wi, l in zip(w, logs)).argmax(1)])
        best = max(best, (s, w))
    print(f"\nbest blend {dict(zip(names, np.round(best[1], 1)))}: {best[0]:.4f}")
    eq = sum(logs) / len(logs)
    print(f"equal-weight blend: {balanced_accuracy_score(y, labels[eq.argmax(1)]):.4f}")


if __name__ == "__main__":
    main()
