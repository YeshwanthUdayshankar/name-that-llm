"""TF-IDF + linear baseline, cross-validated by prompt (test prompts are unseen).

Usage: python src/baseline.py [--clf lr|svc] [--folds 5]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]


def lab(m):
    return m.split("-")[0].replace("qwen3.8", "qwen")


class Features:
    """Word 1-2grams + char 1-5grams. Case and punctuation are kept: they are style signal."""

    def __init__(self):
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=False,
                                    token_pattern=r"(?u)\b\w+\b|[^\w\s]", min_df=2,
                                    sublinear_tf=True, max_features=200_000)
        self.char = TfidfVectorizer(analyzer="char", ngram_range=(1, 5), lowercase=False,
                                    min_df=2, sublinear_tf=True, max_features=400_000)

    def fit_transform(self, texts):
        return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()

    def transform(self, texts):
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()


def make_clf(name):
    if name == "svc":
        return LinearSVC(C=0.5)
    return LogisticRegression(C=20, max_iter=3000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clf", default="lr", choices=["lr", "svc"])
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    tr = pd.read_csv(ROOT / "data/train.csv")
    y = tr["model"].values
    oof = np.empty(len(tr), dtype=object)
    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=0)
    t0 = time.time()
    for k, (a, b) in enumerate(cv.split(tr, y, groups=tr["prompt_id"])):
        f = Features()
        clf = make_clf(args.clf).fit(f.fit_transform(tr.text.values[a]), y[a])
        oof[b] = clf.predict(f.transform(tr.text.values[b]))
        print(f"fold {k}: bal_acc={balanced_accuracy_score(y[b], oof[b]):.4f}  ({time.time()-t0:.0f}s)")

    print(f"\nOOF balanced accuracy ({args.clf}): {balanced_accuracy_score(y, oof):.4f}")
    print(f"OOF lab-level accuracy:            {np.mean([lab(p) == lab(t) for p, t in zip(oof, y)]):.4f}")

    labels = sorted(set(y))
    cm = confusion_matrix(y, oof, labels=labels)
    print("\nper-model recall:")
    for i, m in enumerate(labels):
        print(f"  {m:24s} {cm[i, i] / cm[i].sum():.2f}")
    off = [(cm[i, j], labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels)) if i != j]
    print("\ntop confusions (true -> predicted):")
    for n, t, p in sorted(off, reverse=True)[:12]:
        print(f"  {n:3d}  {t} -> {p}")

    out = ROOT / "submissions"
    out.mkdir(exist_ok=True)
    pd.DataFrame({"id": tr.id, "model": y, "pred": oof}).to_csv(out / f"oof_baseline_{args.clf}.csv", index=False)


if __name__ == "__main__":
    main()
