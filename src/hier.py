"""Hierarchical classifier: P(model) = P(lab) * P(model | lab), same prompt-grouped folds as baseline.py.

Stage 1: lab classifier, class-weighted so every *model* gets equal weight (small labs aren't drowned out).
Stage 2: one specialist per multi-model lab, trained only on that lab's passages.
Each stage = log-prob average of a TF-IDF LR and an LR on frozen Qwen3 embeddings.

Usage: python src/hier.py [--C_emb 0.05] [--w_tfidf 0.5]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from baseline import Features, lab  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EMB = [("Qwen3-4B-Base-L24_train.npz", [4, 8, 20]), ("Qwen3-0.6B-Base_train.npz", [4, 20])]


def load_emb():
    parts = []
    for f, ls in EMB:
        H = np.load(ROOT / "feats" / f)["hidden"]
        parts += [H[:, l].astype(np.float32) for l in ls]
    return np.concatenate(parts, 1)


def fit_predict(Xa, ya, Xb, sw, kind, C):
    """Fit on (Xa, ya) and return class probs for Xb, columns in sorted(set(ya)) order."""
    if kind == "tfidf":
        clf = LogisticRegression(C=C, max_iter=3000)
    else:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    kw = {"logisticregression__sample_weight": sw} if kind == "emb" else {"sample_weight": sw}
    clf.fit(Xa, ya, **kw)
    return clf.predict_proba(Xb)


def run_fold(a, b, texts, E, y, labels, args):
    ylab = np.array([lab(m) for m in y])
    f = Features()
    Ta, Tb = f.fit_transform(texts[a]), f.transform(texts[b])
    # every model weighted equally: weight = 1 / (#train passages of that model)
    cnt = pd.Series(y[a]).value_counts()
    sw = 1.0 / cnt[y[a]].values * len(a) / len(cnt)

    def both(rows_a, ya, kind_C_tfidf=20):
        pt = fit_predict(Ta[rows_a], ya, Tb, sw[rows_a], "tfidf", kind_C_tfidf)
        pe = fit_predict(E[a][rows_a], ya, E[b], sw[rows_a], "emb", args.C_emb)
        return args.w_tfidf * np.log(np.clip(pt, 1e-6, 1)) + (1 - args.w_tfidf) * np.log(np.clip(pe, 1e-6, 1))

    all_rows = np.arange(len(a))
    labs = sorted(set(ylab))
    lab_lp = both(all_rows, ylab[a])  # [nb, n_labs]
    lab_lp -= np.logaddexp.reduce(lab_lp, 1, keepdims=True)  # renormalise blended log-probs

    out = np.full((len(b), len(labels)), -np.inf)
    for li, L in enumerate(labs):
        members = sorted(set(y[a][ylab[a] == L]))
        cols = [labels.index(m) for m in members]
        if len(members) == 1:
            out[:, cols[0]] = lab_lp[:, li]
            continue
        rows = all_rows[ylab[a] == L]
        within = both(rows, y[a][rows])
        within -= np.logaddexp.reduce(within, 1, keepdims=True)
        out[:, cols] = lab_lp[:, [li]] + within
    flat = both(all_rows, y[a])  # flat 18-way with the same recipe, for comparison
    return b, out, flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C_emb", type=float, default=0.05)
    ap.add_argument("--w_tfidf", type=float, default=0.5)
    args = ap.parse_args()

    tr = pd.read_csv(ROOT / "data/train.csv")
    y = tr.model.values
    labels = sorted(set(y))
    E = load_emb()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    res = Parallel(n_jobs=5)(delayed(run_fold)(a, b, tr.text.values, E, y, labels, args)
                             for a, b in cv.split(tr, y, groups=tr.prompt_id))
    hier = np.zeros((len(tr), len(labels)))
    flat = np.zeros((len(tr), len(labels)))
    for b, o, fl in res:
        hier[b], flat[b] = o, fl
    L = np.array(labels)
    yl = np.array([lab(m) for m in y])
    for name, P in [("flat (weighted)", flat), ("hierarchical", hier)]:
        pred = L[P.argmax(1)]
        print(f"{name:18s} bal_acc={balanced_accuracy_score(y, pred):.4f}  "
              f"lab_acc={np.mean([lab(p) == t for p, t in zip(pred, yl)]):.4f}")
    np.save(ROOT / f"submissions/oof_hier_C{args.C_emb}_w{args.w_tfidf}.npy", np.exp(hier))
    for l in sorted(set(yl)):
        m = yl == l
        pred = L[hier.argmax(1)]
        print(f"  {l:8s} lab-recall={np.mean([lab(p) == l for p in pred[m]]):.2f}  "
              f"model-recall={np.mean(pred[m] == y[m]):.2f}")


if __name__ == "__main__":
    main()
