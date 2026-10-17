"""Does external data help? Train on (official fold-train + external), score on official held-out prompts.

External text is flattened to one paragraph and stripped of markdown symbols (official passages have
neither), and capped per model so big sources don't dominate. Uses the TF-IDF LR from baseline.py.

Usage: python src/ext_check.py [--ext external/ext_long_dedup.parquet] [--cap 1000]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from baseline import Features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def clean(t):
    t = re.sub(r"```.*?```", " ", t, flags=re.S)                  # code blocks
    t = re.sub(r"^\s{0,3}(#{1,6}\s*|[-*+]\s+|\d+[.)]\s+|>\s*)", "", t, flags=re.M)  # headings, bullets, quotes
    t = re.sub(r"\*\*|__|`", "", t)                                # bold, code ticks
    t = re.sub(r"(?<!\w)\*(?!\s)|(?<!\s)\*(?!\w)", "", t)          # italics
    return re.sub(r"\s+", " ", t).strip()


def load_ext(path, cap, seed=0):
    e = pd.read_parquet(ROOT / path)
    e["text"] = e.text.map(clean)
    n = e.text.str.split().str.len()
    e = e[(n >= 40) & (n <= 250)]
    keep = [g.sample(min(len(g), cap), random_state=seed).index for _, g in e.groupby("model")]
    e = e.loc[np.concatenate(keep)]
    return e[["text", "model"]].reset_index(drop=True)


def run(a, b, tr, ext, w_ext, mode):
    if mode == "official":
        X, y, sw = tr.text.values[a], tr.model.values[a], np.ones(len(a))
    elif mode == "external_only":
        X, y, sw = ext.text.values, ext.model.values, np.ones(len(ext))
    else:
        X = np.concatenate([tr.text.values[a], ext.text.values])
        y = np.concatenate([tr.model.values[a], ext.model.values])
        # per-model balance inside the external part, then scale it by w_ext relative to official
        cnt = ext.model.value_counts()
        ew = (1.0 / cnt[ext.model].values) * (len(a) / 18)
        sw = np.concatenate([np.ones(len(a)), w_ext * ew])
    f = Features()
    clf = LogisticRegression(C=20, max_iter=3000).fit(f.fit_transform(X), y, sample_weight=sw)
    return b, clf.predict_proba(f.transform(tr.text.values[b])), list(clf.classes_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", default="external/ext_long_dedup.parquet")
    ap.add_argument("--cap", type=int, default=1000)
    args = ap.parse_args()
    tr = pd.read_csv(ROOT / "data/train.csv")
    ext = load_ext(args.ext, args.cap)
    print("external after cleaning/cap:", len(ext), ext.model.value_counts().to_dict())
    y = tr.model.values
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(tr, y, groups=tr.prompt_id))
    configs = [("official", 0), ("external_only", 0), ("mixed", 0.3), ("mixed", 1.0), ("mixed", 3.0)]
    jobs = [(c, k) for c in configs for k in range(5)]
    # each job holds ~2-4 GB of TF-IDF features; 10 at once OOM-killed the desktop
    res = Parallel(n_jobs=3)(delayed(run)(folds[k][0], folds[k][1], tr, ext, c[1], c[0]) for c, k in jobs)
    labels = sorted(set(y))
    for c in configs:
        oof = np.zeros((len(tr), 18))
        for (cc, k), (b, p, cls) in zip(jobs, res):
            if cc == c:
                oof[b][:, :] = 0
                oof[np.ix_(b, [labels.index(x) for x in cls])] = p
        name = c[0] + (f" w_ext={c[1]}" if c[0] == "mixed" else "")
        print(f"{name:22s} bal_acc={balanced_accuracy_score(y, np.array(labels)[oof.argmax(1)]):.4f}")
        if c[0] != "official":
            np.save(ROOT / f"submissions/oof_ext_tfidf_{c[0]}_{c[1]}.npy", oof)


if __name__ == "__main__":
    main()
