"""Style-matched external sets: do they help? TF-IDF LR, official prompt-grouped folds.

Sets: claude5k (seyitkaangunes/ntl-extra-api-passages, 5 Claude labels),
      kaggle720 (prvsiyan/ntl-own-prompt-generations-v1, 12 labels), and both together.
For each set: (1) transfer = train on the set only, score official passages of the labels it covers
(accuracy among those labels; chance = 1/k); (2) mixed = official fold-train + set (weight w), scored on all
18 labels of the official held-out fold.

Usage: python src/ext_check2.py [claude5k|kaggle720|both|own360|own+kaggle720|all|own ...] [--balance]
       [--transfer-only]   (own = external/own_gen.csv, pilot + daily runs from bench/daily.py)
"""
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
from ext_check import clean  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "external"
BALANCE = "--balance" in sys.argv
TRANSFER_ONLY = "--transfer-only" in sys.argv
SETS = [a for a in sys.argv[1:] if not a.startswith("--")] or ["claude5k", "kaggle720", "both"]


def load_sets():
    a = pd.read_csv(E / "ntl-extra-api-passages/ntl_extra_passages_claude.csv")[["text", "model"]]
    b = pd.read_parquet(E / "ntl-own-prompt-generations-v1/generated_v1.parquet").rename(columns={"label": "model"})
    b = b[["text", "model"]]
    own = pd.read_csv(E / "own_gen_pilot.csv")[["text", "model"]]  # our Benchmarks pilot (bench/)
    sets = {"claude5k": a, "kaggle720": b, "both": pd.concat([a, b], ignore_index=True),
            "own360": own, "own+kaggle720": pd.concat([own, b], ignore_index=True),
            "all": pd.concat([a, b, own], ignore_index=True)}
    if (E / "own_gen.csv").exists():
        sets["own"] = pd.read_csv(E / "own_gen.csv")[["text", "model"]]
    return {k: v.assign(text=v.text.map(clean)).reset_index(drop=True) for k, v in sets.items()}


def fit_predict(X, y, sw, Xb):
    f = Features()
    clf = LogisticRegression(C=20, max_iter=3000).fit(f.fit_transform(X), y, sample_weight=sw)
    return clf.predict_proba(f.transform(Xb)), list(clf.classes_)


def mixed(a, b, tr, ext, w):
    cnt = ext.model.value_counts()
    ew = (1.0 / cnt[ext.model].values) * (len(a) / 18) * w  # balance labels inside the set; w scales vs official
    X = np.concatenate([tr.text.values[a], ext.text.values])
    y = np.concatenate([tr.model.values[a], ext.model.values])
    sw = np.concatenate([np.ones(len(a)), ew])
    if BALANCE:  # equal total weight per label, so labels with extra data don't pull predictions toward them
        tot = pd.Series(sw).groupby(y).sum()
        sw = sw / tot[y].values * tot.mean()
    return (b, *fit_predict(X, y, sw, tr.text.values[b]))


def main():
    tr = pd.read_csv(ROOT / "data/train.csv")
    y = tr.model.values
    labels = sorted(set(y))
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(tr, y, groups=tr.prompt_id))
    sets = load_sets()
    for name, ext in sets.items():
        if name not in SETS:
            continue
        cov = sorted(ext.model.unique())
        m = np.isin(y, cov)
        p, cls = fit_predict(ext.text.values, ext.model.values, np.ones(len(ext)), tr.text.values[m])
        acc = balanced_accuracy_score(y[m], np.array(cls)[p.argmax(1)])
        print(f"{name}: {len(ext)} passages, {len(cov)} labels | transfer (train on set only, official "
              f"{m.sum()} passages of covered labels): {acc:.3f} (chance {1/len(cov):.3f})", flush=True)
    if TRANSFER_ONLY:
        return

    jobs = [(name, w, k) for name in SETS for w in (0.25, 0.5, 1.0) for k in range(5)]
    res = Parallel(n_jobs=3)(delayed(mixed)(folds[k][0], folds[k][1], tr, sets[name], w) for name, w, k in jobs)
    base = np.load(ROOT / "submissions/oof_tfidf_lr.npy")
    print(f"\nofficial only (TF-IDF LR): {balanced_accuracy_score(y, np.array(labels)[base.argmax(1)]):.4f}")
    for name in SETS:
        for w in (0.25, 0.5, 1.0):
            oof = np.zeros((len(tr), 18))
            for (n, ww, k), (b, p, cls) in zip(jobs, res):
                if n == name and ww == w:
                    oof[np.ix_(b, [labels.index(c) for c in cls])] = p
            pred = np.array(labels)[oof.argmax(1)]
            cov = sorted(sets[name].model.unique())
            mc = np.isin(y, cov)
            rec_cov = np.mean([np.mean(pred[y == c] == c) for c in cov])
            rec_base = np.mean([np.mean(np.array(labels)[base.argmax(1)][y == c] == c) for c in cov])
            print(f"+{name:9s} w={w:<4}: bal_acc={balanced_accuracy_score(y, pred):.4f}  "
                  f"covered-label recall {rec_base:.3f} -> {rec_cov:.3f}", flush=True)
            np.save(ROOT / f"submissions/oof_ext2_tfidf_{name}_{w}{'_bal' if BALANCE else ''}.npy", oof)


if __name__ == "__main__":
    main()
