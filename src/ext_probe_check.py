"""Does external data help the frozen Qwen3-4B probe? Same design as ext_check.py, with 4B features.

Usage: python src/ext_probe_check.py
Needs feats/Qwen3-4B-Base-L24_train.npz and feats/Qwen3-4B-Base-L24_ext_clean_cap1000.npz.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
LAYERS = [4, 8, 20]


def feats(name):
    H = np.load(ROOT / "feats" / name, allow_pickle=True)["hidden"]
    return np.concatenate([H[:, l].astype(np.float32) for l in LAYERS], 1)


def run(a, b, X, y, Xe, ye, mode, w_ext):
    if mode == "official":
        Xt, yt, sw = X[a], y[a], np.ones(len(a))
    elif mode == "external_only":
        Xt, yt, sw = Xe, ye, np.ones(len(ye))
    else:
        cnt = pd.Series(ye).value_counts()
        ew = (1.0 / cnt[ye].values) * (len(a) / 18)  # balance models inside the external part
        Xt, yt = np.concatenate([X[a], Xe]), np.concatenate([y[a], ye])
        sw = np.concatenate([np.ones(len(a)), w_ext * ew])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
    clf.fit(Xt, yt, logisticregression__sample_weight=sw)
    return b, clf.predict_proba(X[b]), list(clf.classes_)


def main():
    tr = pd.read_csv(ROOT / "data/train.csv")
    ext = pd.read_csv(ROOT / "external/ext_clean_cap1000.csv")
    X, Xe = feats("Qwen3-4B-Base-L24_train.npz"), feats("Qwen3-4B-Base-L24_ext_clean_cap1000.npz")
    y, ye = tr.model.values, ext.model.values
    labels = sorted(set(y))
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(tr, y, groups=tr.prompt_id))
    for mode, w in [("official", 0), ("external_only", 0), ("mixed", 0.3), ("mixed", 1.0)]:
        res = Parallel(n_jobs=2)(delayed(run)(a, b, X, y, Xe, ye, mode, w) for a, b in folds)
        oof = np.zeros((len(tr), 18))
        for b, p, cls in res:
            oof[np.ix_(b, [labels.index(c) for c in cls])] = p
        name = mode + (f" w_ext={w}" if mode == "mixed" else "")
        print(f"{name:22s} bal_acc={balanced_accuracy_score(y, np.array(labels)[oof.argmax(1)]):.4f}", flush=True)
        if mode != "official":
            np.save(ROOT / f"submissions/oof_ext_probe4b_{mode}_{w}.npy", oof)


if __name__ == "__main__":
    main()
