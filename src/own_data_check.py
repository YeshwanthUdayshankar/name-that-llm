"""Does our own generated data (external/own_gen.csv, Kaggle Benchmarks proxy) help on top of the public pool?

Probe: the public 4-view SVC on train folds + pool (as in pool_source_check.py), with our passages added at sample
weight 1 and at a higher weight (our set is small next to the 76k pool, so weight 1 may be invisible). Paired
bootstrap over prompts against the pool alone, plus the recall of the labels we cover.

Usage: python src/own_data_check.py   (about 10 min; run memory-capped)
"""
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import hstack, vstack
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).parent))
from pool_source_check import view_matrices  # noqa: E402
from selftrain_sim import softmax  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = [1.0, 5.0]


def main():
    tr = pd.read_csv(ROOT / "data/train.csv")
    pool = pd.read_parquet(ROOT / "external/nishant_pool/pool_v4.parquet")
    own = pd.read_csv(ROOT / "external/own_gen.csv")
    labels = sorted(tr.model.unique())
    lab = {l: i for i, l in enumerate(labels)}
    y, yp, yo = tr.model.map(lab).values, pool.model.map(lab).values, own.model.map(lab).values
    V = view_matrices(tr.text.astype(str).tolist() + pool.text.tolist() + own.text.astype(str).tolist())
    X = hstack(list(V.values()), format="csr")
    n, m = len(tr), len(pool)
    Xt, Xp, Xo = X[:n], X[n:n + m], X[n + m:]
    del V, X, pool
    gc.collect()
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(tr, y, groups=tr.prompt_id))
    print(f"own: {len(own)} passages, {own.model.nunique()} labels", flush=True)

    def run(w, k):
        a, b = folds[k]
        svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced")
        svc.fit(vstack([Xt[a], Xp, Xo]), np.r_[y[a], yp, yo], sample_weight=np.r_[np.ones(len(a) + m), np.full(len(own), w)])
        return softmax(svc.decision_function(Xt[b]))

    jobs = [(w, k) for w in WEIGHTS for k in range(5)]
    res = Parallel(n_jobs=2)(delayed(run)(w, k) for w, k in jobs)
    P = {"pool": np.load(ROOT / "submissions/oof_poolsrc_full.npy")} | {f"+own w{w:g}": np.zeros((n, 18)) for w in WEIGHTS}
    for (w, k), p in zip(jobs, res):
        P[f"+own w{w:g}"][folds[k][1]] = p

    rng = np.random.default_rng(0)
    pids = tr.prompt_id.values
    up = np.unique(pids)
    rows_of = {p: np.where(pids == p)[0] for p in up}
    boots = [np.concatenate([rows_of[p] for p in rng.choice(up, len(up))]) for _ in range(1000)]
    base_pred = P["pool"].argmax(1)
    base = balanced_accuracy_score(y, base_pred)
    covered = sorted(set(yo))
    print(f"{'config':10s} {'bal_acc':>8s} {'diff':>8s} {'boot sd':>8s} {'covered rec':>12s} {'other rec':>10s}")
    for nm, Pv in P.items():
        pred = Pv.argmax(1)
        acc = balanced_accuracy_score(y, pred)
        d = [balanced_accuracy_score(y[i], pred[i]) - balanced_accuracy_score(y[i], base_pred[i]) for i in boots]
        rec = recall_score(y, pred, labels=range(18), average=None)
        rc, ro = rec[covered].mean(), np.delete(rec, covered).mean()
        print(f"{nm:10s} {acc:8.4f} {acc - base:+8.4f} {np.std(d):8.4f} {rc:12.3f} {ro:10.3f}", flush=True)
        if nm != "pool":
            np.save(ROOT / f"submissions/oof_own_{nm.split()[1]}.npy", Pv)


if __name__ == "__main__":
    main()
