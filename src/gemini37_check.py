"""Are some pool "gemini-3.7-flash" passages really 3.8? Since 2026-10-08 Google routes 3.7 requests to 3.8, and a
train-only SVC reads drjcfmd's (setting "new") and prvsiyan's pool 3.7 passages as 3.8 at least as often as 3.7.

Probe: the public 4-view SVC on train folds + pool (as in pool_source_check.py), with only those sources' 3.7 passages
removed. Paired bootstrap over prompts against the full pool, plus the 3.7 / 3.8 recalls.

Usage: python src/gemini37_check.py   (about 15 min; run memory-capped)
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
SUSPECT = {"-37:drjcfmd": ["drjcfmd"], "-37:prvsiyan": ["prvsiyan"], "-37:both": ["drjcfmd", "prvsiyan"]}


def main():
    tr = pd.read_csv(ROOT / "data/train.csv")
    pool = pd.read_parquet(ROOT / "external/nishant_pool/pool_v4.parquet")
    labels = sorted(tr.model.unique())
    y = tr.model.map({l: i for i, l in enumerate(labels)}).values
    yp = pool.model.map({l: i for i, l in enumerate(labels)}).values
    V = view_matrices(tr.text.astype(str).tolist() + pool.text.tolist())
    X = hstack(list(V.values()), format="csr")
    n = len(tr)
    Xt, Xp = X[:n], X[n:]
    del V, X
    pool = pool.drop(columns=["text"])
    gc.collect()
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(tr, y, groups=tr.prompt_id))
    is37 = (pool.model == "gemini-3.7-flash").values
    masks = {nm: ~(is37 & pool.source.isin(s).values) for nm, s in SUSPECT.items()}

    def run(nm, k):
        a, b = folds[k]
        m = masks[nm]
        svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced")
        svc.fit(vstack([Xt[a], Xp[m]]), np.r_[y[a], yp[m]])
        return softmax(svc.decision_function(Xt[b]))

    jobs = [(nm, k) for nm in masks for k in range(5)]
    res = Parallel(n_jobs=2)(delayed(run)(nm, k) for nm, k in jobs)
    P = {"full": np.load(ROOT / "submissions/oof_poolsrc_full.npy")} | {nm: np.zeros((n, 18)) for nm in masks}
    for (nm, k), p in zip(jobs, res):
        P[nm][folds[k][1]] = p

    rng = np.random.default_rng(0)
    pids = tr.prompt_id.values
    up = np.unique(pids)
    rows_of = {p: np.where(pids == p)[0] for p in up}
    boots = [np.concatenate([rows_of[p] for p in rng.choice(up, len(up))]) for _ in range(1000)]
    base_pred = P["full"].argmax(1)
    base = balanced_accuracy_score(y, base_pred)
    i37, i38 = labels.index("gemini-3.7-flash"), labels.index("gemini-3.8-flash")
    print(f"{'config':14s} {'removed':>7s} {'bal_acc':>8s} {'diff':>8s} {'boot sd':>8s} {'rec3.7':>7s} {'rec3.8':>7s}")
    for nm, Pv in P.items():
        pred = Pv.argmax(1)
        acc = balanced_accuracy_score(y, pred)
        d = [balanced_accuracy_score(y[i], pred[i]) - balanced_accuracy_score(y[i], base_pred[i]) for i in boots]
        rec = recall_score(y, pred, labels=[i37, i38], average=None)
        removed = 0 if nm == "full" else int((~masks[nm]).sum())
        print(f"{nm:14s} {removed:7d} {acc:8.4f} {acc - base:+8.4f} {np.std(d):8.4f} {rec[0]:7.3f} {rec[1]:7.3f}", flush=True)
        if nm != "full":
            np.save(ROOT / f"submissions/oof_g37_{nm[4:]}.npy", Pv)


if __name__ == "__main__":
    main()
