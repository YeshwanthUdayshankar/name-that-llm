"""Which parts of the public pool help on host data? Probe: the public 4-view n-gram LinearSVC, train folds + pool.

Configs (each scored on the same prompt-grouped folds as every OOF file, paired against "full"):
  full              train folds + the whole pool (the public member, about 0.474)
  -src:<s>          leave one source out (sources with >= 1000 passages; smaller ones as one group "-src:small")
  -set:<s>          leave one generation setting out (default / low / new / high / orglm:* as one group / other)
  dw                pool passages weighted by host-likeness, w = p/(1-p) from a host-vs-pool classifier on
                    topic-free views (word shape + small words), fit per fold on that fold's train part only
  top50             keep only the most host-like half of the pool, by the same score
The difference to "full" gets a bootstrap over prompts (1000 resamples), so it can be read against noise.

Usage: python src/pool_source_check.py   (about 1.5 h; run memory-capped)
"""
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).parent))
from selftrain_sim import VIEWS, shape_text, small_text, softmax  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def view_matrices(texts):
    out = {}
    for view, lo, hi, mf, w in VIEWS:
        prep = shape_text if view == "shape" else small_text if view == "small" else (lambda s: s)
        an = "char" if view == "shape" else "word" if view == "small" else view
        opts = dict(analyzer=an, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2,
                    max_features=mf, dtype=np.float32)
        if an == "word":
            opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        out[view] = TfidfVectorizer(**opts).fit_transform([prep(t) for t in texts]) * w
    return out


def host_score(Dt_a, Dp):
    """p(host) for every pool passage: 3-fold over the pool, so no pool passage is scored by a model that saw it."""
    s = np.zeros(Dp.shape[0])
    for fit_i, score_i in StratifiedKFold(3, shuffle=True, random_state=0).split(np.zeros(Dp.shape[0]), np.zeros(Dp.shape[0])):
        X = vstack([Dt_a, Dp[fit_i]])
        yy = np.r_[np.ones(Dt_a.shape[0]), np.zeros(len(fit_i))]
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X, yy)
        s[score_i] = clf.predict_proba(Dp[score_i])[:, 1]
    return s


def main():
    t0 = time.time()
    tr = pd.read_csv(ROOT / "data/train.csv")
    pool = pd.read_parquet(ROOT / "external/nishant_pool/pool_v4.parquet")
    labels = sorted(tr.model.unique())
    y = tr.model.map({l: i for i, l in enumerate(labels)}).values
    yp = pool.model.map({l: i for i, l in enumerate(labels)}).values
    V = view_matrices(tr.text.astype(str).tolist() + pool.text.tolist())
    X = hstack(list(V.values()), format="csr")
    D = hstack([V["shape"], V["small"]], format="csr")   # topic-free views for the host-vs-pool classifier
    n = len(tr)
    Xt, Xp, Dt, Dp = X[:n], X[n:], D[:n], D[n:]
    del V, X, D   # the views were a second copy of every matrix; the run peaked at 18 GB with them kept
    pool = pool.drop(columns=["text"])
    gc.collect()
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(tr, y, groups=tr.prompt_id))
    print(f"features {Xt.shape[0]}+{Xp.shape[0]} x {Xt.shape[1]} ({time.time() - t0:.0f}s)", flush=True)

    big = pool.source.value_counts()
    small = set(big[big < 1000].index)
    src = pool.source.where(~pool.source.isin(small), "small")
    sett = pool.setting.where(pool.setting.isin(["default", "low", "new", "high"]),
                              np.where(pool.setting.str.startswith("orglm:"), "orglm", "other"))
    masks = {"full": np.ones(len(pool), bool)}
    for s in sorted(src.unique()):
        masks[f"-src:{s}"] = (src != s).values
    for s in sorted(sett.unique()):
        masks[f"-set:{s}"] = (sett != s).values
    print("pool groups:", src.value_counts().to_dict(), sett.value_counts().to_dict(), flush=True)

    hs = Parallel(n_jobs=2)(delayed(host_score)(Dt[a], Dp) for a, _ in folds)
    print(f"host scores done ({time.time() - t0:.0f}s); mean p(host) by source:", flush=True)
    print(pd.Series(np.mean(hs, 0)).groupby(pool.source.values).mean().sort_values().round(3).to_string(), flush=True)

    def run(name, k):
        a, b = folds[k]
        if name in ("dw", "top50"):
            p = np.clip(hs[k], 1e-3, 1 - 1e-3)
            if name == "dw":
                w = np.clip(p / (1 - p), None, None)
                w = np.clip(w / w.mean(), 0.1, 10)
                m = np.ones(len(pool), bool)
            else:
                m = p >= np.median(p)
                w = np.ones(m.sum())
        else:
            m, w = masks[name], np.ones(masks[name].sum())
        svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced")
        svc.fit(vstack([Xt[a], Xp[m]]), np.r_[y[a], yp[m]], sample_weight=np.r_[np.ones(len(a)), w if name == "dw" else np.ones(m.sum())])
        return softmax(svc.decision_function(Xt[b]))

    names = list(masks) + ["dw", "top50"]
    jobs = [(nm, k) for nm in names for k in range(5)]
    res = Parallel(n_jobs=2, verbose=0)(delayed(run)(nm, k) for nm, k in jobs)
    P = {nm: np.zeros((n, 18)) for nm in names}
    for (nm, k), p in zip(jobs, res):
        P[nm][folds[k][1]] = p
    print(f"fits done ({time.time() - t0:.0f}s)\n", flush=True)

    rng = np.random.default_rng(0)
    pids = tr.prompt_id.values
    up = np.unique(pids)
    rows_of = {p: np.where(pids == p)[0] for p in up}
    boots = [np.concatenate([rows_of[p] for p in rng.choice(up, len(up))]) for _ in range(1000)]
    base_pred = P["full"].argmax(1)
    base = balanced_accuracy_score(y, base_pred)
    print(f"{'config':22s} {'n_pool':>7s} {'bal_acc':>8s} {'diff':>8s} {'boot sd':>8s}")
    out = []
    for nm in names:
        pred = P[nm].argmax(1)
        acc = balanced_accuracy_score(y, pred)
        d = [balanced_accuracy_score(y[i], pred[i]) - balanced_accuracy_score(y[i], base_pred[i]) for i in boots]
        npool = masks[nm].sum() if nm in masks else (len(pool) if nm == "dw" else len(pool) // 2)
        out.append((nm, npool, acc, acc - base, np.std(d)))
        print(f"{nm:22s} {npool:7d} {acc:8.4f} {acc - base:+8.4f} {np.std(d):8.4f}", flush=True)
        np.save(ROOT / f"submissions/oof_poolsrc_{nm.replace(':', '_')}.npy", P[nm])
    pd.DataFrame(out, columns=["config", "n_pool", "bal_acc", "diff", "boot_sd"]).to_csv(
        ROOT / "submissions/pool_source_check.csv", index=False)


if __name__ == "__main__":
    main()
