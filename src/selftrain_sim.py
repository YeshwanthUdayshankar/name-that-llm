"""Does self-training on the unlabeled test help? A CV simulation.

Each held-out fold of the official train (prompt-grouped, the same folds as every OOF file) plays the hidden
test: about 540 unlabeled passages from unseen prompts. The student is the public notebook's 4-view n-gram
LinearSVC (char, word+punctuation, word shape, small words with content masked), which carries the largest blend
weight there. Per fold:
  V0  student on train folds only
  V1  student on train folds + the public pool (the public notebook's member as it is)
  Vq  V1 + pseudo-labeled held-out passages: the teacher's top-q most confident per predicted class
The teacher is our OOF blend (TF-IDF, Qwen 0.6B/4B probes, LoRA 0.6B; about 0.46) plus V1, Sinkhorn-balanced,
like the real ensemble but weaker (the real one is about 0.64), so the pseudo labels here are noisier than they
would be on Kaggle. We report each student alone and the blend teacher + student.

Usage: python src/selftrain_sim.py   (about 1 h; run memory-capped)
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "submissions"
QS = [0.3, 0.6, 1.0]
TEACHER_OOF = ["oof_tfidf_lr.npy", "oof_probe_Qwen3-0.6B-Base_train_4-20.npy",
               "oof_probe_Qwen3-4B-Base-L24_train_4-8-20.npy", "oof_lora_Qwen3-0.6B-Base_Lall_lr0.0002_e3.npy"]
SMALL = set("a an the and or but if while although because as so than that which who whom whose this these those it "
            "its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to "
            "from with without for of by over under between through into about is are was were be been being have has "
            "had do does did can could will would shall should may might must not no nor all any some each every both "
            "either neither other another such only also just very too more most less least here there when where why "
            "how what whether then now still yet even much many few little".split())
VIEWS = [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5), ("small", 1, 3, 50000, 0.5)]


def shape_text(t):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)


def small_text(t):
    return " ".join(w.lower() if w.lower() in SMALL or not w.isalnum() else "CONTENT" for w in re.findall(r"\w+|[^\w\s]", t))


def features(texts):
    out = []
    for view, lo, hi, mf, w in VIEWS:
        prep = shape_text if view == "shape" else small_text if view == "small" else (lambda s: s)
        an = "char" if view == "shape" else "word" if view == "small" else view
        opts = dict(analyzer=an, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2,
                    max_features=mf, dtype=np.float32)
        if an == "word":
            opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        out.append(TfidfVectorizer(**opts).fit_transform([prep(t) for t in texts]) * w)
    return hstack(out, format="csr")


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def sinkhorn(P, iters=300):
    c = np.ones(P.shape[1])
    for _ in range(iters):
        R = P * c
        R /= R.sum(1, keepdims=True)
        c *= (len(P) / P.shape[1]) / R.sum(0)
    R = P * c
    return R / R.sum(1, keepdims=True)


def fit_svc(X, y, Xb):
    svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced").fit(X, y)
    return softmax(svc.decision_function(Xb))


def pseudo(T, q):
    """Indices and labels of the top-q most confident passages per predicted class."""
    pred, conf = T.argmax(1), T.max(1)
    keep = []
    for c in np.unique(pred):
        idx = np.where(pred == c)[0]
        keep += list(idx[np.argsort(-conf[idx])][:max(1, int(round(q * len(idx))))])
    keep = np.array(sorted(keep))
    return keep, pred[keep]


def main():
    t0 = time.time()
    tr = pd.read_csv(ROOT / "data/train.csv")
    pool = pd.read_parquet(ROOT / "external/nishant_pool/pool_v4.parquet")
    labels = sorted(tr.model.unique())
    y = tr.model.map({l: i for i, l in enumerate(labels)}).values
    yp = pool.model.map({l: i for i, l in enumerate(labels)}).values
    X = features(tr.text.astype(str).tolist() + pool.text.tolist())
    Xt, Xp = X[:len(tr)], X[len(tr):]
    print(f"features {X.shape}, nnz {X.nnz / 1e6:.0f}M ({time.time() - t0:.0f}s)", flush=True)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(tr, y, groups=tr.prompt_id))
    ours = sum(np.log(np.clip(np.load(S / f), 1e-9, 1)) for f in TEACHER_OOF) / len(TEACHER_OOF)
    print(f"our blend alone: {balanced_accuracy_score(y, ours.argmax(1)):.4f}", flush=True)

    # stage 1: V0 and V1 per fold
    jobs = [(k, v) for k in range(5) for v in ("V0", "V1")]

    def stage1(k, v):
        a, b = folds[k]
        if v == "V0":
            return fit_svc(Xt[a], y[a], Xt[b])
        return fit_svc(vstack([Xt[a], Xp]), np.r_[y[a], yp], Xt[b])
    res = Parallel(n_jobs=2)(delayed(stage1)(k, v) for k, v in jobs)
    P = {v: np.zeros((len(tr), 18)) for v in ["V0", "V1"] + [f"q{q}" for q in QS]}
    for (k, v), p in zip(jobs, res):
        P[v][folds[k][1]] = p
    print(f"stage 1 done ({time.time() - t0:.0f}s)", flush=True)

    # stage 2: teacher = our blend + V1, balanced inside each fold like the real test; pseudo-label; refit
    T = np.zeros((len(tr), 18))
    for a, b in folds:
        T[b] = sinkhorn(softmax(ours[b] + np.log(np.clip(P["V1"][b], 1e-9, 1))))
    print(f"teacher (ours + V1, Sinkhorn per fold): {balanced_accuracy_score(y, T.argmax(1)):.4f}", flush=True)
    jobs2 = [(k, q) for k in range(5) for q in QS]

    def stage2(k, q):
        a, b = folds[k]
        keep, lab = pseudo(T[b], q)
        prec = np.mean(lab == y[b][keep])
        p = fit_svc(vstack([Xt[a], Xp, Xt[b][keep]]), np.r_[y[a], yp, lab], Xt[b])
        return p, prec, len(keep)
    res2 = Parallel(n_jobs=2)(delayed(stage2)(k, q) for k, q in jobs2)
    prec = {q: [] for q in QS}
    for (k, q), (p, pr, n) in zip(jobs2, res2):
        P[f"q{q}"][folds[k][1]] = p
        prec[q].append(pr)
    print(f"stage 2 done ({time.time() - t0:.0f}s)\n", flush=True)

    def ba(Pr):
        return balanced_accuracy_score(y, Pr.argmax(1))

    def with_ours(Pv):  # teacher-style blend: our models + this student, Sinkhorn per fold
        out = np.zeros_like(Pv)
        for a, b in folds:
            out[b] = sinkhorn(softmax(ours[b] + np.log(np.clip(Pv[b], 1e-9, 1))))
        return out
    print(f"{'student':10s} {'alone':>7s} {'ours+student':>13s}  pseudo precision")
    for v, Pv in P.items():
        q = float(v[1:]) if v.startswith("q") else None
        pr = f"{np.mean(prec[q]):.3f}" if q else ""
        print(f"{v:10s} {ba(Pv):7.4f} {ba(with_ours(Pv)):13.4f}  {pr}", flush=True)
        np.save(S / f"oof_selftrain_{v}.npy", Pv)


if __name__ == "__main__":
    main()
