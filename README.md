# Name That LLM (Kaggle)

https://www.kaggle.com/competitions/name-that-llm — 18-way authorship attribution of short LLM passages.
Code competition: notebook ≤9h, no internet, writes `submission.parquet`. Metric: balanced accuracy.
Final deadline 2027-01-13 23:59 UTC. 5 submissions/day.

Competition data is in `data/` (gitignored — rules forbid redistribution).

## Layout
- `src/baseline.py` — TF-IDF + linear model, StratifiedGroupKFold by `prompt_id` (test prompts are unseen).
- `kaggle/baseline/` — submission script + kernel metadata (`kaggle kernels push -p kaggle/baseline`).

## Results (5-fold OOF, grouped by prompt)
| model | bal. acc | lab acc |
|---|---|---|
| word1-2 + char1-5 TF-IDF, LinearSVC | 0.334 | 0.693 |
| same, LogisticRegression | 0.338 | 0.705 |
| DeBERTa-v3-base full fine-tune (fold 0 only) | 0.25 | |
| Qwen3-0.6B-Base frozen, layers 4+20, LR probe | 0.393 | |
| Qwen3-1.7B-Base frozen, layers 4+16 | 0.407 | |
| Qwen3-4B-Base frozen (first 24 layers), layers 4+8+20 | 0.436 | |
| blend tfidf + 0.6B + 1.7B (`kaggle/qwen_blend` v1) | 0.440 | public LB 0.427 |
| blend tfidf .5 + 0.6B .1 + 4B .4 (v2) | 0.458 | public LB 0.429 |
| hierarchical lab → model (`src/hier.py`) | 0.456 | no gain |
| extra token pooling: max/std/first8/last8 (`src/pool_probe.py`) | ≤0.429 | worse than mean |
| v2 + prompt-cluster assignment (v3) | 0.660 (sim.) | pending |

### Prompt-group assignment (v3)
Every train prompt is answered exactly once by each of the 18 models. v3 clusters test passages by topic
(TF-IDF content words, agglomerative cosine, distance threshold 0.9) as a proxy for prompt, then assigns
models within each cluster with the Hungarian algorithm (each model at most ceil(n/18) times).
On train OOF: 0.458 → 0.672 with true prompt groups, 0.660 with estimated clusters,
0.590 when 0–7 models are randomly missing per prompt.

Pipeline: `src/embed.py` (features) → `src/probe.py` (per-layer CV) → `src/blend.py` (OOF blend).
