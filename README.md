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
| blend tfidf .5 + 0.6B .1 + 4B .4 (v2) | 0.458 | |

Pipeline: `src/embed.py` (features) → `src/probe.py` (per-layer CV) → `src/blend.py` (OOF blend).
