# 6. Engineering notes

Things that cost us time, so they won't next time.

## Running models offline on Kaggle
- Kaggle hosts official Qwen 3 checkpoints: `qwen-lm/qwen-3/transformers/{0.6b,1.7b,4b,8b,14b}-base/1`.
  Attach with `"model_sources"` in `kernel-metadata.json`; they mount under
  `/kaggle/input/models/qwen-lm/qwen-3/transformers/<variant>/1`. Our script finds them by globbing for
  `config.json`, falling back to the Hugging Face id locally.
- Submission notebooks: `"enable_internet": false`, `"enable_gpu": true`, `"machine_shape": "NvidiaTeslaT4"`.

## Precision
- **transformers 5.x loads the checkpoint's dtype by default.** DeBERTa-v3 came in as fp16 → NaN training.
  Always pass `dtype=` explicitly.
- **T4 has no bfloat16.** Use fp32 for small models, fp16 for the 4B. We checked fp16 vs bf16 4B features
  locally: no overflow, cosine similarity 0.99998, identical CV (0.436).

## Fitting big models in 8 GB (local RTX 3070 Ti)
- Truncate: load with `AutoModel` (no LM head) on CPU, keep `model.layers[:24]`, set
  `config.num_hidden_layers = 24`, then `.cuda()`. Layers below the cut are identical to the full model;
  the last returned hidden state has the final norm applied. Qwen3-4B fits this way.
- Full-vocabulary log-softmax (151K vocab) in fp32 for a whole batch runs out of memory; compute it one
  passage at a time.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces fragmentation OOMs.

## Speed
- Feature extraction for 2,699 passages: 0.6B 35 s, 4B (24 layers) 78 s locally.
- Kaggle check run: v1 (0.6B + 1.7B fp32) ~15 min; v2/v3 (0.6B + 4B-L24 fp16) ~8 min. Hidden re-run on
  ~6,700 passages (train + test) is ~2.5× that.
- Linear probes: `LogisticRegression` on 1–10K dims × 5 folds takes minutes; run layers in parallel with
  `joblib` and cap BLAS threads (`OMP_NUM_THREADS=3`).

## Kaggle CLI quirks (v2.2.4)
- `kaggle competitions pages` ignores the positional competition; use
  `KAGGLE_COMPETITION=name-that-llm kaggle competitions pages list --content`.
- Submit a notebook output: `kaggle competitions submit name-that-llm -k <user>/<kernel> -v <version> -f submission.parquet -m "..."`.
- Wait for a run: poll `kaggle kernels status`; fetch the log with `kaggle kernels output <kernel> -p <dir>`.

## Shell gotcha
`pkill -f "src/probe.py"` inside a command that also contains `src/probe.py` kills its own shell — and any
edits later in that same command never run. Kill in a separate command.
