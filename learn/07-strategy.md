# 7. Strategy (as of 2026-10-16)

## Where we stand
- Public LB: baseline 0.350 → v1 0.427 → v2 0.429 → v3 0.439 (prompt assignment gave +0.01, not +0.20).
- Leader: 0.584. Deadline 2027-01-13 (≈ 13 weeks). Spend so far: $0.

## The plan, in order
The order puts cheap, high-information steps first, and gates anything paid on a cheap check.

1. ~~Read v3's score.~~ Done: 0.439. Test prompts carry only ~2–5 models each, so the assignment trick is
   worth ~+0.01. Keep it in the pipeline; don't invest more ([guide 4](04-prompt-structure-assignment.md)).
2. **Improve the base classifier** (free) — now the main lever:
   - POS-n-gram features (McGovern et al.); check spaCy works offline on Kaggle.
   - LoRA-tune a Qwen3 embedding classifier (the Idiosyncrasies recipe) instead of a frozen probe; fits on the
     3070 Ti for 0.6B/1.7B and on Kaggle's T4 for 4B.
   - Qwen3-8B-Base, truncated, as a stronger frozen reader.
3. **More training data** (paid; the literature's biggest lever): generate passages from the 18 models via their
   APIs with prompts in the same style. Price a pilot model-by-model first and keep within $20/week. Before
   generating, check whether privately generated data counts as allowed external data: the rules allow
   "freely & publicly available external data", so this may require publishing it on Kaggle (ask on the forum).
4. **Self-training on test** (free): pseudo-label confident test passages and retrain — 4,000 unlabeled
   passages from unseen prompts are more data than train itself.
5. **Final picks:** one best-CV submission and one robust/diverse one (e.g. with and without assignment).

## Things we won't do
- Probe the leaderboard to infer test composition (wastes submissions, overfits public LB).
- Make the GitHub repo public (rules forbid public sharing outside Kaggle).
- Rank small changes by public LB.

## Status at end of 2026-10-16 (resume here)
- **LoRA 0.6B classifier** (`src/lora_clf.py`): 0.402 over 5 folds; blended with the rest it reaches 0.472 tuned /
  0.463 equal weights (was 0.458). Not submitted yet: it is the main candidate for v4. LoRA 1.7B was worse (0.363, fold 0).
- **External data** (`src/ext_check*.py`, TF-IDF, official folds):
  - Generic public corpus (Oleg): doesn't transfer (0.11–0.12 alone), and it hurts when mixed in. Dropped.
  - Style-matched sets transfer: claude5k 0.557 on the Claude labels, kaggle720 0.362. Mixed with balanced
    label weights: 0.360 vs 0.338.
  - **Our own free pilot** (`bench/`, Kaggle Benchmarks proxy, 30 prompts × 12 labels): transfer 0.287 (chance
    0.083). Mixed: 0.343, or 0.362 with all three generated sets. Cost about $4.50 of the $10/day free quota,
    mostly hidden reasoning (Gemini 3.1 Pro uses about 3.2k output tokens per passage).
- **Daily free run, built but not scheduled yet:**
  - `bench/topics.txt` holds 579 new topics. `bench/build_prompts.py` turns them into `bench/prompts_bank.csv`:
    8 days × 70 prompts plus 19, with target lengths matched to train (median about 101).
  - `bench/daily.py`, once per UTC day: push the day's prompts as task `ntl-gen-daily`, run the models in batches,
    download, collect into `external/own_gen.csv`, then run the transfer check (`ext_check2.py own --transfer-only`).
  - The push itself runs gemini-3.8-flash.
  - To schedule it: `17 * * * * cd /data/projects/name-that-llm && .venv/bin/python -I bench/daily.py`.
    It is safe to run hourly: a lock and a once-per-day check stop repeat runs.
- **Pending: the low-reasoning test.**
  - Task `ntl-gen-lowthink` runs the pilot prompts with `reasoning="low"` on the 3 Geminis. It was pushed
    2026-10-17 03:21 UTC; the push-triggered gemini-3.8-flash run was in progress. Still to run:
    `kaggle b t run ntl-gen-lowthink -m gemini-3.6-flash -m gemini-3.1-pro-preview --wait`.
  - Then download it and compare against the pilot outputs: style stats, and accuracy of a train-fitted
    classifier on each set.
  - If the style holds, use low reasoning for the Geminis in the daily run and raise the prompts per day.
- **Open questions:**
  - The host hasn't answered forum threads 749613 (self-generated data) and 750038 (Benchmarks quota). Keep
    generated data local and unpublished until they do.
  - The 6 labels the proxy can't reach (fable-5, gemini-3.7-flash, qwen3.8-max, glm-5.3, glm-5.3-flash, kimi-k3)
    would need OpenRouter, at about $1.4 per 100 passages per model. That needs an explicit go-ahead.
