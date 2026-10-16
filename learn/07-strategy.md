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
