# 7. Strategy (as of 2026-10-16)

## Where we stand
- Public LB: baseline 0.350 → v1 0.427 → v2 0.429 → v3 pending (simulated 0.59–0.66).
- Leader: 0.584. Deadline 2027-01-13 (≈ 13 weeks). Spend so far: $0.

## The plan, in order
The order puts cheap, high-information steps first, and gates anything paid on a cheap check.

1. **Read v3's score.** Logs from the hidden re-run aren't visible to us, so the score itself is the evidence:
   near 0.43 means the test set doesn't share train's prompt structure; 0.55+ means it does, and the
   assignment trick is the main lever. Everything below depends on it.
2. **Strengthen the assignment** (free) if v3 confirms the structure: soft co-membership features,
   pseudo-label self-training on the test set, and a more robust clustering threshold (validated on train subsets).
3. **Improve the base classifier** (free):
   - POS-n-gram features (McGovern et al.); check spaCy works offline on Kaggle.
   - LoRA-tune a Qwen3 embedding classifier (the Idiosyncrasies recipe) instead of a frozen probe; fits on the
     3070 Ti for 0.6B/1.7B and on Kaggle's T4 for 4B.
   - Qwen3-8B-Base, truncated, as a stronger frozen reader.
4. **More training data** (paid; the literature's biggest lever): generate passages from the 18 models via their
   APIs with prompts in the same style. Price a pilot model-by-model first and keep within $20/week. Before
   generating, check whether privately generated data counts as allowed external data: the rules allow
   "freely & publicly available external data", so this may require publishing it on Kaggle (ask on the forum).
5. **Final picks:** one best-CV submission and one robust/diverse one (e.g. with and without assignment).

## Things we won't do
- Probe the leaderboard to infer test composition (wastes submissions, overfits public LB).
- Make the GitHub repo public (rules forbid public sharing outside Kaggle).
- Rank small changes by public LB.
