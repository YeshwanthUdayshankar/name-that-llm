# 2. Validation and the leaderboard

**In one sentence:** split train by prompt so every validation passage answers an unseen prompt — that local
score tracks the public leaderboard within about ±0.02, which is about the leaderboard's own noise.

## Why split by prompt
Test prompts never appear in train. If a random split put one model's answer to prompt *p* in training and
another model's answer to *p* in validation, the classifier could lean on topic words and we'd overrate it.
So every experiment uses `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)` grouped by
`prompt_id`, stratified by model. All scripts use the same folds, so their out-of-fold (OOF) predictions can
be blended.

## How well local scores predict the leaderboard
| submission | local (5-fold OOF) | public LB |
|---|---|---|
| TF-IDF + LinearSVC baseline | 0.334 | 0.350 |
| v1: TF-IDF + Qwen3 0.6B + 1.7B blend | 0.440 | 0.427 |
| v2: TF-IDF + Qwen3 0.6B + 4B blend | 0.458 | 0.429 |
| v3: v2 + prompt-cluster assignment | 0.660 (simulated, 18 models/prompt) | 0.439 |

The direction is reliable; small differences are not. v2's +0.018 local gain over v1 showed up as +0.002.

## Two reasons small gains don't show
1. **Leaderboard noise.** The public LB scores only part of ~4,000 passages, with a different per-model mix
   from the private LB. With ~18 classes and a few hundred passages per class, per-class recall wobbles by
   several points, so 1–2 point differences between submissions are within noise.
2. **Optimistic blend weights.** We pick blend weights by grid search on the same OOF predictions we report.
   That's a small "winner's curse". The fair comparison is the equal-weight blend: v2 0.451 vs v1 0.428.

## Practical rules
- Trust local differences of ~0.03+; treat smaller ones as ties unless they hold across folds.
- Report the equal-weight blend next to the tuned one.
- Use the leaderboard to confirm big changes, not to rank small ones (5 submissions/day go fast).
