# 4. Exploiting the prompt structure

**In one sentence:** every training prompt was answered once by each model, so if we group test passages by
prompt we can assign models jointly instead of one at a time — in simulation this lifts 0.458 to 0.66.

## The observation
```
passages per prompt: {18: 149, 17: 1}
max times one model answers one prompt: 1
```
If the test set was built the same way, then among passages answering the same prompt, each model appears at
most once. Predicting each passage independently ignores that — two passages in the same group can both be
labelled `claude-opus-5`, and one of them is guaranteed wrong.

## Step 1: recover the groups (no prompt IDs in test)
Passages answering the same prompt share a topic. Cluster on content words:
`TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2)`, L2-normalise, then
`AgglomerativeClustering(metric="cosine", linkage="average", distance_threshold=0.9)`.

On train (true groups known), with k = 150 given: ARI 0.909, purity 0.921, completeness 0.991.
Qwen3-4B layer-24 embeddings were worse (ARI 0.773) — content words are the better topic signal.

## Step 2: assign within each group
For each cluster, solve a maximum-likelihood assignment with the Hungarian algorithm
(`scipy.optimize.linear_sum_assignment` on the blend's log-probabilities). If a cluster has more than 18
passages (two prompts merged), each model may be used `ceil(n/18)` times (tile the cost matrix).

## Simulation on train OOF predictions
| setting | balanced acc. |
|---|---|
| v2 blend, independent argmax | 0.458 |
| assignment within **true** prompt groups | 0.672 |
| assignment within **estimated** clusters, threshold 0.85 / 0.90 / 0.93 / 0.95 | 0.638 / **0.660** / 0.651 / 0.624 |
| 0–7 models randomly missing per prompt: argmax → assignment (threshold 0.9) | 0.455 → 0.590 |

Threshold 0.9 doesn't need the number of prompts, and stays good when groups are incomplete.

## Risks
- **The test set may be built differently.** Per-model test counts are unequal, so test prompts likely have
  subsets of models. The fewer passages per prompt, the less the constraint helps; singleton groups gain nothing.
- **Threshold calibrated on train.** Test TF-IDF is fit on ~4,000 test passages instead of 2,700; cluster sizes
  may shift. v3 prints cluster count and median size, but hidden-run logs aren't visible to us, so we can
  only judge it by the score.
- **Untested on Kaggle before scoring.** The check run has only 3 rows, so the step is skipped there
  (`len(test) >= 100`). We tested it locally on a 400-row stand-in test file.

## Ideas to push it further
- Soft version: use cluster co-membership as a feature instead of hard assignment.
- Iterate: assignment → retrain on confident test pseudo-labels → re-assign.
- Use the per-model test counts implied by the LB? No — those aren't given, and probing them wastes submissions.
