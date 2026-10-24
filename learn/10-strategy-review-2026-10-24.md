# 10. Strategy review (2026-10-24, ET)

**Trigger:** weekly review, plus a big change. A new public frontier (0.636–0.639) appeared within a day, and we fell
from rank 102 to 434. Progress has also stalled: four negative local screens on 10-23/24 and a flat own-data check.

**Verdict: weakened.**
- The direction holds: data is the lever, and we fork and add an edge.
- The anchor is stale. The public frontier moved from 0.624 to 0.636–0.639 by stacking more public members, which are
  themselves trained on community-generated pools.
- Neither of our own levers (own data, hard pairs) has produced anything measurable.
- The true frontier is above any public notebook: 16 teams are above 0.659 and the top is 0.703. A plain fork gives no
  medal margin.

## Claim ledger (after the critique)
| # | Claim | Evidence | Source | Effect ± uncertainty | Confidence |
|---|---|---|---|---|---|
| C1 | Our 0.624 anchor is well behind the public frontier | LB CSV 10-24 20:29 UTC: 3,041 teams, 417 strictly above us, 293 ≥ 0.636, 171 ≥ 0.639; our rank 434 | our LB file (recomputed twice) | gap 0.012–0.015 | high |
| C2 | The public 0.636 (fleongg, "NTL Part Two Complement") = part-two main (0.620) + 15% complementary stack (12 members, incl. 4 Gaussian group heads) + 10% ATLAS on rows with top-2 margin < 0.30 + Sinkhorn | notebook code + markdown | others' code | — | medium-high |
| C3 | The 0.639's private-data helper adds anything | 0.639 vs 0.637 vs 0.636, single submissions | others' LB | +0.001–0.003, inside noise | low |
| C4 | Repeating levers in the lineage: pool size +0.028, Gaussian heads ~+0.005 (×5), complementary stack up to +0.016, ATLAS +0.002–0.007. Flat: reweighting, guards, Hungarian assignment | forum 752357 (55 notebooks, 134 scored versions) | others' data | each repeat is within LB noise; the repetition is the evidence | medium |
| C5 | Gaussian heads on the two 14B LoRAs are an untried lever | 752357 claim. But the 0.636 already has 4 Gaussian heads, and the 14B LoRAs ship without fold checkpoints, so heads fit on their train readouts can't be validated out of fold | others' claim + our check of the dataset | ≤ +0.005, unmeasurable OOF | low |
| C6 | The host has still not ruled on generated data or the Benchmarks quota | threads 752084 and 752270 (0 replies) plus 7 earlier ones | our reading | — | high (as of 10-24) |
| C7 | Removing pool data hurts the SVC: every removal is ≤ 0 and size dominates | `pool_source_check.py`: 11 configs with z ≤ −2 (−set:default z −5.0, lxd −3.4); 3 are about 0; zbigniew +0.003 (z 0.9) is what chance gives in 23 tests; top50 is confounded with size | our measurement | sd 0.003–0.008 per config | medium (SVC only) |
| C8 | Our own data is flat at 1,171 passages (tested 08:56 ET; the file now has 2,009) | `own_data_check.py`: +0.000 (126 predictions changed, net 0) at w1, −0.002 at w5. Our data has no gemini-3.7, GLM, kimi, qwen or fable rows, so it can't help the hard pairs. The vocabulary refit is a small confound | our measurement | ±0.003 | high that it is flat; nothing known about larger sizes |
| C9 | Dropping the suspect gemini-3.7 pool rows doesn't help | `gemini37_check.py`: bal acc −0.002. 3.7 recall falls 0.233 → 0.140 and 3.8 recall rises 0.733 → 0.820, which is partly mechanical (fewer 3.7 labels) | our measurement | — | medium; it says nothing about whether the labels are contaminated |
| C10 | No surface feature separates the hard pairs | 17 features, \|d\| ≤ 0.4, mostly not replicated in the pool | our measurement | d noise ±0.12 | medium |
| C11 | LB resolution | about 1,600 public rows (assumed: 40% of a reported 4,000) → sd ≈ 0.012, 2 sd ≈ 0.024; paired variants are smaller | our calculation, assumption | — | low-medium |
| C12 | Medal lines sit inside score clusters | 120 teams at exactly 0.639 (ranks 52–171) contain the silver line (rank ~152); 90 at 0.636 (ranks ~204–293) sit just above bronze (rank ~304, 0.635); gold ≈ 0.659 (rank ~16) | our LB file | will move up | medium |

## Assumptions
1. **Data volume is the main lever — holds.** Supported by C4 and C7.
2. **Fork and add an edge beats rebuilding — holds**, but the base to fork has moved (C1, C2).
3. **Our own daily data reaches a useful size — weakened.** It is flat at 1.2k, misses the hard-pair labels, and
   similar proxy sets are already public and inside the 0.636.
4. **The edge is in the hard pairs — weakened.** Our surface features failed; nobody publicly reports a lever (752197,
   0 replies).
5. **Rules risk — unchanged.** Every public notebook ≥ 0.62, ours included, trains on community API-generated data.
   The 0.636 adds more datasets of the same kind (seyit, abdullah1afify, dophuchung, lucasboesen, prvsiyan).
6. **Budget — holds.** $0 spent; Kaggle T4 quota is free.

## Critique (independent agent) and our responses
1. Medal lines sit inside tie clusters. → **Accepted**, C12 reworded.
2. The 0.636 still uses community API-generated sets, and it warns that input visibility may block forks. →
   **Accepted.** Same risk class as our current anchor. Check that all 33 inputs are public before forking.
3. The "train-only 0.637" option is invalid: the current version loads the 4,404 private passages. → **Accepted**,
   option dropped.
4. Re-anchoring rests on LB evidence only. 0.636 is the best of the author's versions (winner's curse), and a fork
   reproduces the LB by construction. → **Accepted in part.** We justify the switch by structure instead: the main is
   unchanged, the added members have small fixed weights and there are no knobs tuned to our data. We keep the 0.624 as
   the hedge pick, and check hidden-test runtime before adopting.
5. Gaussian heads are already in the 0.636. The 14B LoRAs are in-sample on train, and no fold checkpoints exist
   (confirmed). → **Accepted.** Idea demoted to low.
6. C3 lowered to low. → **Accepted.**
7. C7 count overstated. → **Accepted**, reworded as "every removal is ≤ 0; size dominates".
8. C9 conclusion too strong (the recall drop is partly mechanical). → **Accepted**, reworded.
9. C8 test was narrow (no hard-pair labels; tested at 1,171, not 2,009). → **Accepted.** That feeds plan step 3.
10. Noise understated; the 1,600 rows are an assumption. → **Accepted.**
11. The real frontier is 0.66–0.70, not 0.639. → **Accepted**, added to the verdict.
12. The own-data stop rule sits inside noise. → **Accepted**, rule changed to z ≥ 1.5.

## Updated plan
1. **Re-anchor on the 0.636 structure.** Free; uses 1 Kaggle submission, so it needs the user's OK.
   - Fork fleongg's scored v1 unchanged, after checking that all 33 inputs are public and attachable.
   - Prediction: public LB 0.630–0.640. Rule: adopt if ≥ 0.630 and the scoring run finishes. Keep the 0.624 as the hedge.
2. **Daily generation:** keep running (free). Rerun `own_data_check.py` at about 5k passages. Rule: continue only if
   z ≥ 1.5. Our data can't reach the hard-pair labels (the proxy has no GLM, gemini-3.7, kimi, qwen or fable).
3. **Gold-level levers** (ranked; each needs a cheap check first):
   - (a) Our own stage-1 + stage-2 LoRA with fold checkpoints. This gives honest OOF for everything downstream (heads,
     weights) and lets us add pool + own data. Paid or Kaggle GPU, so price it first.
   - (b) Self-training the strong ensemble's heads on test pseudo-labels.
   - (c) Gaussian heads on the 14B readouts, only after (a).
4. **Parked:** hard-pair surface features; pool filtering; LB tuning of Sinkhorn, weights or guards.
