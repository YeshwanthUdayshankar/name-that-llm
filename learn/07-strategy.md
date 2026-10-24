# 7. Strategy (as of 2026-10-23)

## Where we stand
- **Public LB 0.624**, rank 102 of 2,784. The anchor is our private fork of the public 0.624 notebook
  (`kaggle/fork624/`). Our own pipeline topped out at 0.439 (v3).
- The leader is at 0.700, #8 at about 0.65, and 31 teams are tied at 0.624. Deadline 2027-01-13 (about 12 weeks). Spend so far: $0.
- Expect a big private shake-up around the 0.60–0.64 band of near-identical forks. From here on, rank our changes by CV
  (the public recipe's OOF), not by the public LB.

## What we learned this week (details in [09-community](09-community-2026-10-23.md))
- **Data volume beats everything.** On the 35k pool, four backbones all scored 0.569. On the 84k pool the same model
  reached 0.611.
- **Pool ablation:** every source and every generation setting helps, and filtering for host-likeness hurts. So we add to
  the pool and don't clean it.
- **Self-training the n-gram student with our weak teacher: negative.**
- **Known dead ends:** Hungarian assignment, bigger backbones, seed averaging, stackers, family gates.

## The plan, in order
1. **Grow our own data (free).** Run the daily Kaggle Benchmarks generation (`bench/daily.py`, cron hourly; about
   840 passages/day over 12 labels).
   - The proxy reaches several of the thinnest pool labels: claude-opus-5 (2.3k), gemini-3.1-pro (2.4k),
     claude-opus-4-6 (3.0k).
   - Keep the data local until the host rules on generated data.
   - Gate: once we have about 3k passages, test whether adding them to the pool raises the SVC CV (paired bootstrap,
     same harness as `pool_source_check.py`). If yes, plan a LoRA stage-1 retrain, priced first.
2. **Attack the hardest pairs (free, local).** These plateau for every member: glm-5.3 vs glm-5.3-flash,
   gemini-3.7 vs 3.8-flash, and the GPT-5.6 siblings. Look for pair-specific signals (length, typography, rare tokens)
   and test a small pairwise specialist that overrides the ensemble only inside a pair. Rule: at least +0.5 points of
   CV balanced accuracy over the pool SVC + OOF blend.
3. **Paid GPU only after a cheap check passes:** retrain the two-stage LoRA on the pool plus our data, or self-train the
   LoRA heads on the strong ensemble's test predictions. State the cost first and stay within the $15/week + $5 budget.
4. **Final picks:** the best-CV version, plus the plain 0.624 fork as the robust fallback.

## Things we won't do
- Probe the LB to infer test composition, or rank small changes by the public LB.
- Make the repo or our generated data public without a host ruling.
- Rebuild the public recipe from scratch. We fork it and add an edge.

## Open questions
- **Rules on generated data:** threads 749613, 750038, 750512, 751011, 751294, 751544 and 751954 still have no answer.
  Keep a no-pool fallback in mind.
- **The 6 labels the proxy can't reach** (fable-5, gemini-3.7-flash, qwen3.8-max, glm-5.3, glm-5.3-flash, kimi-k3)
  would need OpenRouter, at about $1.4 per 100 passages per model. That needs an explicit go-ahead.

## Status at end of 2026-10-23 (resume here)
- **Cron is installed** (`17 * * * *`, `bench/daily.py`, log at `bench/out/cron.log`).
  - Day 1 produced 811 passages, bringing our own data to 1,171. Transfer score 0.380, chance 0.083.
  - Day 2 was refused at 00:05 UTC: the free quota is a rolling ~24 h window, not a UTC-midnight reset. The script now
    retries hourly, so day 2 should run around 17:17 UTC (1:17 PM ET) on 10-24.
- **Next:** at about 3k own passages (day 3), test pool + own data against the pool alone with the SVC (paired bootstrap).
- **Closed today, all negative:**
  - pool source/setting filtering
  - host-likeness weighting
  - gemini-3.7 relabelling
  - surface features for the hard pairs
