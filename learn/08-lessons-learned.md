# 8. Lessons learned (as of 2026-10-16)

1. **Look at how the data was made before modelling.** "Every prompt answered once by every model" was
   visible with a three-line groupby and is worth more (simulated +0.20) than all our feature work combined
   (+0.12).
2. **Validate the way the test set differs from train.** Test prompts are unseen, so split by prompt. That
   local score has tracked the leaderboard within ~0.02.
3. **With little data, freeze the big model and train a small head.** Frozen Qwen3 + logistic regression
   (0.436) crushed fine-tuning DeBERTa (0.25).
4. **Check what a layer encodes.** Early layers (style) beat late layers (meaning) for authorship.
5. **Same features, new structure ≠ new information.** The hierarchical classifier and extra pooling both
   failed for the same reason: no new signal, more parameters.
6. **Read the papers early.** One paper told us data volume (10K/class) and same-family difficulty were the
   real limits — before we spent more time on architecture.
7. **Test the code path the check run skips.** The Kaggle check run has 3 rows; anything that only triggers on
   the full test set must be tested locally on a realistic stand-in, or it fails during scoring.
8. **Be explicit about dtype.** Library defaults changed (transformers 5.x), and T4s lack bf16.
9. **Don't trust small LB differences.** v2's +0.018 local gain became +0.002 on the leaderboard.
10. **Know the sharing rules before choosing repo visibility.** Kaggle forbids private sharing outside the team
    and public sharing outside Kaggle.
