# Step 8 Fairness and Robustness Summary

## Scope
- Step 8 only adds fairness and robustness controls.
- It does not modify split, data construction, or reconstruction encoders.
- Latent artifact path reproducibility was already fixed before Step 8.
- No random split, no no-purge split, no new reconstruction models, and no multi-symbol or multi-horizon expansion are introduced.

## Tuned Raw-Window Logistic Control
- selected C: `0.1`
- validation macro-F1 at selected C: `0.489837`
- test macro-F1 at selected C: `0.390383`
- untuned Step 5 raw logistic test macro-F1: `0.397216`
- tuned raw-window logistic beats untuned raw-window logistic: `False`
- raw-window logistic test-oracle best C: `0.01` with test macro-F1 `0.410148`; this is not selection-valid.

## Fair Transfer Comparison
- best frozen latent head test macro-F1: `0.435540`
- tuned raw-window logistic test macro-F1: `0.390383`
- delta: `0.045157`
- best frozen latent head still beats tuned raw-window logistic: `True`
- best frozen latent head also remains above the raw-window logistic test-oracle point by `0.025392` macro-F1.
- best_frozen_latent_head is selected post hoc from Step 7 test macro-F1, so the paired bootstrap comparison is descriptive rather than a fully pre-registered confirmatory test.
- The raw-window logistic grid contains a test-oracle best point at C=0.01 with test macro-F1 0.4101. This is not used as the selected tuned baseline because C is selected by validation macro-F1, but it remains below the post hoc best frozen latent head at 0.4355.

## Paired Bootstrap Delta
- best latent vs tuned raw logistic macro-F1 delta: `0.045157`
- 95% CI: `[0.008235, 0.082278]`
- fraction_delta_gt_0: `0.993000`
- Interpretation is paired on the same test samples and descriptive.

## Rank Sensitivity
- all_latent_variants: `rank_mismatch_persists`
- exclude_last_snapshot_repeat: `rank_mismatch_weakens`
- pca_only: `rank_mismatch_weakens`
- mlp_ae_only: `rank_mismatch_persists`

## Last-Snapshot Sensitivity
- last_snapshot_repeat@40 has last_step_mse=0 by construction.
- with last_snapshot best prediction: `last_snapshot_repeat@40`
- without last_snapshot best prediction: `pca@128`
- removing it changes last_step_mse Spearman from `-0.733333` to `-0.619048`.

## Final Claim Update
- The best frozen latent head remains stronger than both the untuned and tuned raw-window logistic controls on test macro-F1 in this stride-4 subset.
- Overall reconstruction MSE is not a reliable standalone downstream proxy across all variants in this one-symbol, one-horizon, stride-4 subset.
- The rank-mismatch claim weakens after excluding `last_snapshot_repeat@40`; without it, `pca@128` is both reconstruction-best and prediction-best.

## Limits
- One symbol.
- One horizon.
- One subset.
- No random/no-purge ablation.
- No multi-symbol/multi-horizon robustness.
