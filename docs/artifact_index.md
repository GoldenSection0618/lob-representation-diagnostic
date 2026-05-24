# Artifact Index

This index maps committed artifacts to the current main protocol:

- symbol: `sz000001`
- label: `trend5`
- window length: `100`
- feature dimension: `40`
- sample stride: `4`
- split: boundary-purged chronological `70/15/15`

Raw datasets, generated NumPy arrays, local latent arrays, and checkpoints are not committed.

## Primary Evidence

### Step 5: Prediction Baselines

Directory: `results/step5_prediction_baselines/`

Primary files:

- `metrics.csv`: prediction-only validation/test metrics for majority, logistic regression, and MLP baselines.
- `per_sample_predictions.csv`: val/test per-sample predictions used by Step 7 joins.
- `run_config.json`: Step 5 configuration and Step 7 join interface.
- `summary.md`: compact Step 5 result.

Figures:

- `figures/step5_prediction_baselines/primary_metrics_by_model.png`
- `figures/step5_prediction_baselines/confusion_matrix_best_model_normalized.png`
- `figures/step5_prediction_baselines/log_loss_by_model.png`
- `figures/step5_prediction_baselines/directional_error_summary.png`

### Step 6: Reconstruction Baselines

Directory: `results/step6_reconstruction_baselines/`

Primary files:

- `metrics.csv`: normalized/original-space reconstruction metrics.
- `lobench_compatible_reconstruction_metrics.csv`: LOBench-style reconstruction loss diagnostics.
- `per_sample_reconstruction_errors.csv`: per-sample reconstruction diagnostics used by Step 7.
- `rate_distortion.csv`: reconstruction quality versus compression.
- `latent_manifest.json`: local latent artifact manifest; latent arrays remain under ignored `artifacts/`.
- `run_config.json`: Step 6 protocol, metric-space notes, and latent policy.
- `summary.md`: compact Step 6 result.

Figures:

- `figures/step6_reconstruction_baselines/reconstruction_scorecard_by_model.png`
- `figures/step6_reconstruction_baselines/rate_distortion_curve.png`
- `figures/step6_reconstruction_baselines/derived_lob_error_by_model.png`
- `figures/step6_reconstruction_baselines/temporal_error_profile.png`

### Step 7: Reconstruction-Prediction Alignment

Directory: `results/step7_alignment/`

Primary files:

- `join_contract.json`: sample-level join validation.
- `sample_alignment_panel.csv`: joined prediction and reconstruction diagnostics for val/test.
- `latent_head_metrics.csv`: frozen-latent logistic head metrics.
- `latent_head_predictions.csv`: frozen-latent head predictions.
- `transfer_baseline_comparison.csv`: raw-window baselines, matched raw-window head, and frozen latent heads.
- `model_level_rank_alignment.csv`: reconstruction ranking versus frozen-head prediction ranking.
- `summary.md`: compact Step 7 result and limits.

Figures:

- `figures/step7_alignment/transfer_vs_raw_baselines.png`
- `figures/step7_alignment/latent_head_primary_metrics.png`
- `figures/step7_alignment/reconstruction_prediction_rank_alignment.png`
- `figures/step7_alignment/diagnostic_outcome_association_heatmap.png`

### Step 8: Fairness and Robustness

Directory: `results/step8_fairness_robustness/`

Primary files:

- `raw_logistic_tuning_grid.csv`: raw-window logistic C-grid metrics.
- `raw_logistic_tuned_metrics.csv`: validation-selected tuned raw-window logistic metrics.
- `raw_logistic_tuned_predictions.csv`: tuned raw-window logistic predictions.
- `fair_transfer_comparison.csv`: fair transfer comparison, including the post hoc raw logistic test-oracle reference.
- `paired_bootstrap_delta.csv`: paired test-sample bootstrap deltas.
- `rank_sensitivity.csv`: rank sensitivity with within-set and global rank fields.
- `last_snapshot_sensitivity.csv`: last-snapshot baseline sensitivity.
- `final_claim_table.csv`: C1-C7 claim status table.
- `summary.md`: compact Step 8 result and final claim update.

Figures:

- `figures/step8_fairness_robustness/fair_transfer_macro_f1_with_ci.png`
- `figures/step8_fairness_robustness/tuning_grid_val_test_curve.png`
- `figures/step8_fairness_robustness/rank_sensitivity_by_variant_set.png`
- `figures/step8_fairness_robustness/last_snapshot_sensitivity.png`

### Step 9: Validation-Selected Transfer Audit

Directory: `results/step9_validation_selection_audit/`

Primary files:

- `candidate_selection_audit.csv`: validation-ranked frozen latent candidates with test metrics retained only for final evaluation/reference.
- `fair_transfer_comparison.csv`: validation-selected latent head compared with raw-window controls and reference rows.
- `paired_bootstrap_delta.csv`: paired test-sample bootstrap deltas for the validation-selected latent head.
- `step9_manifest.json`: selection policy, selected variants, bootstrap status, and interpretation.

## Supporting Diagnostics

Supporting files include classification reports, prediction distributions, level-wise reconstruction errors, feature-group errors, temporal errors, error quantile responses, failure-mode deltas, and model-level correlations. They are retained for auditability but are not the shortest path through the core claim.

## Current Claim Boundary

The current evidence is limited to one symbol, one horizon, and one stride-4 subset. It does not claim SOTA, trading profitability, cross-symbol robustness, or cross-horizon generality.
