# Step 7 Reconstruction-Prediction Alignment Summary

## Scope
- Step 7 uses the stride-4 boundary-purged chronological baseline protocol, later named `chronological_purged` in Step 10.
- Step 7 does not run random split or no-purge split ablations.
- Reconstruction encoders are frozen; Step 7 trains only logistic heads on saved latent arrays and one matched flattened raw-window control.
- Evidence is limited to one symbol (`sz000001`), one horizon (`trend5`), and one subset.

## Join Contract
- status: `passed`
- expected joined rows: `70560`
- actual joined rows: `70560`
- duplicate key count: `0`

## Sample-Level Difficulty Alignment
- Primary sample-analysis prediction model: `logistic_regression`.
- Diagnostic-outcome associations are descriptive and computed across sample-level reconstruction diagnostics.
- The heatmap uses AUROC where failure is binary; for low `proba_true`, it uses absolute Spearman association.

## Frozen Latent Transfer
- Best fixed Step 5 raw-window baseline by test macro-F1: `logistic_regression` (`0.397216`).
- Matched raw-window logistic head by test macro-F1: `raw_window_logistic_tuned` (`0.390383`, selected C=`0.1`).
- Best frozen latent head by test macro-F1: `last_snapshot_repeat@40` (`0.435540`).
- Frozen latent head beat fixed Step 5 raw-window baseline: `True`.
- Frozen latent head beat matched tuned raw-window head: `True`.

## Reconstruction-Prediction Rank Alignment
- Best test reconstruction normalized_mse variant: `pca@128`.
- Best test frozen-head macro-F1 variant: `last_snapshot_repeat@40`.
- Same variant: `False`.
- Spearman(test_recon_normalized_mse, test_pred_macro_f1): `-0.200000`.

## Failure Diagnostics
- Highest mean AUROC for incorrect prediction under `logistic_regression`: `spread_mae` (`0.520404`).
- Compare `normalized_mse`, `top_of_book_mse`, and `last_step_mse` in the quantile curve before treating overall reconstruction MSE as a downstream proxy.

## Limits
- This is one-symbol, one-horizon, one-subset evidence.
- Rank correlations use only nine frozen latent variants and should not be read as significance tests.
- Step 7 does not claim cross-symbol, cross-horizon, or trading-PnL generality.
