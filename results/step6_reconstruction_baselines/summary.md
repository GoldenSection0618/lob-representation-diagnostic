# Step 6 Reconstruction Baseline Summary

## Split Sizes
- train: 5600
- val: 1200
- test: 1152

## Models Run
- selected: ['train_mean_window', 'last_snapshot_repeat', 'pca', 'mlp_ae']
- pca latent dims: [8, 16, 32, 64, 128]
- mlp_ae latent dims: [16, 32, 64]
- imbalance gate: eps_threshold=1e-06, valid_ratio_threshold=0.95
- LOBench-compatible metrics: `lobench_compatible_reconstruction_metrics.csv` (factor=1.5, alpha=0.8, reg_factor=10.0)

## Best Test Reconstruction
- best by normalized_mse: pca@128; normalized_mse=0.183824, normalized_mae=0.187050, original_mae=0.126703
- best with latent_dim<=40: pca@32; normalized_mse=0.424496

- best pca (pca@128) beat last_snapshot_repeat on test normalized_mse
- best mlp_ae (mlp_ae@64) beat last_snapshot_repeat on test normalized_mse
- best by LOBench-compatible weighted_mse_loss: pca@128; weighted_mse_loss=0.291670, all_loss=1.206065

## Error Concentration (Test)
- normalized_mse by group (best model): price=0.066371, volume=0.301277, top_of_book=0.406778, last_step=0.302393
- derived MAE (best model): midprice=0.020960, spread=0.000272, top1_volume_sum=0.294491, top5_volume_sum=0.597874, top1_volume_diff=0.354954, top5_volume_diff=0.651458
- imbalance validity (best model): top1_valid=False, top5_valid=False, top1_valid_ratio=0.0507, top5_valid_ratio=0.2568
- top1_imbalance_mae(valid-only)=null (valid ratio below threshold)
- top5_imbalance_mae(valid-only)=null (valid ratio below threshold)

## Scope Guard
- Step 6 measures reconstruction quality only.
- Step 6 does not train prediction heads.
- Step 6 does not run reconstruction-prediction alignment.
- Step 7 will use per_sample_reconstruction_errors.csv for alignment analysis.
- Imbalance metrics are reported only when non-negative volume and denominator-validity checks pass.
- When imbalance validity is weak, volume-sum and volume-difference diagnostics are preferred.
- `original_mae`/`original_rmse` are measured in Step 3 input feature space after inverse-transforming the Step 6 scaler; they are not raw exchange order-flow scale.
