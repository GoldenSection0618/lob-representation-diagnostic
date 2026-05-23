# Step 6 Reconstruction Baseline Summary

## Split Sizes
- train: 5600
- val: 1200
- test: 1002

## Models Run
- selected: ['train_mean_window', 'last_snapshot_repeat', 'pca', 'mlp_ae']
- pca latent dims: [8, 16, 32, 64, 128]
- mlp_ae latent dims: [16, 32, 64]

## Best Test Reconstruction
- best by normalized_mse: pca@128; normalized_mse=0.191170, normalized_mae=0.239629, original_mae=0.135957
- best with latent_dim<=40: pca@32; normalized_mse=0.292410

- best pca (pca@128) beat last_snapshot_repeat on test normalized_mse
- best mlp_ae (mlp_ae@32) beat last_snapshot_repeat on test normalized_mse

## Error Concentration (Test)
- normalized_mse by group (best model): price=0.006855, volume=0.375486, top_of_book=0.152614, last_step=0.260467
- derived MAE (best model): midprice=0.017396, spread=0.000521, top1_imbalance=2.017892

## Scope Guard
- Step 6 measures reconstruction quality only.
- Step 6 does not train prediction heads.
- Step 6 does not run reconstruction-prediction alignment.
- Step 7 will use per_sample_reconstruction_errors.csv for alignment analysis.
