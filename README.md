# LOB Representation Diagnostic

This is an independent PoW repo, not a LOBench fork and not a SimLOB/LOBench reproduction.

The question is simple: when a model reconstructs the limit order book better, does that improvement survive into downstream mid-price trend prediction once the evaluation split is leakage-aware? If it does not, I want to see where the mismatch comes from: top-of-book versus deeper levels, quiet versus stressed regimes, or the reconstruction objective itself.

I am not optimizing for a leaderboard score or a trading PnL claim. The value of this repo is a clean diagnostic trail: data contract, split policy, baseline results, and failure analysis that can be audited later.

## Current Position

The main evaluation path is locked to a boundary-purged chronological split. Train comes first, validation follows, test comes last, and overlapping sliding-window history is purged at train/validation and validation/test boundaries. In code, that policy is enforced through `chronological_split()` and `_enforce_non_overlap_boundary()`.

Random split is not part of the main experiment. A no-purge chronological split is also not part of Step 4/5/6; if I add either later, it will be labeled as an auxiliary diagnostic rather than the primary result.

Step 6 reconstruction baselines are complete on the locked Step 3 subset. Reconstruction-prediction alignment is still pending, so the repo does not yet claim transfer from reconstruction quality to downstream prediction quality.

Current data run:

- Source: `~/datasets/LOBench-A-share-processed/sz000001-level10_processed.csv`
- Output: `data/processed/minimal_subset/`
- Window: `100`
- Feature width: `40`
- Label: `trend5`
- Threshold: `0.0001`
- Split: `70/15/15`, boundary-purged chronological
- Samples: `7802`
- Shapes: `X=(7802, 100, 40)`, `y=(7802,)`
- Data checks passed: feature contract, label contract, window alignment, chronological split, output safety

Step 5 prediction-only test results:

- `majority`: `macro_f1=0.2069`, `balanced_accuracy=0.3333`, `mcc=0.0000`, `log_loss=1.2228`
- `logistic_regression`: `macro_f1=0.3338`, `balanced_accuracy=0.3504`, `mcc=0.0250`, `log_loss=9.2487`
- `mlp`: `macro_f1=0.2760`, `balanced_accuracy=0.3535`, `mcc=0.0589`, `log_loss=1.7594`

`logistic_regression` is the best test model by macro-F1, but its log loss is poor. I treat that as a useful warning: directional class separation and probability quality are not the same thing.

Artifacts from Step 5 live under:

- `results/step5_prediction_baselines/`
- `figures/step5_prediction_baselines/`

Step 6 reconstruction-only test snapshot:

- Best test normalized-MSE model: `pca@128` (`normalized_mse=0.1912`, `normalized_mae=0.2396`, `original_mae=0.1360`)
- Strong compression-constrained point (`latent_dim<=40`): `pca@32` (`normalized_mse=0.2924`)
- Both best `pca` and best `mlp_ae` variants beat `last_snapshot_repeat` on test normalized-MSE
- Error concentration for best model is much higher on volume-related dimensions than price dimensions
- In Step 6, `original_mae` / `original_rmse` mean errors in Step 3 input feature space after inverse-transforming the Step 6 train-only scaler; they are not raw exchange order-flow scale.

Artifacts from Step 6 live under:

- `results/step6_reconstruction_baselines/`
- `figures/step6_reconstruction_baselines/`

## Scope

This repo does:

- Build a small chronological subset from external LOBench-style A-share data.
- Keep a fixed 10-level, 100-step, 40-feature input contract.
- Establish prediction-only floors before representation experiments.
- Compare reconstruction quality against downstream prediction once alignment analysis is available.
- Break reconstruction error down by book level and regime.
- Track practical costs such as latency and compression only when they clarify the trade-off.

This repo does not:

- Claim SOTA.
- Fully reproduce LOBench or SimLOB.
- Evaluate real trading profitability.
- Commit or redistribute external, private, or proprietary LOB data.
- Generalize one small subset to every symbol, venue, and market period.

## Layout

- `configs/`: optional experiment configs added only when used.
- `src/data/`: external data loading, field mapping, labels, subset construction.
- `src/models/`: implemented prediction baselines and future reconstruction baselines.
- `src/analysis/`: implemented metrics and future diagnostics.
- `src/utils/`: shared utilities only when needed.
- `scripts/`: runnable stage entry points.
- `docs/`: protocol and methodology notes.
- `results/`: experiment outputs.
- `figures/`: plots and visual diagnostics.

LOBench and SimLOB remain external references. This repo stores my own scripts, configs, notes, analysis code, and memos.
