# Step 5 Prediction-Only Baseline Summary

## Split Sizes
- train: 5600
- val: 1200
- test: 1152

## Best Test Model
- best model by test macro_f1 tie-broken by mcc then log_loss: `logistic_regression`
- test macro_f1=0.397216, balanced_accuracy=0.409817, mcc=0.100677, log_loss=4.162411

## Metric Selection Policy
- primary for this PoW: macro_f1, tie-broken by mcc then log_loss
- LOBench reference metric: cross_entropy_loss/log_loss
- best_by_macro_f1: `logistic_regression`
- best_by_log_loss / best_by_cross_entropy: `majority` (test log_loss=0.898034)
- Macro-F1 is used for class-imbalance-aware directional diagnosis; log_loss is reported as the LOBench-compatible CE-style probability-quality metric.

## Probability Quality Warning
- logistic_regression has the best test macro-F1, but its test log_loss is high (4.162411). This suggests poor probability calibration or overconfident errors.
- It should not be treated as the best calibrated predictor.

## Majority Baseline Comparison (Test)
- logistic_regression: macro_f1 beat majority, balanced_accuracy beat majority, mcc beat majority
- mlp: macro_f1 beat majority, balanced_accuracy beat majority, mcc beat majority

## Class-Coverage and Collapse Check
- majority: neutral prediction ratio=1.0000; predicted distribution: down=0, neutral=1152, up=0; neutral collapse risk; missing predicted classes: down, up; low-coverage classes(<2%): down, up; directional collapse risk
- logistic_regression: neutral prediction ratio=0.5321; predicted distribution: down=292, neutral=613, up=247; no severe class-collapse signal
- mlp: neutral prediction ratio=0.2821; predicted distribution: down=522, neutral=325, up=305; no severe class-collapse signal

## Protocol Scope
- Step 5 does not use reconstruction models.
- Step 5 does not use randomized split protocols.
- Step 5 does not use plain non-purged chronological split.
- Per-sample prediction outputs are saved for Step 7 alignment in `per_sample_predictions.csv`.
- Per-sample prediction outputs cover val/test only. For Step 7, join predictions with reconstruction diagnostics on `sample_id` and `split`; treat Step 5 `model` as `prediction_model`.
- `direction_correct_non_neutral` is encoded as 1.0/0.0 for true non-neutral samples and null for neutral samples.
