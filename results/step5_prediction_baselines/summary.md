# Step 5 Prediction-Only Baseline Summary

## Split Sizes
- train: 5600
- val: 1200
- test: 1002

## Best Test Model
- best model by test macro_f1 tie-broken by mcc then log_loss: `logistic_regression`
- test macro_f1=0.333802, balanced_accuracy=0.350423, mcc=0.025026, log_loss=9.248650

## Probability Quality Warning
- logistic_regression has the best test macro-F1, but its test log_loss is high (9.248650). This suggests poor probability calibration or overconfident errors.
- It should not be treated as the best calibrated predictor.

## Majority Baseline Comparison (Test)
- logistic_regression: macro_f1 beat majority, balanced_accuracy beat majority, mcc beat majority
- mlp: macro_f1 beat majority, balanced_accuracy beat majority, mcc beat majority

## Class-Coverage and Collapse Check
- majority: neutral prediction ratio=1.0000; predicted distribution: down=0, neutral=1002, up=0; neutral collapse risk; missing predicted classes: down, up; low-coverage classes(<2%): down, up; directional collapse risk
- logistic_regression: neutral prediction ratio=0.6986; predicted distribution: down=199, neutral=700, up=103; no severe class-collapse signal
- mlp: neutral prediction ratio=0.8892; predicted distribution: down=111, neutral=891, up=0; missing predicted classes: up; low-coverage classes(<2%): up; directional collapse risk

## Protocol Scope
- Step 5 does not use reconstruction models.
- Step 5 does not use randomized split protocols.
- Step 5 does not use plain non-purged chronological split.
- Per-sample prediction outputs are saved for Step 7 alignment in `per_sample_predictions.csv`.
