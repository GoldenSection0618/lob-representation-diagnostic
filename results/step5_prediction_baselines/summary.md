# Step 5 Prediction-Only Baseline Summary

## Split Sizes
- train: 5600
- val: 1200
- test: 1002

## Best Test Model
- best model by test macro_f1 tie-broken by mcc then log_loss: `logistic_regression`
- test macro_f1=0.333802, balanced_accuracy=0.350423, mcc=0.025026, log_loss=9.248650

## Majority Baseline Comparison (Test)
- logistic_regression: macro_f1 beat majority, balanced_accuracy beat majority, mcc beat majority
- mlp: macro_f1 beat majority, balanced_accuracy beat majority, mcc beat majority

## Neutral Collapse Check
- majority: neutral prediction ratio=1.0000; collapse risk
- logistic_regression: neutral prediction ratio=0.6986; no severe neutral collapse
- mlp: neutral prediction ratio=0.8892; no severe neutral collapse

## Protocol Scope
- Step 5 does not use reconstruction models.
- Step 5 does not use randomized split protocols.
- Step 5 does not use plain non-purged chronological split.
