# Technical Memo

This memo is the working narrative for the experiment. I keep the framing explicit so later model results do not rewrite the question after the fact.

## Core Question

The project is not asking whether a model can make the LOB look better under reconstruction loss. It asks whether that better reconstruction transfers into mid-price trend prediction under a leakage-aware chronological evaluation.

The distinction matters. Reconstruction loss averages across the book. Trend prediction often lives closer to top-of-book behavior, short-term imbalance, spread changes, and local liquidity shocks. A model can spend capacity on distant levels or stable noise and still look good on reconstruction while doing little for the decision target.

That is the diagnostic angle of this PoW.

## Current Setup

The first controlled subset is:

- Symbol: `sz000001`
- Source: `~/datasets/LOBench-A-share-processed/sz000001-level10_processed.csv`
- Input: `(N, 100, 40)`
- Label: `trend5`
- Sample stride: `4`
- Split: boundary-purged chronological `70/15/15`
- Samples: `7952`

The split is stricter than plain chronological evaluation because overlapping historical windows are removed at train/validation and validation/test boundaries. Random split is not part of the main experiment. No-purge chronological split is left for later ablation work.

## Baseline Snapshot

Step 5 prediction-only baselines are being rerun under the stride-4 main protocol. The older stride-1 metrics were a dense-window pilot and are no longer active evidence.

## Step 6 Reconstruction Snapshot

Step 6 reconstruction-only baselines are also being rerun under the stride-4 main protocol. Step 6 will keep `model_variant` as the canonical reconstruction variant key and will continue exporting LOBench-compatible reconstruction metrics after rerun.

## Next Comparison

The next step is Step 7 alignment analysis under the same split. I want the comparison to answer:

- Does lower reconstruction error improve accuracy or macro-F1?
- Are top-of-book errors more predictive of downstream failure than deeper-level errors?
- Do spread widening, higher volatility, or weak-trend periods break the representation first?
- Does any representation justify its latency or compression cost?
- Step 7 now has both interfaces needed for sample-level alignment: `per_sample_reconstruction_errors.csv` (Step 6) and `per_sample_predictions.csv` (Step 5).
- Step 7 should rename Step 5 `model` to `prediction_model` and Step 6 `model_variant` to `reconstruction_variant`; the sample-level join key is `sample_id + split`.

## Metrics

Reconstruction metrics:

- Overall MSE / MAE
- Price-side and volume-side error
- LOBench-compatible weighted MSE / all-loss
- Level-wise error
- Top-of-book error

Prediction metrics:

- Accuracy
- Balanced accuracy
- Macro-F1
- MCC
- Log loss
- Per-class precision / recall / F1
- Raw and row-normalized confusion matrix

Directional diagnostics:

- Non-neutral recall
- Non-neutral precision
- Directional accuracy on true non-neutral samples
- `direction_correct_non_neutral` is encoded as 1.0/0.0 for true non-neutral samples and null for neutral samples.
- Up/down macro-F1
- Opposite-direction rate

Diagnostic cuts:

- Correlation between reconstruction metrics and prediction metrics
- Level-wise error versus prediction failure
- Performance by spread, volatility, and mid-price movement strength

## Limits

The current evidence is intentionally narrow:

- Only one segment of `sz000001` is covered.
- The external dataset is required locally and is not committed to the repo.
- The active label is `trend5`; other horizons remain future sensitivity checks.
- There is no multi-symbol, multi-date, or cross-regime robustness result yet.
- Step 6 runs reconstruction-only baselines; it does not include prediction-head training.
- Reconstruction-prediction alignment is still not claimed.

The next meaningful milestone is Step 7: align per-sample reconstruction errors with downstream prediction outcomes under the same locked protocol. Step 7 should treat `direction_correct_non_neutral` as numeric 1.0/0.0 for true non-neutral samples and null for neutral samples.
