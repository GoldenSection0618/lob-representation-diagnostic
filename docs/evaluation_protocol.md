# Evaluation Protocol

## Main Protocol

Boundary-purged chronological split.

## Invariants

- train, validation, and test preserve chronological order
- no sample_id overlap
- no label_row overlap
- no shuffled split
- no overlapping historical rows across train/validation and validation/test boundaries
- split metadata must record actual counts and label-row ranges

## Current Locked Parameters

- symbol: sz000001
- window_len: 100
- feature_dim: 40
- label: trend5
- split_ratio: 70/15/15
- sample_stride: 4
- boundary purge: enabled and mandatory

## Excluded From Step 4

- random split
- no-purge chronological split
- reconstruction models
- prediction heads
- multi-symbol experiments

## Future Extensions

Random split and no-purge split may be added later only as explicitly labeled ablations.
