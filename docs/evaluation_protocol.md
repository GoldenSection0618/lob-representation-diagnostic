# Evaluation Protocol

## Protocol Framing

Steps 3 through 9 used `chronological_purged` as a conservative baseline protocol. Step 10 adds a split-protocol comparison layer and treats the split itself as an experimental variable.

The random split is now a central diagnostic protocol, not merely a future ablation. It is not interpreted as a recommended evaluation protocol.

## Invariants

- every protocol must have explicit train, validation, and test assignments
- no sample_id overlap within a run
- split metadata must record counts, random seeds, and label-row ranges
- random protocols must report split-integrity diagnostics before performance interpretation
- post hoc or oracle rows must be explicitly marked

## Current Subset Parameters

- symbol: sz000001
- window_len: 100
- feature_dim: 40
- label: trend5
- split_ratio: 70/15/15
- sample_stride: 4
- current Step 3 split sizes: train=5600, val=1200, test=1152
- current Step 3 boundary drops: 48 samples

## Step 10 Protocols

| Protocol | Role | Interpretation |
| --- | --- | --- |
| `chronological_purged` | Conservative baseline used by Steps 3-9 | Boundary-purged chronological evaluation |
| `random_window_naive` | Core random diagnostic | Optimistic, leakage-prone window-level random split |
| `random_block_purged` | Core control | Block-level random split with embargo to reduce near-neighbor exposure |
| `chronological_no_purge` | Boundary diagnostic | No additional purge on the existing kept sample universe |

Interpretation rules:

- `random_window_naive - chronological_purged`: temporal mixing, near-neighbor exposure, and purge differences mixed together.
- `random_block_purged - chronological_purged`: closer to temporal or regime mixing with near-neighbor exposure reduced.
- `random_window_naive - random_block_purged`: closer to near-neighbor exposure and overlapping-window effect.
- `chronological_no_purge - chronological_purged`: boundary-purge effect proxy on the existing kept sample universe.

## Still Out of Scope

- full LOBench reproduction
- multi-symbol experiments
- multi-horizon experiments
- new model families
- trading PnL
