# LOB Representation Diagnostic

This repo is a diagnostic PoW testing whether better LOB reconstruction transfers to downstream `trend5` prediction under a leakage-aware chronological split.

## Main Finding

- In this one-symbol, one-horizon, stride-4 subset, aggregate reconstruction MSE is not a reliable standalone proxy for downstream prediction.
- The strongest mismatch is driven by `last_snapshot_repeat@40`, which has weak full-window reconstruction but the strongest frozen-head test macro-F1.
- After excluding `last_snapshot_repeat@40`, the rank mismatch weakens and `pca@128` becomes both reconstruction-best and prediction-best.
- The post hoc best frozen latent head beats the tuned raw-window logistic control in this subset, but this remains descriptive rather than fully pre-registered confirmatory evidence.

This is not a full LOBench reproduction, not a state-of-the-art claim, not a trading PnL study, and not a general market prediction claim.

## Protocol at a Glance

| Field | Current main protocol |
| --- | --- |
| Dataset source | External LOBench A-share processed data |
| Symbol | `sz000001` |
| Label | `trend5` |
| Window | `100` |
| Feature dimension | `40` |
| Sample stride | `4` |
| Split | Boundary-purged chronological `70/15/15` |
| Samples | `7952` |
| Train / val / test | `5600 / 1200 / 1152` |
| Data policy | External data, generated tensors, checkpoints, and latent arrays are not committed |

## Evidence Snapshot

The main evidence combines raw-window prediction baselines, reconstruction-only baselines, frozen-latent transfer, and robustness checks.

| Evidence | Result | Caveat |
| --- | --- | --- |
| Best raw-window Step 5 baseline | logistic regression, test macro-F1 `0.3972` | Fixed-C baseline |
| Tuned raw-window logistic control | test macro-F1 `0.3904`, selected `C=0.1` | Selected by validation macro-F1 |
| Raw-window test-oracle reference | test macro-F1 `0.4101` | Post hoc only, not selection-valid |
| Best frozen latent head | `last_snapshot_repeat@40`, test macro-F1 `0.4355` | Selected post hoc from Step 7 test macro-F1 |
| Paired bootstrap, best latent vs tuned raw | macro-F1 delta `0.0452`, 95% CI `[0.0082, 0.0823]`, `fraction_delta_gt_0=0.9930` | Descriptive, not fully pre-registered |
| Best reconstruction variant | `pca@128`, test normalized MSE `0.1838` | Best reconstruction is not best prediction across all variants |
| Rank sensitivity | excluding `last_snapshot_repeat@40` makes `pca@128` both reconstruction-best and prediction-best | Weakens the broad rank-mismatch claim |
| Step 7 join validation | `70560` joined rows, zero duplicate joined keys | Supports sample-level alignment contract |

![Fair transfer macro-F1 with CI](figures/step8_fairness_robustness/fair_transfer_macro_f1_with_ci.png)

*Caption: The post hoc best frozen latent head remains above the tuned raw-window logistic control on test macro-F1 in this subset.*

![Rank alignment](figures/step7_alignment/reconstruction_prediction_rank_alignment.png)

*Caption: Reconstruction rank and frozen-head prediction rank do not perfectly align across all variants, with the strongest mismatch driven by `last_snapshot_repeat@40`.*

![Rank sensitivity by variant set](figures/step8_fairness_robustness/rank_sensitivity_by_variant_set.png)

*Caption: Rank mismatch weakens after excluding `last_snapshot_repeat@40`, so the mismatch claim is partial rather than general.*

## Evidence Map

| File | Purpose |
| --- | --- |
| [technical_memo.md](technical_memo.md) | Final technical memo and conservative interpretation |
| [docs/artifact_index.md](docs/artifact_index.md) | Primary and supporting evidence files |
| [docs/reproduction_guide.md](docs/reproduction_guide.md) | Commands to reproduce Step 3 to Step 8 |
| [results/step8_fairness_robustness/final_claim_table.csv](results/step8_fairness_robustness/final_claim_table.csv) | Claim status table |
| [results/step7_alignment/join_contract.json](results/step7_alignment/join_contract.json) | Join validation |
| [docs/data_note.md](docs/data_note.md) | Data contract, subset facts, and split policy |
| [docs/environment.md](docs/environment.md) | Local runtime and external data assumptions |

## Reproduction

Reproduction commands are collected in `docs/reproduction_guide.md`. The pipeline requires the external processed A-share dataset locally. Raw data, generated tensors, checkpoints, and latent arrays are not committed.

## Scope and Limitations

| Boundary | Current status |
| --- | --- |
| Symbol coverage | One symbol, `sz000001` |
| Horizon coverage | One label horizon, `trend5` |
| Sampling protocol | One stride-4 subset |
| Split protocol | Boundary-purged chronological only |
| Multi-symbol robustness | Not evaluated |
| Multi-horizon robustness | Not evaluated |
| Trading PnL | Not evaluated |
| Best frozen latent head | Selected post hoc |
| Bootstrap comparison | Descriptive, not fully pre-registered confirmatory evidence |

## Repository Layout

- `src/data/`: data loading, field mapping, labels, subset construction.
- `src/models/`: prediction and reconstruction baseline models.
- `src/analysis/`: metrics and diagnostic utilities.
- `scripts/`: runnable stage entry points.
- `docs/`: protocol, artifact index, and reproduction notes.
- `results/`: committed result and audit artifacts.
- `figures/`: plots and visual diagnostics.
