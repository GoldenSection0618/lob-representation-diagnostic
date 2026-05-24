# LOB Representation Diagnostic

This repo is a diagnostic PoW testing whether better LOB reconstruction transfers to downstream `trend5` prediction, and how split protocol choices affect that conclusion.

## Main Finding

- In this one-symbol, one-horizon, stride-4 subset, aggregate reconstruction MSE is not a reliable standalone proxy for downstream prediction.
- The strongest mismatch is driven by `last_snapshot_repeat@40`, which has weak full-window reconstruction but the strongest frozen-head test macro-F1.
- After excluding `last_snapshot_repeat@40`, the rank mismatch weakens and `pca@128` becomes both reconstruction-best and prediction-best.
- Step 9 reduces the post hoc representation-selection caveat: validation macro-F1 and test macro-F1 both select `last_snapshot_repeat@40` in this run.
- The validation-selected frozen latent head beats the tuned raw-window logistic control in this subset, but this remains descriptive rather than general evidence.
- Step 10 shows that naive random window-level splitting is optimistic in this subset: it produces full train/test near-neighbor exposure and materially higher macro-F1, while blocked random with purge removes that exposure and stays close to chronological performance.

This is not a full LOBench reproduction, not a state-of-the-art claim, not a trading PnL study, and not a general market prediction claim.

## Protocol at a Glance

| Field | Setting |
| --- | --- |
| Dataset source | External LOBench A-share processed data |
| Symbol | `sz000001` |
| Label | `trend5` |
| Window | `100` |
| Feature dimension | `40` |
| Sample stride | `4` |
| Conservative baseline split | Boundary-purged chronological `70/15/15` |
| Step 10 split diagnostics | `random_window_naive`, `random_block_purged`, `chronological_no_purge` |
| Samples | `7952` |
| Train / val / test | `5600 / 1200 / 1152` |
| Data policy | External data, generated tensors, checkpoints, and latent arrays are not committed |

## Evidence Snapshot

The main evidence combines raw-window prediction baselines, reconstruction-only baselines, frozen-latent transfer, robustness checks, a validation-selected representation audit, and split-protocol decomposition.

| Evidence | Result | Caveat |
| --- | --- | --- |
| Best raw-window Step 5 baseline | logistic regression, test macro-F1 `0.3972` | Fixed-C baseline |
| Tuned raw-window logistic control | test macro-F1 `0.3904`, selected `C=0.1` | Selected by validation macro-F1 |
| Raw-window test-oracle reference | test macro-F1 `0.4101` | Post hoc only, not selection-valid |
| Validation-selected frozen latent head | `last_snapshot_repeat@40`, test macro-F1 `0.4355` | Selected by validation macro-F1, not test |
| Step 9 selection audit | validation-selected and test-posthoc best are both `last_snapshot_repeat@40` | Reduces but does not eliminate selection-bias caveats |
| Paired bootstrap, validation-selected latent vs tuned raw | macro-F1 delta `0.0452`, 95% CI `[0.0082, 0.0799]`, `fraction_delta_gt_0=0.9930` | Descriptive, not fully pre-registered |
| Best reconstruction variant | `pca@128`, test normalized MSE `0.1838` | Best reconstruction is not best prediction across all variants |
| Rank sensitivity | excluding `last_snapshot_repeat@40` makes `pca@128` both reconstruction-best and prediction-best | Weakens the broad rank-mismatch claim |
| Step 7 join validation | `70560` joined rows, zero duplicate joined keys | Supports sample-level alignment contract |
| Step 10 integrity audit | naive random train/test overlap risk `1.0000`; blocked random overlap risk `0.0000` | Random split result is diagnostic, not recommended evaluation |
| Step 10 raw tuned contrast | naive random improves test macro-F1 by `0.0583`; blocked random improves by `0.0004` | Suggests most naive-random gain comes from near-neighbor exposure, not pure regime mixing |

Note: Step 10 is a lightweight within-step protocol rerun. Its absolute metrics should be interpreted through within-step contrasts rather than as replacements for the Step 8/9 headline metrics. `chronological_no_purge` is a no-extra-purge diagnostic on the existing Step 3 kept sample universe; it does not restore Step 3 boundary-dropped samples.

![Step 10 protocol diagnostic overview](figures/step10_split_protocol_decomposition/protocol_diagnostic_overview.png)

*Caption: Naive random window-level splitting has full train/test overlap and k5 near-neighbor exposure in this subset, while blocked random with embargo removes that exposure. The performance panel shows that naive random also raises test macro-F1.*

![Step 10 macro-F1 delta decomposition](figures/step10_split_protocol_decomposition/macro_f1_delta_decomposition.png)

*Caption: Most of the naive-random macro-F1 gain disappears when switching from naive window-level randomization to blocked random with embargo, especially for the tuned raw-window logistic control.*

![Fair transfer macro-F1 with CI](figures/step8_fairness_robustness/fair_transfer_macro_f1_with_ci.png)

*Caption: Step 8 fair-transfer visualization. Step 9 shows the same latent variant is selected by validation macro-F1, so the plotted best-frozen-latent comparison matches the validation-selected comparison in this run.*

![Rank alignment](figures/step7_alignment/reconstruction_prediction_rank_alignment.png)

*Caption: Reconstruction rank and frozen-head prediction rank do not perfectly align across all variants, with the strongest mismatch driven by `last_snapshot_repeat@40`.*

![Rank sensitivity by variant set](figures/step8_fairness_robustness/rank_sensitivity_by_variant_set.png)

*Caption: Rank mismatch weakens after excluding `last_snapshot_repeat@40`, so the mismatch claim is partial rather than general.*

## Evidence Map

| File | Purpose |
| --- | --- |
| [technical_memo.md](technical_memo.md) | Final technical memo and conservative interpretation |
| [docs/artifact_index.md](docs/artifact_index.md) | Primary and supporting evidence files |
| [docs/reproduction_guide.md](docs/reproduction_guide.md) | Commands to reproduce Step 3 to Step 10 |
| [results/step9_validation_selection_audit/step9_manifest.json](results/step9_validation_selection_audit/step9_manifest.json) | Current validation-selected representation audit |
| [results/step9_validation_selection_audit/fair_transfer_comparison.csv](results/step9_validation_selection_audit/fair_transfer_comparison.csv) | Current fair transfer comparison after validation selection |
| [results/step10_split_protocol_decomposition/protocol_contrasts.csv](results/step10_split_protocol_decomposition/protocol_contrasts.csv) | Split protocol decomposition contrasts |
| [results/step10_split_protocol_decomposition/split_integrity_audit.csv](results/step10_split_protocol_decomposition/split_integrity_audit.csv) | Overlap and near-neighbor risk audit by split protocol |
| [results/step8_fairness_robustness/final_claim_table.csv](results/step8_fairness_robustness/final_claim_table.csv) | Step 8 claim table before the Step 9 representation-selection audit |
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
| Split protocol | Step 10 compares chronological, naive random, blocked random, and no-purge diagnostics on the same subset |
| Multi-symbol robustness | Not evaluated |
| Multi-horizon robustness | Not evaluated |
| Trading PnL | Not evaluated |
| Best frozen latent head | Validation-selected in Step 9; candidate set fixed by earlier steps |
| Bootstrap comparison | Descriptive, not fully pre-registered confirmatory evidence |

## Repository Layout

- `src/data/`: data loading, field mapping, labels, subset construction.
- `src/models/`: prediction and reconstruction baseline models.
- `src/analysis/`: metrics and diagnostic utilities.
- `scripts/`: runnable stage entry points.
- `docs/`: protocol, artifact index, and reproduction notes.
- `results/`: committed result and audit artifacts.
- `figures/`: plots and visual diagnostics.
