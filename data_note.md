# Data Note

This document is the data contract for the PoW. Model code, baselines, and analysis should follow it rather than reinterpreting the dataset midstream.

## Upstream Reference

I use the local `~/LOBench` checkout as read-only reference material. I inspected:

- `~/LOBench/data/data_ashare.py`
- `~/LOBench/data/data_processing.py`
- `~/LOBench/data/data_prepare.py`
- `~/LOBench/data/data_sampling.py`
- `~/LOBench/config_template.json`
- `~/LOBench/README.md`

The useful split of responsibilities is clear enough:

- `data_ashare.py` is the main A-share pipeline: raw CSV reading, trading-session filtering, 3-second resampling, normalization, sliding windows, trend labels, balanced sampling, and NPZ-to-PyTorch loading.
- `data_processing.py` is the processed/simulation-style path with 40-feature extraction, `get_labels`, and sequence sample generation.
- `data_prepare.py` loads already materialized tensors and applies random split. I do not use it as the PoW contract.
- The upstream code I inspected works with `CSV`, `NPZ`, and `PT`. Hugging Face is a distribution channel in the README, not the runtime loader I observed.

The local external dataset was present on `2026-05-23` at `~/datasets/LOBench-A-share-processed`. It stays outside git.

## LOB Window Contract

A window is represented as:

$$
X_t \in \mathbb{R}^{T \times L \times F}
$$

The current contract fixes:

- `T=100`, aligned with upstream `observe_time=100` / `window=100`.
- `L=10`, using 10 book levels.
- `F=4`, with bid price, ask price, bid volume, and ask volume per level.
- Canonical storage: flattened `[T, 40]`.
- Optional model view: `[2, T, 20]`, derived from `[T, 40]` only when a model needs channel splitting.

Expected shapes:

- One sample: `(100, 40)`
- Batch: `(N, 100, 40)`
- Structured view: `(N, 100, 10, 4)`

## Feature Order

The canonical 40-column order follows `data_processing.py`:

- Bid prices: `bestBidPrice10` through `bestBidPrice1`
- Ask prices: `bestAskPrice1` through `bestAskPrice10`
- Bid volumes: `bestBidVolume10` through `bestBidVolume1`
- Ask volumes: `bestAskVolume1` through `bestAskVolume10`

Top-of-book is `bestBidPrice1`, `bestBidVolume1`, `bestAskPrice1`, and `bestAskVolume1`.

Some A-share files use `BidPrice* / AskPrice* / BidVolume* / AskVolume*` without the `best` prefix. The PoW loader maps those fields explicitly into the canonical names. I do not let downstream code guess the convention.

## Labels

The main label contract follows `data_processing.py`:

- `midPrice = (bestBidPrice1 + bestAskPrice1) / 2`
- `spread = bestAskPrice1 - bestBidPrice1`
- Trend is the gap between a future rolling mean and current mid-price.
- Horizons are `h in {1,3,5,7,10}`.
- Label fields are `trend1`, `trend3`, `trend5`, `trend7`, `trend10`.
- Default threshold is `0.0001`.
- Class mapping is `2` for up, `0` for down, and `1` for neutral.

`data_ashare.py` has a related relative-threshold label using `theta` and `{-1,0,1}`. I keep it documented as an upstream variant, but it is not the main PoW label definition. Mixing both would make the metrics ambiguous.

The first downstream target is `trend5`; the other horizons are reserved for later sensitivity checks.

## Split Policy

The main split is boundary-purged chronological. The Step 3 subset already applies it, and Step 4 locked it as the current evaluation protocol.
The main window sampling protocol uses `sample_stride=4`, aligned with the upstream LOBench-style A-share sampling convention. The earlier dense `sample_stride=1` run was a pilot and is no longer the active main evidence.

The rule is intentionally conservative. LOB windows overlap heavily, so a plain random split can put near-neighbor windows into train and test. Even a chronological split needs boundary purge, otherwise adjacent segments can share historical rows through the sliding window.

Current rules:

- Train, validation, and test preserve time order.
- Windows are sampled every 4 label rows to reduce near-duplicate overlap while preserving chronological order.
- Sample IDs and label rows do not overlap across splits.
- Historical rows do not overlap across train/validation or validation/test boundaries.
- Random split is auxiliary only if it is ever added.
- No-purge chronological split is a future ablation, not the current main protocol.

Default ratio: `70/15/15`.

## Step 3 Subset

The minimal subset build completed on `2026-05-23`.

Input and parameters:

- Symbol: `sz000001`
- CSV: `~/datasets/LOBench-A-share-processed/sz000001-level10_processed.csv`
- Source fields: `BidPrice* / AskPrice* / BidVolume* / AskVolume*`
- Output: `data/processed/minimal_subset/`
- `window_len=100`
- `label_horizon=5`
- `threshold=0.0001`
- `split_ratio=70/15/15`
- `row_limit=50000`
- `max_samples=8000`

Output facts:

- Usable rows after label trimming: `49990`
- Final samples after boundary purge: `7802`
- `X` shape: `(7802, 100, 40)`
- `y` shape: `(7802,)`
- Split sizes: `train=5600`, `val=1200`, `test=1002`
- Train label rows: `99..5698`
- Validation label rows: `5798..6997`
- Test label rows: `7097..8098`

Checks passed:

- `feature_contract_check`
- `label_contract_check`
- `window_alignment_check`
- `chronological_split_check`
- `output_safety_check`

The sample count is not the headline. The headline is that field mapping, labels, windows, split boundaries, and output safety are all explicit and auditable through metadata.

## Remaining Choices

- Fixed subsets can stay on `row_limit + max_samples`, or move to explicit date ranges.
- Boundary purge is mandatory for the current protocol; making it configurable belongs in ablation work.
- Multi-symbol experiments still need a decision on per-symbol versus cross-symbol normalization and splitting.
