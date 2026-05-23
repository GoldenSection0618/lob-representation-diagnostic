# Data Note

This note locks the data contract. Baselines, prediction heads, and analysis code should follow this document so that we do not keep changing the dataset definition while interpreting results.

## Upstream Reading

I use the local `~/LOBench` checkout as reference material, not as project code.

Files inspected:

- `~/LOBench/data/data_ashare.py`
- `~/LOBench/data/data_processing.py`
- `~/LOBench/data/data_prepare.py`
- `~/LOBench/data/data_sampling.py`
- `~/LOBench/config_template.json`
- `~/LOBench/README.md`

Key decisions:

- `data_ashare.py` is the main A-share LOB pipeline. It covers raw CSV reading, trading-session filtering, 3-second resampling, normalization, sliding windows, trend labels, balanced sampling, and NPZ-to-PyTorch dataset/datamodule loading.
- `data_processing.py` is the processed/simulation-style entry point. It contains 40-feature extraction, `get_labels`, and sequence sample generation.
- `data_prepare.py` is closer to a tensor dataloader helper: it loads with `torch.load` and applies random split. It is not the raw A-share preprocessing entry point.
- The inspected upstream code uses `CSV`, `NPZ`, and `PT`. Hugging Face appears as a distribution path in the README, not as the direct runtime loader in the inspected pipeline.
- The upstream default is random-split oriented. The main experiment in this PoW uses chronological split instead.

Local data status was checked on `2026-05-23`: external processed data exists at `~/datasets/LOBench-A-share-processed`, and the data is not committed to this repository.

## Input Object

I represent one LOB window as:

$$
X_t \in \mathbb{R}^{T \times L \times F}
$$

For the first runnable contract in this PoW:

- `T=100`, following the common upstream `observe_time=100` / `window=100` setup.
- `L=10`, using 10 book levels.
- `F=4`, with bid price, ask price, bid volume, and ask volume per level.
- The canonical storage format is flattened `[T, 40]`.
- `[2, T, 20]` is only a derived view for models that need channel splitting. It is not the canonical storage format.

Default shapes:

- Single sample: `(100, 40)`
- Batched samples: `(N, 100, 40)`
- Structured interpretation: `(N, 100, 10, 4)`

## Feature Order

The 40-column order from `data_processing.py` is:

- Bid prices: `bestBidPrice10` to `bestBidPrice1`
- Ask prices: `bestAskPrice1` to `bestAskPrice10`
- Bid volumes: `bestBidVolume10` to `bestBidVolume1`
- Ask volumes: `bestAskVolume1` to `bestAskVolume10`

Top-of-book fields:

- Bid: `bestBidPrice1`, `bestBidVolume1`
- Ask: `bestAskPrice1`, `bestAskVolume1`

`data_ashare.py` uses the same type of fields without the `best` prefix: `BidPrice* / AskPrice* / BidVolume* / AskVolume*`. The PoW code maps these source names explicitly into the canonical `best*` names. I do not want metrics or model code to infer the naming convention implicitly.

## Mid-Price, Spread, and Trend Labels

Following `data_processing.py`:

- `midPrice = (bestBidPrice1 + bestAskPrice1) / 2`
- `spread = bestAskPrice1 - bestBidPrice1`

Trend labels use the gap between a future rolling mean and the current mid-price:

- Horizons: `h in {1,3,5,7,10}`
- Label fields: `trend1`, `trend3`, `trend5`, `trend7`, `trend10`
- Default threshold: `0.0001`
- Three-class mapping: up is `2`, down is `0`, neutral is `1`

`data_ashare.py` also has a relative-threshold variant using `theta` and labels in `{-1,0,1}`. I record it as an upstream variant, but I do not use it as the main experiment contract. Mixing the two label definitions would make downstream metrics hard to interpret.

The first downstream run uses `trend5`. `trend1/3/7/10` can be added later for sensitivity analysis.

## Chronological Split Policy

The main experiment uses boundary-purged chronological split.

Boundary purge is mandatory for the current main protocol, and the Step 3 subset already applies it.

I do not use random split in the main workflow. LOB windows are highly time-dependent and heavily overlapping; random split can place near-neighbor windows into train and test and leak temporal context.

Main rules:

- Train, validation, and test must preserve time order.
- The three segments must not overlap in sample IDs or label rows.
- Boundary windows must be purged so adjacent split boundaries do not share historical rows.
- Any random-split result must be labeled as an auxiliary diagnostic only.
- No-purge chronological split is intentionally postponed because it is an ablation, not required for the current PoW main delivery.

The current default split ratio is `70/15/15`.

## Step 3 Actual Subset

I completed the minimal subset build on `2026-05-23`.

Input:

- Symbol: `sz000001`
- CSV: `~/datasets/LOBench-A-share-processed/sz000001-level10_processed.csv`
- Source fields: `BidPrice* / AskPrice* / BidVolume* / AskVolume*`
- Output directory: `data/processed/minimal_subset/`

Parameters:

- `window_len=100`
- `label_horizon=5`
- `threshold=0.0001`
- `split_ratio=70/15/15`
- `row_limit=50000`
- `max_samples=8000`

Results:

- Usable rows after label trimming: `49990`
- Final samples after boundary purge: `7802`
- `X` shape: `(7802, 100, 40)`
- `y` shape: `(7802,)`
- Split sizes: `train=5600`, `val=1200`, `test=1002`
- Train label rows: `99..5698`
- Validation label rows: `5798..6997`
- Test label rows: `7097..8098`

Checks:

- `feature_contract_check` passed
- `label_contract_check` passed
- `window_alignment_check` passed
- `chronological_split_check` passed
- `output_safety_check` passed

The important part is not the sample count. The important part is that field mapping, windowing, labeling, splitting, and boundary handling are all recorded and auditable through metadata.

## Open Decisions

- Whether future fixed subsets should keep using `row_limit + max_samples`, or move to explicit date ranges.
- For the current main protocol, boundary purge is mandatory. Making it configurable is postponed to future ablation work.
- For multi-symbol experiments, whether normalization and splitting should happen per symbol or through a broader cross-symbol design.
