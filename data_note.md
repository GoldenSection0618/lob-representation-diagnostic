# Data Note

## Purpose

This file records LOBench-style data shape, feature layout, label definition, split protocol, and subset protocol.

## Data Source

- Upstream repository: LOBench (external reference only)
- Local upstream path: `~/LOBench`
- Inspected upstream files:
  - `~/LOBench/data/data_ashare.py`
  - `~/LOBench/data/data_processing.py`
  - `~/LOBench/data/data_prepare.py`
  - `~/LOBench/data/data_sampling.py`
  - `~/LOBench/config_template.json`
  - `~/LOBench/README.md`
- Raw data path assumption (upstream): `dataset/real_data/` and `dataset/simu_data/raw_csv/`
- Processed data path assumption (upstream): `dataset/real_data/processed/`, `dataset/train_data/`
- Raw data availability: not committed in this PoW repo; expected to exist outside git.
- Data license / redistribution constraint: follow upstream license and do not redistribute proprietary/private LOB data in this repo.

## Step 2 Decision Summary

- `data/data_ashare.py` is the main A-share LOB data pipeline file in LOBench.
  It covers:
  1. raw CSV ingestion via `AShare.__init__`;
  2. trading-session filtering and 3-second resampling via `AShare.resample_data`;
  3. price/volume normalization via `AShare.normalize_data`;
  4. sliding-window construction via `AShare.unbalance_data`;
  5. trend-label construction and balanced sampling via `AShare.balance_data`;
  6. NPZ-to-PyTorch dataset/datamodule loading via `AShareData`, `AShareVaeData`, `AShareDataModule`, and `AShareVaeDataModule`.
- `data/data_processing.py` is the complementary processed/simulation pipeline entry (`SimDataset`), including normalized feature extraction, `get_labels`, and sequence sample generation.
- `data/data_prepare.py` is a dataloader helper that splits existing tensor data (`torch.load` + random split), not the raw A-share preprocessing entry.
- Input file formats observed in upstream code:
  - `CSV` via `pd.read_csv`
  - `NPZ` via `np.load`
  - `PT` via `torch.load` / `torch.save`
  - HuggingFace appears in README as distribution link, not direct runtime loader in inspected pipeline.
- Canonical PoW reconstruction input contract:
  - use normalized LOB window `X` in flattened shape `[T, 40]` as the primary format.
  - `[2, T, 20]` is treated as a model-specific derived view (channel split), not canonical storage format.
- Local small-data availability status (checked on 2026-05-23):
  - no runnable local `.csv/.npz/.pt/.npy` data files were found under `~/LOBench` or this PoW repo.

## LOB Object Definition

A LOB window will be represented as:

X_t ∈ R^{T × L × F}

where:

- T: lookback window length
- L: number of order book levels
- F: features per level, such as bid price, ask price, bid volume, ask volume

For Step 2 PoW contract:

- expected default T: 100 (from `observe_time=100` and `window=100` in upstream pipeline)
- expected L: 10
- expected F: 4
- flattened feature width: 40
- tensor view equivalence:
  - flattened: `(T, 40)`
  - structured: `(T, 10, 4)`

## Feature Layout

Based on `data/data_processing.py` (40 columns):

- bid price columns:
  - `bestBidPrice10, bestBidPrice9, ..., bestBidPrice1`
- ask price columns:
  - `bestAskPrice1, bestAskPrice2, ..., bestAskPrice10`
- bid volume columns:
  - `bestBidVolume10, bestBidVolume9, ..., bestBidVolume1`
- ask volume columns:
  - `bestAskVolume1, bestAskVolume2, ..., bestAskVolume10`
- top-of-book level:
  - bid: `bestBidPrice1`, `bestBidVolume1`
  - ask: `bestAskPrice1`, `bestAskVolume1`

Note on naming variant in `data/data_ashare.py`:

- uses `BidPrice* / AskPrice* / BidVolume* / AskVolume*` naming without `best` prefix.
- PoW scripts should treat this as an upstream naming variant to be mapped explicitly.

## Mid-Price and Spread Definition

From `data/data_processing.py`:

- mid-price:
  - `midPrice = (bestBidPrice1 + bestAskPrice1) / 2`
- spread:
  - `spread = bestAskPrice1 - bestBidPrice1`

## Label / Trend Definition

Observed in upstream:

- In `data/data_processing.py`:
  - trend uses future rolling mean gap:
    - `rolling(midPrice, window=h).mean().shift(-h) - midPrice`
  - class mapping (3-class):
    - `2` if value `> threshold`
    - `0` if value `< -threshold`
    - `1` otherwise
  - default threshold in function: `0.0001`
  - generated horizons: `h in {1,3,5,7,10}` as `trend1, trend3, trend5, trend7, trend10`

- In `data/data_ashare.py`:
  - alternative trend definition with relative threshold `theta` around current midprice
  - class values in `{-1, 0, 1}`

PoW contract for Step 2:

- use `data_processing.py` style labels as the primary reference for downstream prediction contract.
- keep `data_ashare.py` trend as an inspected alternative, not default.

## Window Length and Expected Tensor Shape

From upstream pipeline:

- `observe_time=100` (`SimDataset`)
- `window=100` for sliding-window sampling (`data_sampling.py`, `data_ashare.py`)

Expected shapes:

- reconstruction input baseline: `X` shaped `(N, 100, 40)`
- structured interpretation: `(N, 100, 10, 4)`
- model-specific channelized variant seen in upstream VAE path: `(N, 2, 100, 20)` derived from `(N, 100, 40)`
- label table row fields (from `get_labels`):
  - `midPrice, spread, trend1, trend3, trend5, trend7, trend10`

## Upstream Split Behavior

Observed split logic is random by default:

- `data_prepare.py`: random split `70% / 20% / 10%`
- `data_processing.py` datamodule: random split `80% / 10% / 10%`
- `data_ashare.py` datamodules: random split `80% / 10% / 10%`
- training loader often uses `shuffle=True`

Conclusion:

- upstream default split behavior is random-split oriented, not strict chronological split.

## PoW Chronological Split Policy

Main experiment policy in this repo:

- Splits must preserve time order.
- Random split is not allowed for the main experiment.
- Train / validation / test split should be chronological and non-overlapping.
- Any random-split baseline must be explicitly labeled as auxiliary only.

## Minimal Subset Protocol

Step 2 contract for initial PoW subset:

- instrument: single symbol first (candidate from upstream set, e.g., `sz000001`)
- time range: one contiguous chronological block (exact dates to be finalized after local data inspection)
- window length: 100
- level count: 10
- feature width: 40
- label horizon for first downstream run: `trend5` (primary), with `trend1/3/7/10` optional
- split policy: chronological `70/15/15` (train/val/test)
- sample target: minimal runnable subset (exact N decided after file-level inspection)
- saved path (PoW repo, ignored by git):
  - `data/processed/minimal_subset/`
- metadata file:
  - `data/processed/minimal_subset/metadata.json`

## Known Uncertainties

- Whether processed LOBench data is available locally in a directly consumable format.
- Whether labels are already precomputed in local upstream artifacts.
- Exact compatibility between `bestBidPrice*` and `BidPrice*` naming pipelines for the same dataset source.
- Whether reconstruction and prediction tasks in upstream always share the same normalization settings.
- Whether PoW subset should use step=1 or step=4 sampling for the first controlled run.
