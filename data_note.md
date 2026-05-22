# Data Note

## Purpose

This file records LOBench-style data shape, feature layout, label definition, split protocol, and subset protocol.

## Data Source

- Upstream repository: TODO
- Local upstream path: TODO
- Processed data path: TODO
- Raw data availability: TODO
- Data license / redistribution constraint: TODO

## LOB Object Definition

A LOB window will be represented as:

X_t ∈ R^{T × L × F}

where:

- T: lookback window length
- L: number of order book levels
- F: features per level, such as bid price, ask price, bid volume, ask volume

## Feature Layout

- bid price columns: TODO
- ask price columns: TODO
- bid volume columns: TODO
- ask volume columns: TODO
- top-of-book level: TODO
- mid-price definition: TODO
- spread definition: TODO
- order imbalance definition: TODO

## Label Definition

- prediction target: TODO
- horizon h: TODO
- binary or multi-class: TODO
- stationary threshold: TODO
- label source: TODO
- whether labels are precomputed or reconstructed: TODO

## Chronological Split Protocol

- Splits must preserve time order.
- Random split is not allowed for the main experiment.
- Train / validation / test split should be chronological.

## Subset Protocol

- instrument: TODO
- time range: TODO
- number of samples: TODO
- train size: TODO
- validation size: TODO
- test size: TODO
- saved path: TODO
- metadata file: TODO

## Known Uncertainties

- Whether processed LOBench data is available locally.
- Whether labels are precomputed.
- Whether SimLOB and LOBench use exactly the same window definition.
- Whether reconstruction and prediction tasks share the same input normalization.
