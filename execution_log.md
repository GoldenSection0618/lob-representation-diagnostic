# PoW Execution Log

This log records decisions and reproducibility-relevant facts. It is not a full activity diary.

## Step 1: Lock the Boundary First

I initialized an independent PoW repository at `~/lob-representation-diagnostic`.

The repository is a diagnostic experiment, not a LOBench fork and not a full reproduction. The core question is whether better LOB reconstruction reliably transfers into better mid-price trend prediction.

Initial project files and directories:

- `README.md`
- `environment.md`
- `data_note.md`
- `technical_memo.md`
- `configs/`
- `src/`
- `scripts/`
- `results/`
- `figures/`

The next move was to inspect the LOBench / SimLOB-style data pipeline and lock the dataset format, loader entry points, split behavior, label definition, and minimal subset requirements.

## Step 2: Inspect the Upstream Data Pipeline

I inspected the local `~/LOBench` checkout at commit `c8fe9e7`.

Files reviewed:

- `~/LOBench/data/data_ashare.py`
- `~/LOBench/data/data_processing.py`
- `~/LOBench/data/data_prepare.py`
- `~/LOBench/data/data_sampling.py`
- `~/LOBench/config_template.json`
- `~/LOBench/README.md`

Findings that drive the PoW design:

- The main A-share path is in `data_ashare.py`.
- Processed/simulation-style logic is in `data_processing.py`.
- `data_prepare.py` works on already saved tensors and random split; it is not a good main contract for this PoW.
- The inspected upstream code uses `CSV`, `NPZ`, and `PT`. Hugging Face is a distribution path, not the direct runtime loader I observed.
- The 40-feature LOB layout expands 10 levels of bid/ask prices and volumes.
- `midPrice = (bestBidPrice1 + bestAskPrice1)/2`.
- `spread = bestAskPrice1 - bestBidPrice1`.
- `data_processing.py` builds three-class trend labels from a thresholded future rolling-mean gap, with horizons `{1,3,5,7,10}`.
- `data_ashare.py` has another label convention using `{-1,0,1}` and a relative threshold, so I do not mix the two definitions.
- Upstream defaults lean toward random split. This PoW uses chronological split for the main experiment.

I also confirmed that local external processed A-share files exist under `~/datasets/LOBench-A-share-processed` as `*-level10_processed.csv`. The data remains outside git.

The main open points after Step 2 were field naming (`bestBidPrice*` vs `BidPrice*`), whether labels should be regenerated, and what sampling stride to use in Step 3. Step 3 converted those into an explicit code contract.

## Step 3: Build the Minimal Chronological Subset

The goal was to run one clean data path: read one external processed CSV, map it into the canonical 40-feature layout, generate LOBench-style `trend5` labels, build `window=100` samples, and enforce a chronological `70/15/15` split.

Files touched:

- `src/data/load_lobench.py`
- `src/data/labeling.py`
- `src/data/make_subset.py`
- `src/data/checks.py`
- `scripts/01_prepare_data.py`
- `data_note.md`
- `README.md`
- `execution_log.md`

I ran a dry run first, then wrote outputs.

Dry run:

```bash
mamba run -n lob python scripts/01_prepare_data.py --input-csv ~/datasets/LOBench-A-share-processed/sz000001-level10_processed.csv --symbol sz000001 --output-dir data/processed/minimal_subset --window-len 100 --label-horizon 5 --threshold 0.0001 --split-ratio 70/15/15 --row-limit 50000 --max-samples 8000 --dry-run
```

Output run:

```bash
mamba run -n lob python scripts/01_prepare_data.py --input-csv ~/datasets/LOBench-A-share-processed/sz000001-level10_processed.csv --symbol sz000001 --output-dir data/processed/minimal_subset --window-len 100 --label-horizon 5 --threshold 0.0001 --split-ratio 70/15/15 --row-limit 50000 --max-samples 8000
```

Input file:

- `~/datasets/LOBench-A-share-processed/sz000001-level10_processed.csv`

Output directory:

- `data/processed/minimal_subset/`

Metadata:

- `data/processed/minimal_subset/metadata.json`

Final run facts:

- Raw rows used: `50000`
- Usable rows after label trimming: `49990`
- Final samples: `7802`
- `X=(7802, 100, 40)`
- `y=(7802,)`
- Split sizes: `train=5600`, `val=1200`, `test=1002`
- Train max label row: `5698`
- Validation min label row: `5798`
- Validation max label row: `6997`
- Test min label row: `7097`

Checks passed:

- `feature_contract_check`
- `label_contract_check`
- `window_alignment_check`
- `chronological_split_check`
- `output_safety_check`

Step 3 is complete. Step 4 locks the evaluation protocol before baseline modeling.

## Step 4: Lock the Leakage-Aware Chronological Evaluation Protocol

Step 4 is a documentation and protocol-locking step.

What changed:

- Step 4 did not change the Step 3 data pipeline.
- Step 4 did not add random split.
- Step 4 did not add no-purge split.
- The existing `chronological_split()` implementation remains the main protocol.
- Boundary purge is mandatory for the current main evaluation.

Key invariant:

The main split must satisfy:

- train label rows < validation label rows < test label rows
- train/validation boundary windows must not overlap in historical rows
- validation/test boundary windows must not overlap in historical rows

Next step:

Step 5 will build prediction-only baselines before reconstruction baselines.
