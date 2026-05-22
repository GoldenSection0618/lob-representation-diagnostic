# PoW Execution Log

## Step 1: Repository Initialization and Scope Locking

### Objective

Initialize an independent PoW repository and lock the research scope before touching upstream code or data.

### Scope Decision

- Project type: independent diagnostic PoW repo.
- Not a fork of LOBench.
- Not a full reproduction repo.
- Core question: Does better LOB reconstruction imply better downstream prediction?
- Main output: GitHub repo + technical memo.

### Repository Path

~/lob-representation-diagnostic

### Initialized Components

- README.md
- environment.md
- data_note.md
- technical_memo.md
- configs/
- src/
- scripts/
- results/
- figures/

### Next Step

Step 2: Inspect LOBench / SimLOB data pipeline and identify dataset format, loader entry, split logic, label definition, and minimal subset requirements.

## Step 2: LOBench / SimLOB Data Pipeline Inspection

### Objective

Inspect upstream data pipeline definitions and lock a verifiable data contract for this independent PoW repository without copying upstream code.

### Inspected Files

- `~/LOBench/data/data_ashare.py`
- `~/LOBench/data/data_processing.py`
- `~/LOBench/data/data_prepare.py`
- `~/LOBench/data/data_sampling.py`
- `~/LOBench/config_template.json`
- `~/LOBench/README.md`

### Findings

- Upstream reference repository is available at `~/LOBench` (commit `c8fe9e7`).
- `data_processing.py` defines a 10-level, 40-feature flattened LOB layout:
  - prices: `bestBidPrice10..1 + bestAskPrice1..10`
  - volumes: `bestBidVolume10..1 + bestAskVolume1..10`
- Mid-price and spread in `data_processing.py`:
  - `midPrice = (bestBidPrice1 + bestAskPrice1)/2`
  - `spread = bestAskPrice1 - bestBidPrice1`
- Trend labels in `data_processing.py` are 3-class with thresholded future rolling-mean gap and horizons `{1,3,5,7,10}`.
- `data_ashare.py` contains a related but different trend rule (`{-1,0,1}` and relative `theta` threshold), so label conventions are not fully unified across upstream files.
- Upstream split behavior is random-split oriented:
  - `data_prepare.py`: 70/20/10 random split
  - `data_processing.py` and `data_ashare.py` datamodules: 80/10/10 random split
- PoW policy for this repo remains chronological split for the main experiment.

### Unresolved Uncertainties

- Whether processed LOBench files are fully available locally in a directly consumable format.
- Which upstream naming convention should be canonical for PoW ingestion (`bestBidPrice*` vs `BidPrice*`).
- Whether precomputed labels exist locally or must be regenerated during PoW preprocessing.
- Whether Step 3 subset should use step=1 or step=4 window sampling.

### Next Step

Step 3: implement minimal data inspection / subset construction script for chronological slicing and metadata export in the PoW repo.
