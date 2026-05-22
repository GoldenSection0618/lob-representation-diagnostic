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
