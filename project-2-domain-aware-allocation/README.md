# Project 2: Domain-Aware Allocation

Project 2 extends the Project 1 federated LoRA runtime while keeping Project 2
in its own redesigned repository structure. The current phase is infrastructure
hardening before the final Experiment 1 and Experiment 2 rerun.

Project 1 mathematical formulations, adaptive rank logic, lambda calibration,
and aggregation code are not redefined here.

## Current Status

- Experiment 1 runner exists at `experiment/experiment1/run.py`.
- Experiment 2 runner exists at `experiment/experiment2/run.py`.
- Existing outputs under `outputs/` are historical artifacts and should not be
  regenerated until the Project 1 mathematical review is complete.
- Dataset loading is centralized in `framework/datasets/factory.py`.

## Structure

- `framework/datasets/`: centralized dataset factory plus modality adapters.
- `framework/partitioning/`: IID and Dirichlet client partitioning utilities.
- `framework/models/`: Project 2 model support code.
- `framework/federated/`: Project 2 federated training infrastructure.
- `framework/aggregation/`: aggregation utilities.
- `framework/rank_allocation/`: rank allocation utilities.
- `framework/analysis/`: analysis utilities.
- `framework/visualization/`: plotting utilities.
- `framework/configuration/`: default infrastructure configuration.
- `framework/utils/`: reproducibility and runtime environment helpers.
- `experiment/data/`: Project 2 dataset cache root.
- `experiment/experiment1/`: domain-signal and marginal-contribution runner.
- `experiment/experiment2/`: lambda calibration from Experiment 1 outputs.
- `outputs/exp1/`, `outputs/exp2/`: experiment output directories.

## Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r project-2-domain-aware-allocation/requirements.txt
```

Required core packages are declared in `requirements.txt`, including `torch`,
`torchvision`, `torchtext`, `torchaudio`, `pandas`, `numpy`, `scipy`,
`scikit-learn`, and `matplotlib`.

## Dataset Workflow

All Project 2 experiments should load datasets through
`framework.datasets.DatasetFactory`.

Defaults:

- data root: `project-2-domain-aware-allocation/experiment/data`
- real datasets only
- automatic synthetic fallback disabled
- automatic download disabled
- missing real datasets fail with a clear error

To explicitly download missing real datasets during a future rerun:

```powershell
python project-2-domain-aware-allocation/experiment/experiment1/run.py --download-datasets
```

Synthetic data is allowed only when requested explicitly for supported tasks:

```powershell
python project-2-domain-aware-allocation/experiment/experiment1/run.py --synthetic-datasets AGNews-LSTM
```

## Validation And Manifests

Before training begins, each loaded dataset is validated for:

- cache presence
- dataset object type
- split sample counts when known
- class count
- synthetic flag
- download status

Each experiment output directory receives `dataset_manifest.json`. Experiment 1
records raw dataset provenance directly. Experiment 2 requires Experiment 1's
dataset manifest and stores it as its source dataset provenance.

## Experiment Workflow

Experiment 1:

```powershell
python project-2-domain-aware-allocation/experiment/experiment1/run.py --partition dirichlet --alpha 0.5 --seed 42
```

Experiment 2:

```powershell
python project-2-domain-aware-allocation/experiment/experiment2/run.py
```

Do not rerun experiments during the current infrastructure hardening phase.
See `EXPERIMENT1.md` for Experiment 1 details.
