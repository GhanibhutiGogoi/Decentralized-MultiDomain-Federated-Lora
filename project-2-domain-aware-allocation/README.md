# Project 2: Domain-Aware Allocation

Project 2 extends the Project 1 federated LoRA runtime while keeping Project 2
in its own redesigned repository structure. The implementation and real-data
regeneration are complete; artifacts include Experiment 1/2 outputs,
checkpoint/resume support, strict provenance validation, and conservative
domain weighting.

Project 1 mathematical formulations, adaptive rank logic, lambda calibration,
and aggregation code are not redefined here.
Experiment 1 and the Experiment 3 runtime reuse the historical stateless
`estimate_optimal_rank` rule. They do not run the later stateful
`AdaptiveRankController`, its warmup, or its quality-drop recovery.

## Current Status

- Experiment 1 runner exists at `experiment/experiment1/run.py`.
- Experiment 2 runner exists at `experiment/experiment2/run.py`.
- Real-data outputs and provenance manifests are recorded under
  `docs/artifacts/p2-exp1-real-seed42/` and `docs/artifacts/p2-exp2-real-seed42/`.
- Dataset loading is centralized in `framework/datasets/factory.py`.
- Experiment 2 evaluation and reporting are implemented in
  `experiment/experiment2/evaluation.py`, `figures.py`, and `reporting.py`.
- Experiment 2 writes a strict `exp3-calibration-bundle/v3` bundle for
  Experiment 3 only when at least one treatment form is supported by the
  null-model and Ridge-boundary gates.
- Experiment 3 infrastructure is implemented under `experiment/experiment3/`;
  scientific runs require a validated calibration bundle, explicit source run
  IDs, and a configured MDE.

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

Prerequisite: Python 3.10 or newer. Project 2 uses modern Python type syntax
that is evaluated at import time and is not supported by Python 3.9.

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

Experiment 2 automatically computes regression metrics, ranking metrics,
ranking permutation tests, and fold-safe null comparisons during future reruns.
Pooled permutation tests are stratified by the configured aggregation context
columns. Ranking metrics are reported only. Form A and Form B retain the
original raw-RMSE support gate; Form C is a new post-hoc exploratory candidate
that uses within-task-round normalized predictors and a within-task-round
relative contribution target.
After a successful real-data calibration with at least one supported treatment
form, Experiment 2 writes `outputs/exp2/calibration_bundle.json` with schema
`exp3-calibration-bundle/v3`; the bundle is written only after input
provenance, numeric validation, support-decision evaluation, calibration,
report generation, and artifact writes succeed.

Experiment 2 requires explicit `is_synthetic` provenance in Experiment 1
measurement tables. It does not backfill missing provenance as real data.

Class imbalance is computed with a finite missing-class penalty:
`max_count / min_positive_count * (1 + zero_class_count / num_classes)`.

Prepared optional Ridge-alpha controls for a future rerun:

If Ridge alpha selection lands on the minimum or maximum tested value, that
Ridge form is recorded as unsupported for Experiment 3 eligibility. The search
grid is not expanded automatically, and no fallback alpha or lambda parameters
are fabricated.

```powershell
python project-2-domain-aware-allocation/experiment/experiment2/run.py --include-extended-ridge-alphas
```

or:

```powershell
python project-2-domain-aware-allocation/experiment/experiment2/run.py --ridge-alphas 0.01 0.1 1 10 100 300 500 1000
```

Do not rerun experiments during the current infrastructure hardening phase.
See `EXPERIMENT1.md` for Experiment 1 details and `EXPERIMENT2.md` for
Experiment 2 evaluation infrastructure.

Experiment 3 production CLI:

```powershell
python project-2-domain-aware-allocation/experiment/experiment3/run.py --calibration-bundle project-2-domain-aware-allocation/outputs/exp2/calibration_bundle.json --experiment1-run-id <exp1-run-id> --experiment2-run-id <exp2-run-id> --mde <minimum-detectable-effect>
```

Experiment 3 writes paired arm differences, paired permutation tests,
confidence intervals, task summaries, realized aggregation weights, and MDE
comparisons for baseline plus the supported treatment arms listed in the v3
bundle. It refuses synthetic or engineering calibration bundles in normal
production mode.
