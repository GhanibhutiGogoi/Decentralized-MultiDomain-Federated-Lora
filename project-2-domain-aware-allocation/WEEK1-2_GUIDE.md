# Project 2 Guide

This guide reflects the redesigned Project 2 repository structure. Historical
Week 1-2 CIFAR-100 scaffolding still exists in selected framework modules, but
the active Project 2 runners are `experiment/experiment1/run.py` and
`experiment/experiment2/run.py`.

Do not rerun experiments until the final Project 1 mathematical review is
complete.

## Active Workflow

Experiment 1 measures domain heterogeneity signals and leave-one-client-out
marginal contribution while reusing Project 1 training, rank, and aggregation
runtime code.

Experiment 2 calibrates candidate lambda forms from Experiment 1 output tables.
It does not load raw datasets directly; it requires Experiment 1's
`dataset_manifest.json` as source provenance.

During a future Experiment 2 rerun, the evaluation pipeline also writes
regression metrics, ranking metrics, statistical tests, comparison tables, and
summary figures from generated output tables.

## Infrastructure Layout

- `framework/datasets/factory.py`: central dataset loading, validation, and
  manifest records.
- `framework/datasets/image.py`: CIFAR-10, CIFAR-100, and FashionMNIST adapters.
- `framework/datasets/text.py`: AG News adapter.
- `framework/datasets/tabular.py`: UCI Heart Disease adapter.
- `framework/datasets/audio.py`: SpeechCommands adapter.
- `framework/partitioning/`: shared IID and Dirichlet partitioning.
- `framework/utils/reproducibility.py`: seed and environment manifest helpers.
- `framework/configuration/default_config.yaml`: default dataset and experiment
  infrastructure settings.
- `experiment/experiment2/evaluation.py`: unified regression, ranking, and
  permutation-test metrics.
- `experiment/experiment2/figures.py`: automatic evaluation SVG generation.
- `experiment/experiment2/reporting.py`: neutral report generation.
- `experiment/data/`: central Project 2 data cache root.
- `outputs/exp1/` and `outputs/exp2/`: experiment output directories.

## Dataset Rules

All datasets must be loaded through `DatasetFactory`.

Default behavior is strict:

- real datasets only
- no automatic synthetic fallback
- no automatic download
- fail before training when a required dataset is absent

Explicit download for a future rerun:

```powershell
python project-2-domain-aware-allocation/experiment/experiment1/run.py --download-datasets
```

Explicit synthetic audit for a supported task:

```powershell
python project-2-domain-aware-allocation/experiment/experiment1/run.py --synthetic-datasets AGNews-LSTM
```

## Validation And Provenance

The factory validates each dataset before any training begins:

- dataset exists in cache unless downloads were explicitly requested
- loaded object is a `torch.utils.data.Dataset`
- test loader is a `torch.utils.data.DataLoader`
- expected class count matches
- expected sample counts match when known
- synthetic and download status are recorded

Each run writes `dataset_manifest.json` to the experiment output directory.
This manifest is part of the reproducibility record and should be preserved
with the run outputs.

## Reproducibility

Experiment runners should use a single seed and record:

- partition strategy, alpha, and seed
- client batch sizes
- selected tasks
- dataset manifest
- package and platform versions
- output file inventory

Seed setup is centralized in `framework/utils/reproducibility.py`.

## Experiment 2 Evaluation

The unified evaluation pipeline computes:

- regression metrics: RMSE, MAE, R squared, Pearson
- ranking metrics: Spearman, pairwise ranking accuracy, Kendall tau
- statistical test: one-sided permutation p-value for positive Spearman
  association

Ranking metrics are not used to choose Form A or Form B during Phase 2A. Ridge
alpha selection remains based on minimum mean leave-one-task-out RMSE.

The default alpha grid remains `[0.01, 0.1, 1.0, 10.0, 100.0]`. Future reruns
can opt into prepared larger candidates with:

```powershell
python project-2-domain-aware-allocation/experiment/experiment2/run.py --include-extended-ridge-alphas
```

## Legacy CIFAR-100 Scaffold

`framework/datasets/cifar100_domains.py` remains available for domain-split
CIFAR-100 utilities. It now obtains CIFAR-100 through `DatasetFactory`, so it
inherits the same real-data, cache, download, and validation policy as the
active experiments.
