# Project 2 Infrastructure Plan

## Scope

This branch is in an infrastructure hardening phase before the final Project 2
Experiment 1 and Experiment 2 rerun. The goal is reproducible, robust execution
on real datasets only by default.

Out of scope until the Project 1 mathematical review is complete:

- Project 1 mathematical formulation changes
- lambda formulation changes
- gamma calibration changes
- adaptive rank logic changes
- federated aggregation changes
- experiment reruns or output regeneration

## Current Architecture

Project 2 keeps research runners under `experiment/` and reusable
infrastructure under `framework/`.

Dataset loading is centralized in `framework/datasets/factory.py`. Modality
files such as `image.py`, `text.py`, `tabular.py`, and `audio.py` are adapter
modules used by the factory, not experiment entry points.

Experiment 1 reuses Project 1 runtime modules through
`experiment/experiment1/project1_bridge.py` for model, training, rank, and
aggregation behavior. It uses Project 2 infrastructure for dataset loading,
validation, partitioning, manifest creation, and reproducibility metadata.

Experiment 2 consumes Experiment 1 outputs and now requires Experiment 1's
dataset manifest before calibration.

Experiment 2 evaluation is organized as reusable infrastructure:

- `experiment/experiment2/evaluation.py`: regression metrics, ranking metrics,
  and ranking permutation tests.
- `experiment/experiment2/figures.py`: automatic SVG figure generation for
  regression performance, ranking performance, alpha comparison, and form
  comparison.
- `experiment/experiment2/reporting.py`: neutral report generation from output
  tables.

## Dataset Policy

- Real datasets are the default.
- Synthetic datasets are never selected automatically.
- Synthetic datasets require explicit task configuration.
- Dataset downloads are disabled by default.
- Missing real datasets fail before training starts.
- Dataset validation runs immediately after loading and before training.
- Each experiment writes `dataset_manifest.json` to its output directory.

## Reproducibility Policy

Experiment runners should record:

- dataset manifest and cache provenance
- experiment configuration
- random seed
- partition configuration
- environment and package versions
- output file inventory

Random seeds should be applied through `framework/utils/reproducibility.py`.

## Evaluation Policy

Experiment 2 reports both regression and ranking metrics:

- RMSE
- MAE
- R squared
- Pearson correlation
- Spearman rank correlation
- pairwise ranking accuracy
- Kendall tau
- permutation p-value for positive Spearman association

These metrics are evaluation outputs only. They do not select Form A or Form B
and they do not change the existing Ridge-alpha selection rule, which remains
minimum mean leave-one-task-out RMSE.

The default Ridge alpha grid remains `[0.01, 0.1, 1.0, 10.0, 100.0]`. Larger
prepared candidates `[300.0, 500.0, 1000.0]` can be enabled in a future rerun
through the Experiment 2 CLI without another code change.

## Phase 2A.5 Robustness Fixes

- Class imbalance is finite for missing classes:
  `max_count / min_positive_count * (1 + zero_class_count / num_classes)`.
- Experiment 2 requires explicit `is_synthetic` provenance in measurements and
  refuses to assume missing provenance means real data.
- Ridge alpha selection emits a boundary warning when RMSE selects the minimum
  or maximum tested alpha; the grid is not expanded automatically.

## Remaining Work After Mathematical Review

After Project 1 mathematics are finalized, rerun Experiment 1 exactly once with
the approved configuration, then rerun Experiment 2 from the new Experiment 1
outputs. Any downstream Experiment 3 work should consume those final outputs
rather than historical artifacts.
