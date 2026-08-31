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

## Remaining Work After Mathematical Review

After Project 1 mathematics are finalized, rerun Experiment 1 exactly once with
the approved configuration, then rerun Experiment 2 from the new Experiment 1
outputs. Any downstream Experiment 3 work should consume those final outputs
rather than historical artifacts.
