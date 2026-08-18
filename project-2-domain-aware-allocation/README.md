# Project 2: Domain-Aware Allocation

Project 2 extends the validated Project 1 federated LoRA baseline. Experiment 1
is implemented under `experiment1/` and reuses Project 1 runtime modules
directly rather than copying or redefining their mathematics.

No lambda allocation logic is implemented.

## Status

Experiment 1 support is available:

- domain-heterogeneous Dirichlet label partitioning
- Project 1-compatible IID partitioning
- label-distribution signal logging
- update dissimilarity logging
- leave-one-client-out marginal contribution measurement
- correlation and controlled-regression analysis
- publication-oriented figures
- outputs stored under `outputs/exp1/`

## Structure

- `framework/`
- `framework/datasets/`
- `framework/models/`
- `framework/federated/`
- `framework/aggregation/`
- `framework/rank_allocation/`
- `framework/partitioning/`
- `framework/analysis/`
- `framework/visualization/`
- `framework/configuration/`
- `framework/utils/`
- `experiment/`
- `experiment/data/`
- `experiment/experiment1/`
- `experiment/experiment2/`
- `experiment/experiment3/`
- `outputs/`
- `outputs/exp1/`
- `outputs/exp2/`
- `outputs/exp3/`

See `EXPERIMENT1.md` for run commands, architecture notes, and output schemas.
