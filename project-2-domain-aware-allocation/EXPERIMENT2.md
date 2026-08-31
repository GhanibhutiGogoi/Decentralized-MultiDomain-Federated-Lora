# Experiment 2: Lambda Calibration Evaluation Infrastructure

Experiment 2 calibrates candidate lambda forms from Experiment 1 measurement
tables. During Phase 2A, the repository prepares evaluation infrastructure only;
it does not rerun Experiment 2 and does not change scientific conclusions.

## Inputs

Experiment 2 reads from `project-2-domain-aware-allocation/outputs/exp1/` by
default:

- `per_round_client_measurements.csv`
- `signal_contribution_correlations.csv`
- `controlled_regression.csv`
- `label_distribution_summary.csv`
- `manifest.json`
- `dataset_manifest.json`

The dataset manifest is required so Experiment 2 can preserve dataset
provenance from Experiment 1. Synthetic source measurements are rejected by
default.

The `is_synthetic` column in Experiment 1 measurements is required. Experiment
2 raises an error if the column is missing, contains nulls, or contains
unrecognized values.

## Evaluation Pipeline

Reusable evaluation code lives in:

- `experiment/experiment2/evaluation.py`
- `experiment/experiment2/figures.py`
- `experiment/experiment2/reporting.py`

The pipeline computes:

- RMSE
- MAE
- R squared
- Pearson correlation
- Spearman rank correlation
- pairwise ranking accuracy
- Kendall tau
- one-sided permutation p-value for positive Spearman association

Ranking metrics are reported only. They do not select Form A or Form B.

## Ridge Alpha Search

The default Ridge alpha grid remains:

```text
0.01, 0.1, 1.0, 10.0, 100.0
```

Prepared larger candidates are available for a future rerun:

```text
300.0, 500.0, 1000.0
```

Enable them explicitly:

```powershell
python project-2-domain-aware-allocation/experiment/experiment2/run.py --include-extended-ridge-alphas
```

Or provide a custom grid:

```powershell
python project-2-domain-aware-allocation/experiment/experiment2/run.py --ridge-alphas 0.01 0.1 1 10 100 300 500 1000
```

The alpha selection rule remains minimum mean leave-one-task-out RMSE.

If the selected alpha equals the minimum or maximum tested value, Experiment 2
emits a `RidgeAlphaBoundaryWarning`. The warning states that the optimum may lie
outside the tested range. The code does not automatically expand the search.

## Class Imbalance Input

Experiment 2 consumes `class_imbalance_ratio` from Experiment 1 and applies the
existing `log1p` feature transform. Experiment 1 now computes that ratio as:

```text
max_count / min_positive_count * (1 + zero_class_count / num_classes)
```

This avoids EPS-driven extreme values while preserving the fact that missing
classes indicate stronger imbalance.

## Future Outputs

When Experiment 2 is rerun after the Project 1 mathematical review, it will
produce:

- `lambda_values.csv`
- `lambda_validation.csv`
- `orthogonality_report.csv`
- `cross_validation.csv`
- `fitted_coefficients.csv`
- `evaluation_metrics.csv`
- `alpha_evaluation.csv`
- `ranking_significance.csv`
- `dataset_manifest.json`
- `manifest.json`
- `comparison_report.md`
- summary SVG figures under `figures/`

Do not generate these outputs during Phase 2A.
