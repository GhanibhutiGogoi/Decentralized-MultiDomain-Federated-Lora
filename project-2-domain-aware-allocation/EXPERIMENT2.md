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
