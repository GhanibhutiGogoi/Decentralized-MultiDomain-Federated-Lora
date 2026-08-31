# Experiment 1: Domain Heterogeneity and Marginal Contribution

This implementation extends Project 1 without changing its mathematical
framework. Project 1 runtime modules are reused directly through
`experiment1/project1_bridge.py`; they are not copied into Project 2.

## Project 1 Reuse

The runner activates the Project 1 root on `sys.path` before importing:

- `config`
- `Federated.client`
- `Federated.fedavg_aggregation`
- `Federated.utilities`
- `rank_allocation.LoRa_rank_projection`
- `rank_allocation.rank_selector`
- `Source.Models`

This preserves the existing implementations of adaptive rank selection,
stable-rank computation, capability constraints, quality score, rank
projection, quality-weighted aggregation, SVD projection aggregation, training,
and evaluation.

Dataset loading is Project 2 infrastructure and goes through
`framework.datasets.DatasetFactory`.

No lambda allocation is implemented here.

## Run

From the repository root:

```powershell
python project-2-domain-aware-allocation/experiment/experiment1/run.py --partition dirichlet --alpha 0.5 --seed 42
```

To reproduce Project 1's original client split behavior:

```powershell
python project-2-domain-aware-allocation/experiment/experiment1/run.py --partition project1_iid --seed 42
```

Useful development options:

```powershell
python project-2-domain-aware-allocation/experiment/experiment1/run.py --tasks CIFAR-CNN --num-rounds 1
```

Do not run these commands during the current infrastructure hardening phase.

## Dataset Loading

Experiment 1 loads every selected task through
`framework/datasets/factory.py`.

Defaults:

- `--data-root project-2-domain-aware-allocation/experiment/data`
- real datasets only
- `--download-datasets` disabled
- no automatic synthetic fallback

If a real dataset cache is missing, execution aborts before training. To make a
future rerun download missing real data explicitly:

```powershell
python project-2-domain-aware-allocation/experiment/experiment1/run.py --download-datasets
```

Synthetic datasets are allowed only for fallback-capable tasks and only through
the explicit `--synthetic-datasets` option.

## Dataset Validation

Validation occurs immediately after dataset loading and before any training.
The factory checks:

- dataset object type
- cache status
- train/test sample counts when known
- class count
- synthetic flag
- download status

## Partitioning

Partitioning lives in `framework/partitioning/`. The historical
`experiment/experiment1/partitioning.py` module is now only a compatibility
re-export.

Supported strategies:

- `project1_iid`, `legacy_iid`, `iid`: contiguous split matching Project 1's
  `split_dataset()` behavior, including dropping the final remainder.
- `dirichlet`: label-based Dirichlet partitioning.

Dirichlet parameters:

- `alpha`: concentration parameter. Smaller values create stronger label
  divergence; larger values approach IID.
- `seed`: reproducibility seed.
- `min_client_size`: minimum allowed samples per client.

Hardware heterogeneity remains independent because DataLoader batch sizes still
come from Project 1's `CLIENT_BATCH_SIZES`.

## Label Extraction

The partitioner extracts labels from all five Project 1 benchmark datasets:

- CIFAR-10: `targets`
- FashionMNIST: `targets`
- AGNews: tuple labels in `data`
- Tabular: `y`
- Audio: `synth`, or dataset iteration for loaded SpeechCommands

## Output Schema

Outputs are written to `project-2-domain-aware-allocation/outputs/exp1/` by
default.

### `dataset_manifest.json`

One manifest is written for each run and contains:

- dataset name
- source library
- dataset version when available
- cache location and cache status
- synthetic flag
- train/test/total sample counts
- class count
- download request and download status
- relevant package versions

The top-level `manifest.json` embeds this dataset manifest and records the file
name for downstream experiments.

### `label_distribution_summary.csv`

One row per task/client:

- `task`
- `client_id`
- `num_samples`
- `raw_class_counts`
- `class_frequency`
- `entropy`
- `normalized_entropy`
- `class_imbalance_ratio`
- `kl_to_global`
- `js_to_global`
- `zero_class_count`

`class_imbalance_ratio` is finite even when a client has missing classes:

```text
max_count / min_positive_count * (1 + zero_class_count / num_classes)
```

This preserves the imbalance signal from absent classes without using an
EPS-sized denominator.

### `label_distribution_raw.json`

Machine-readable raw vectors for downstream divergence work:

```json
{
  "CIFAR-CNN": {
    "task": "CIFAR-CNN",
    "num_classes": 10,
    "global_class_counts": [],
    "global_class_frequency": [],
    "client_class_counts": [],
    "client_class_frequency": []
  }
}
```

`client_class_counts` is a clients-by-classes integer matrix. It is the primary
raw input for later divergence computations.

### `per_round_client_measurements.csv`

One row per task/round/client:

- partition metadata: `partition_strategy`, `partition_alpha`, `partition_seed`
- hardware covariate: `hardware_batch_size`
- Project 1 inherited covariates: `adaptive_rank`, `local_loss`,
  `quality_score`
- label signals: `entropy`, `normalized_entropy`,
  `class_imbalance_ratio`, `kl_to_global`, `js_to_global`,
  `zero_class_count`
- update signals: `update_cosine_distance_to_mean`,
  `update_l2_distance_to_mean`, `update_norm`
- contribution outputs: `full_accuracy`, `loo_accuracy`, `delta_accuracy`

`delta_accuracy = full_accuracy - loo_accuracy`.

### Analysis Tables

- `signal_contribution_correlations.csv`: Pearson and Spearman correlations
  between domain/update signals and marginal contribution.
- `controlled_regression.csv`: standardized OLS coefficients for each signal
  while controlling for `adaptive_rank` and `local_loss`.

### Figures

The `figures/` directory contains:

- label distribution heatmaps
- per-client label histograms
- sample-count plots
- signal-vs-marginal-contribution scatter plots

## Extension Points

To add a partition strategy, implement it in `experiment1/partitioning.py` and
register it in `partition_indices()`. The strategy should return one list of
dataset indices per client and must not change Project 1 training, rank, or
aggregation functions.
