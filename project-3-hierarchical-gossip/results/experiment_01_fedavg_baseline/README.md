# Experiment 01: FedAvg Baseline

## Problem

Before evaluating adaptive LoRA rank allocation strategies, we need a baseline federated learning method for comparison. This experiment establishes a standard FedAvg baseline in a heterogeneous multi-domain federated learning setting.

## Approach

We trained a LoRA-ResNet18 model using the Federated Averaging (FedAvg) algorithm on CIFAR-100.

Experimental setup:

* Dataset: CIFAR-100
* Domains: 5
* Clients per Domain: 3
* Total Clients: 15
* Dirichlet α: 0.5
* LoRA Rank: 16
* LoRA Alpha: 32
* Local Epochs: 5
* Global Rounds: 2
* Optimizer: Adam
* Learning Rate: 0.001

Performance was evaluated using average accuracy, per-domain accuracy, and fairness metrics.

## Results

Generated files:

* fedavg_history.json
* fedavg_summary.json
* fedavg_convergence.png
* fedavg_per_domain.png
* fedavg_final_per_domain.png
* fedavg_summary_bars.png

Overall performance:

* Final Accuracy: 0.0179
* Best Accuracy: 0.0179

Per-domain accuracy:

| Domain   | Accuracy |
| -------- | -------- |
| Domain 0 | 0.0030   |
| Domain 1 | 0.0455   |
| Domain 2 | 0.0185   |
| Domain 3 | 0.0085   |
| Domain 4 | 0.0140   |

Fairness metrics:

* Accuracy Gap: 0.0425
* Accuracy Variance: 0.000218
* Minimum Domain Accuracy: 0.0030
* Maximum Domain Accuracy: 0.0455

## Findings

## Findings

This experiment should be interpreted as a 2-round pipeline-validation run rather than a full research result.

The purpose of this run was to verify that the FedAvg pipeline, data loading, client training, aggregation, evaluation, and result logging work end-to-end.

Because only 2 communication rounds were executed, the reported accuracy values are not meaningful for algorithmic comparison or convergence analysis.

Full-length experiments with 50+ communication rounds are required before drawing conclusions about FedAvg performance in this setting.
