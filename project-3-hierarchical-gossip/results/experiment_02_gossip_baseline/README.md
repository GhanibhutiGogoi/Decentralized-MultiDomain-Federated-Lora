# Experiment 02: Gossip Baseline

## Problem

While FedAvg relies on a centralized server to aggregate model updates, decentralized federated learning removes the central server and allows clients to communicate directly. This experiment evaluates a Gossip-based baseline to understand whether decentralized communication can improve performance in a heterogeneous multi-domain federated learning setting.

## Approach

We implemented a decentralized Gossip learning framework using a ring topology.

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
* Topology: Ring
* Average Degree: 2
* Total Messages Exchanged: 30

Instead of communicating with a central server, each client exchanges model information only with its neighboring clients in the communication graph.

## Results

Generated files:

* gossip_history.json
* gossip_summary.json
* gossip_convergence.png
* gossip_per_domain.png
* gossip_final_per_domain.png
* gossip_summary_bars.png

Overall performance:

* Final Accuracy: 0.1583
* Best Accuracy: 0.1583

Per-domain accuracy:

| Domain   | Accuracy |
| -------- | -------- |
| Domain 0 | 0.1385   |
| Domain 1 | 0.2035   |
| Domain 2 | 0.1110   |
| Domain 3 | 0.1415   |
| Domain 4 | 0.1970   |

Fairness metrics:

* Accuracy Gap: 0.0925
* Accuracy Variance: 0.001289
* Minimum Domain Accuracy: 0.1110
* Maximum Domain Accuracy: 0.2035

Training progress:

* Round 0 Accuracy: 0.0846
* Round 1 Accuracy: 0.1583

## Findings

This experiment should be interpreted as a 2-round pipeline-validation run rather than a full research result.

The purpose of this run was to verify that the Gossip pipeline, client-to-client communication, decentralized update flow, evaluation, and result logging work end-to-end.

The reported accuracy values should not be used to claim that Gossip outperforms FedAvg. FedAvg and Gossip are evaluated under different protocols in these smoke tests: FedAvg evaluates one global model, while Gossip evaluates personalized client models.

Therefore, the FedAvg and Gossip accuracy numbers are not directly comparable in this setting.

Full-length experiments with 50+ communication rounds and aligned evaluation protocols are required before drawing conclusions about relative algorithm performance.
