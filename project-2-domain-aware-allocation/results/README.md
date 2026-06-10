# Experiment 01: Data and Complexity Analysis

## Problem

Before designing adaptive LoRA rank allocation, we need to understand whether different clients exhibit different levels of data complexity and whether the federated dataset is heterogeneous.

## Approach

The CIFAR-100 dataset was partitioned into 5 domains with 3 clients per domain (15 clients total). For each client, a composite complexity score was computed using five components:

* Label Entropy (weight = 0.3)
* Feature Diversity (weight = 0.2)
* Intrinsic Dimension (weight = 0.2)
* Task Difficulty (weight = 0.2)
* Data Imbalance (weight = 0.1)

We also analyzed the number of samples assigned to each client and visualized the contribution of each complexity component.

## Results

Generated files:

* data_distribution.png
* complexity_breakdown.png
* complexity_scores.json

Complexity scores ranged from 0.6004 to 0.6224 across the 15 clients.

Highest complexity:

* Client 10: 0.6224

Lowest complexity:

* Client 4: 0.6004

Client sample counts ranged from 2187 to 5210 samples.

## Findings

The federated dataset exhibits heterogeneous data distributions across clients, with substantial differences in sample counts and class distributions.

However, the overall complexity scores are relatively close to each other, indicating that the current complexity metric provides only limited separation between clients. In particular, the task difficulty component remained constant across all clients and therefore did not contribute to distinguishing client complexity.

These observations motivate further refinement of the complexity metric, which is investigated in later experiments through oracle rank search and correlation analysis.

