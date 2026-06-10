# Experiment 06: Alpha Policy Analysis

## Problem

Experiment 05 identified the oracle alpha value for each client. However, using client-specific oracle alpha values is impractical in real federated learning systems because oracle information is unavailable during deployment.

The goal of this experiment is to analyze the alpha-search results and derive simple alpha allocation policies that can be applied without performing exhaustive alpha search.

## Approach

Using the results from Experiment 05, we analyzed:

* Global alpha performance across all clients
* Alpha distribution among clients
* Best alpha for different LoRA ranks
* A tolerance-based policy that selects the smallest alpha whose accuracy is within 0.5% of the oracle solution

The analysis was performed using the `alpha_policy.py` module.

## Results

Generated files:

* alpha_policy_report.json

### Global Alpha Analysis

Average accuracy across all clients:

| Alpha | Average Accuracy |
| ----- | ---------------- |
| 4     | 0.1923           |
| 8     | 0.2243           |
| 16    | 0.2462           |
| 32    | 0.2615           |
| 64    | 0.2651           |
| 128   | 0.2644           |

Best global alpha:

* Alpha = 64
* Average Accuracy = 0.2651

### Alpha Distribution

Best alpha selected by clients:

| Alpha | Number of Clients |
| ----- | ----------------- |
| 32    | 5                 |
| 64    | 5                 |
| 128   | 5                 |

### Rank-Specific Analysis

| Rank | Best Alpha | Average Accuracy |
| ---- | ---------- | ---------------- |
| 32   | 64         | 0.2648           |
| 64   | 64         | 0.2661           |

### Tolerance-Based Policy

Using a 0.5% accuracy tolerance, several clients can reduce alpha from 128 to 64 or 32 with negligible performance loss.

Examples:

* Client 0: 128 → 64 (accuracy drop = 0.0045)
* Client 1: 128 → 64 (accuracy drop = 0.0030)
* Client 7: 128 → 32 (accuracy drop = 0.0045)

## Findings

The best global alpha across all clients is 64.

Interestingly, both rank-32 and rank-64 clients achieve their highest average accuracy with alpha 64, suggesting that alpha selection may be largely independent of rank in this setting.

Furthermore, a simple policy that selects the smallest alpha within 0.5% accuracy of the oracle solution frequently reduces alpha from 128 to 64 or 32 with negligible performance loss.

These results indicate that moderate alpha values can provide nearly the same performance as larger alpha values while avoiding unnecessary scaling. Therefore, alpha = 64 represents a strong default choice for future experiments and adaptive LoRA allocation strategies.
