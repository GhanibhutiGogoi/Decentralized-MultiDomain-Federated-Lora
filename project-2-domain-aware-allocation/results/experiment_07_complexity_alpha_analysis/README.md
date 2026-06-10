## Goal

Analyze the relationship between dataset complexity characteristics and the oracle LoRA alpha value.

## Method

For each client, the oracle alpha obtained in Experiment 05 was compared against:

- Complexity Score
- Label Entropy
- Feature Diversity
- Intrinsic Dimension
- Data Imbalance
- Number of Samples
- Number of Classes

Pearson, Spearman, and Kendall correlations were computed.

## Main Findings

The strongest relationships with alpha were:

| Metric | Pearson r |
|----------|----------|
| Complexity Score | -0.478 |
| Data Imbalance | 0.455 |
| Entropy | -0.444 |

Feature Diversity and Number of Classes showed almost no relationship with alpha.

## Conclusion

The results suggest that alpha may be more sensitive to dataset characteristics than LoRA rank.

Although statistical significance was not reached due to the small number of clients (n=15), complexity score, entropy, and data imbalance appear to be promising signals for future alpha allocation policies.
