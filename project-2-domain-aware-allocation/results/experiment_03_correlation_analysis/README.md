# Experiment 03: Correlation Analysis

## Problem

The goal of this experiment is to determine whether the complexity score developed in Experiment 01 can predict the oracle rank discovered in Experiment 02.

If a strong relationship exists between complexity and oracle rank, then complexity scores could be used to automatically allocate LoRA ranks without performing expensive oracle searches.

## Approach

We compared each client's complexity score against its oracle rank obtained from Experiment 02.

Multiple correlation measures were evaluated:

- Pearson correlation
- Spearman rank correlation
- Kendall rank correlation

We also analyzed the contribution of each individual complexity component:

- Entropy
- Diversity
- Intrinsic Dimension
- Task Difficulty
- Data Imbalance

## Results

Generated files:

- correlation_analysis.txt
- correlation_scatter.png
- correlation_summary.json

Main correlation results:

- Pearson r = 0.2624 (p = 0.3449)
- Spearman ρ = -0.0349 (p = 0.9017)
- Kendall τ = -0.0294 (p = 0.8961)
- R² = 0.0688

Sub-metric analysis:

| Metric | Spearman ρ | p-value |
|----------|----------|----------|
| Entropy | 0.2443 | 0.3803 |
| Diversity | 0.0000 | 1.0000 |
| Intrinsic Dimension | 0.3531 | 0.1968 |
| Data Imbalance | -0.2791 | 0.3137 |
| Task Difficulty | N/A | N/A |

None of the individual metrics achieved statistically significant correlation with oracle rank. :contentReference[oaicite:0]{index=0}

## Findings

The proposed complexity metric does not successfully predict oracle LoRA rank.

All correlation measures indicate a weak relationship between complexity score and oracle rank, and none of the observed correlations are statistically significant. The strongest individual signal comes from intrinsic dimension (ρ = 0.3531), but it remains insufficient for reliable rank prediction. :contentReference[oaicite:1]{index=1}

These results suggest that the current complexity metric requires further refinement before it can be used for adaptive rank allocation. This motivates the development of improved allocation strategies in later experiments.
