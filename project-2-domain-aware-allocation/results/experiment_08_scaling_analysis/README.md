# Experiment 08: Scaling Factor Analysis

## Problem

LoRA uses the scaling factor alpha / rank. Previous experiments analyzed alpha directly, but the effective LoRA update depends on scaling rather than alpha alone.

## Approach

For each client, we combined the oracle rank and oracle alpha:

scaling = alpha / rank

Then we analyzed the relationship between scaling and:

- complexity score
- entropy
- data imbalance
- rank
- alpha

## Results

Generated files:

- scaling_analysis.json
- complexity_scaling_scatter.png

Key result:

- Rank vs Scaling:
  - Spearman = -0.601
  - p-value = 0.0178

## Findings

The analysis found a statistically significant negative relationship between oracle rank and LoRA scaling. Higher-rank adapters tend to require smaller scaling factors.

This suggests that alpha should not be selected independently. Instead, alpha should be interpreted together with rank through the effective scaling factor alpha / rank.
