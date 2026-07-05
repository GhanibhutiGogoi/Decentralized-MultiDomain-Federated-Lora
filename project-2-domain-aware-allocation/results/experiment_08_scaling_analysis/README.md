# Experiment 08: Scaling Factor Analysis

## Problem

LoRA uses the effective scaling factor alpha / rank. Previous experiments analyzed alpha directly, but alpha should also be interpreted together with rank because the LoRA update magnitude depends on their ratio.

However, directly correlating rank with alpha / rank can be misleading because rank appears in the denominator of the scaling formula. Therefore, rank-vs-scaling correlation should not be treated as a standalone research conclusion.

## Approach

For each client, we combined the oracle rank and oracle alpha from previous experiments.

We computed:

```text
scaling = alpha / rank

* Then we analyzed the relationship between:
* oracle rank and best alpha
* complexity score and scaling
* entropy and scaling
* data imbalance and scaling
* alpha and scaling

The main direct check is rank vs best alpha, because this tests whether the selected alpha changes with rank without placing rank directly in the denominator.

## Results

Generated files:

* scaling_analysis.json
* complexity_scaling_scatter.png

This experiment is exploratory and should be interpreted with caution because it uses only 15 clients and evaluates multiple correlations.

## Findings

The original rank-vs-scaling correlation is deprecated as a standalone finding because scaling is defined as alpha / rank. Since rank appears directly in the denominator, correlating rank with scaling can produce a negative relationship by construction.

The more meaningful direct analysis is rank vs best alpha. In this setting, the results do not provide strong evidence for a reliable proportional relationship between oracle rank and best alpha.

The main takeaway is therefore limited but still useful: alpha should be interpreted jointly with rank because LoRA uses the effective scaling factor alpha / rank. However, this experiment does not prove that higher-rank adapters require smaller scaling factors.

Because the analysis is based on a small number of clients and multiple correlation tests, the results should be treated as exploratory rather than confirmatory.
```text

