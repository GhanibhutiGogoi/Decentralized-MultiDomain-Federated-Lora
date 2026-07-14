# Experiment 10: Weighted Assignment Policy

## Purpose

This experiment defines the first weighted assignment formula for Project 2: Domain-Aware LoRA Allocation for Heterogeneous Federated Learning.

The goal is to map client/domain signals into normalized assignment weights and use those weights to distribute a fixed LoRA rank budget across clients.

This experiment is a formula / policy validation step, not a final performance comparison.

## Relation to Gabriel's Adaptive Rank Work

This experiment should be interpreted as the Project 2 allocation-side formula, not as a replacement for Gabriel's adaptive rank work.

Gabriel's Project 1 direction focuses on adaptive LoRA rank selection. In other words, Gabriel's side studies how to choose or estimate the client-level rank using adaptive-rank signals such as stable rank, capability constraints, and hardware-aware ceilings.

Project 2 focuses on the allocation side. Given client/domain signals and a controlled total-rank budget, this experiment defines how to convert those signals into normalized allocation weights and then assign rank, alpha, and scaling values.

The intended joint pipeline is:

```text
Gabriel adaptive rank signal / rank output
-> Project 2 weighted assignment
-> alpha / scaling allocation
-> matched-budget Experiment 04 evaluation
```

In this first version, the weighted assignment formula is validated independently. The next step is to connect it more explicitly with Gabriel's actual adaptive-rank output or agreed rank-output format.

## Weighted Assignment Formula

For each client `i`, we compute a client score:

```text
score_i = lambda_c C_i + lambda_e E_i + lambda_b B_i
```

where:

* `C_i` is the client/domain complexity score
* `E_i` is entropy
* `B_i` is data imbalance
* `lambda_c`, `lambda_e`, and `lambda_b` are weighting coefficients

The score is normalized into a client weight:

```text
w_i = score_i / sum_j score_j
```

Then a fixed total rank budget is allocated:

```text
raw_rank_i = r_min + w_i * (B_total - N * r_min)
```

The continuous rank is mapped to the nearest allowed LoRA rank:

```text
r_i = nearest_allowed_rank(raw_rank_i)
```

Finally, LoRA scaling is computed as:

```text
scaling_i = alpha_i / r_i
```

The final allocation is:

```text
client_i -> {score_i, weight_i, rank_i, alpha_i, scaling_i}
```

## Current Implementation

This first version uses the client table from Experiment 08 as input because it already contains:

* complexity score
* entropy
* data imbalance
* client/domain metadata

The current fixed total-rank budget is:

```text
B_total = 240
```

This matches the total rank of a uniform rank-16 allocation over 15 clients:

```text
15 clients x rank 16 = 240
```

The current alpha value is:

```text
alpha = 64
```

The current allowed LoRA rank ladder is:

```text
[4, 8, 16, 32, 64]
```

## Default Run Outcome: Uniform Collapse

The default run uses the real Experiment 08 client/domain signals and a fixed total-rank budget of 240.

In this setting, the weighted assignment policy collapses to uniform rank-16 allocation:

```text
rank 16: 15 clients
```

This happens because the client signals are not sufficiently dispersed across the 15 shards. The normalized weights are all close to `1/15`, so the raw ranks land close to rank 16. With the allowed LoRA rank ladder:

```text
[4, 8, 16, 32, 64]
```

all raw ranks round to 16.

Therefore, the default run validates the arithmetic and budget-matching behavior of the formula, but it does not demonstrate heterogeneous rank allocation.

## Heterogeneity Probe

To verify that the formula can produce heterogeneous assignments, this experiment also includes a heterogeneity probe.

The probe amplifies the spread of the same client/domain signals before applying the weighted assignment formula.

This is not intended as a final training configuration. It is only a formula stress test showing that when client/domain signals are sufficiently dispersed, the policy can assign different ranks under the same rank ladder.

This exposes an important limitation of the current setup: with a coarse rank ladder and weakly varying client signals, weighted assignment may collapse to uniform allocation.

## Rank-Ladder Granularity Limitation

The current allowed rank ladder is:

```text
[4, 8, 16, 32, 64]
```

Because the ladder is coarse, small differences in client weights may not change the final assigned rank. In the default 240-budget setting, raw ranks fall near 16, so all clients round to rank 16.

Future versions should either:

1. improve the signal spread / predictor quality,
2. use a budget or rank ladder that better exposes allocation differences,
3. or connect the policy with Gabriel's adaptive-rank output or agreed rank-output format.

## Outputs

Generated files:

* `weighted_assignment_results.json`

The output JSON includes:

* default matched-budget run results
* heterogeneity probe results
* rank counts
* raw rank ranges
* the weighted assignment formula
* hyperparameters
* total rank budget
* actual total rank
* budget error
* client-level scores
* normalized weights
* assigned ranks
* alpha values
* scaling values

## Interpretation

This experiment provides the first explicit implementation of a weighted assignment formula for Project 2.

The default real-signal run should be interpreted carefully. It validates the formula arithmetic and budget-matching behavior, but it collapses to uniform rank-16 allocation under the current data and budget.

The heterogeneity probe shows that the formula can produce non-uniform rank assignments when the input signals are sufficiently dispersed. However, this probe is not a final performance configuration and should not be treated as evidence of improved training accuracy.

Because the current complexity signal was previously found to be a weak predictor of oracle rank, this weighted assignment should be treated as an allocation heuristic rather than a validated oracle-rank predictor.

## Next Steps

Future work should integrate this weighted assignment policy into Experiment 04 and evaluate it against:

* Uniform allocation
* Random heterogeneous matched-budget allocation
* Domain-Aware allocation
* Dynamic Allocation
* Gabriel adaptive-rank output or agreed rank-output format

The key evaluation question is whether weighted assignment improves performance beyond simple heterogeneous rank diversity under matched rank / communication budgets.
