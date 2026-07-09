# Experiment 10: Weighted Assignment Policy

## Purpose

This experiment defines the first weighted assignment formula for Project 2: Domain-Aware LoRA Allocation for Heterogeneous Federated Learning.

The goal is to map client/domain signals into normalized assignment weights and use those weights to distribute a fixed LoRA rank budget across clients.

This experiment is a formula / policy validation step, not a final performance comparison.

## Weighted Assignment Formula

For each client `i`, we compute a client score:

score_i = λ_c C_i + λ_e E_i + λ_b B_i

where:

- C_i is the client/domain complexity score
- E_i is entropy
- B_i is data imbalance
- λ_c, λ_e, and λ_b are weighting coefficients

The score is normalized into a client weight:
w_i = score_i / Σ score_j

Then a fixed total rank budget is allocated:
raw_rank_i = r_min + w_i · (B_total - N · r_min)

The continuous rank is mapped to the nearest allowed LoRA rank:
r_i = nearest_allowed_rank(raw_rank_i)

Finally, LoRA scaling is computed as:
scaling_i = α_i / r_i

The final allocation is:
client_i → {score_i, weight_i, rank_i, alpha_i, scaling_i}

## Current Implementation

This first version uses the client table from Experiment 08 as input because it already contains:

- complexity score
- entropy
- data imbalance
- client/domain metadata

The current fixed total-rank budget is:
B_total = 240

This matches the total rank of a uniform rank-16 allocation over 15 clients:
15 clients × rank 16 = 240

The current alpha value is:
α = 64

## Outputs:

Generated files:
- weighted_assignment_results.json

The output JSON includes:
- the weighted assignment formula
- hyperparameters
- total rank budget
- actual total rank
- budget error
- client-level scores
- normalized weights
- assigned ranks
- alpha values
- scaling values

## Interpretation

It should not be interpreted as final evidence that weighted assignment improves accuracy. The purpose is to define the allocation rule and verify that it produces client-level rank, alpha, and scaling assignments under a fixed total-rank budget.

Because the current complexity signal was previously found to be a weak predictor of oracle rank, this weighted assignment should be treated as an allocation heuristic rather than a validated oracle-rank predictor.

## Next Steps

Future work should integrate this weighted assignment policy into Experiment 04 and evaluate it against:

- Uniform allocation
- Random heterogeneous matched-budget allocation
- Domain-Aware allocation
- Dynamic Allocation

The key evaluation question is whether weighted assignment improves performance beyond simple heterogeneous rank diversity under matched rank / communication budgets.
