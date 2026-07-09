# Experiment 04: Allocation Strategy Comparison

## Purpose

This experiment compares different LoRA rank allocation strategies in heterogeneous federated learning.

The compared strategies are:

1. **Uniform allocation**  
   All clients receive the same LoRA rank.

2. **Random allocation**  
   Each client receives a randomly selected LoRA rank.

3. **Random Matched-Budget allocation**  
   Clients receive heterogeneous random LoRA ranks while keeping the total rank budget fixed.

4. **Domain-Aware allocation**  
   LoRA rank is allocated based on the estimated domain complexity score.

5. **Dynamic Allocation**  
   Heterogeneous adaptive-rank-style assignments are connected with the Project 2 alpha/scaling policy.

6. **Weighted Assignment**  
   Client/domain signals are converted into normalized weights and used to allocate a fixed total LoRA rank budget.

## Allocation Pipelines

### Dynamic Allocation

The Dynamic Allocation pipeline is:

```text
client_i -> rank r_i -> alpha alpha_i -> scaling alpha_i / r_i
```

For each client `i`, the intended full dynamic allocation equation is:

```text
r_i = AdaptiveRankPolicy(s(G_i), batch_i)

alpha_i = AlphaPolicy(r_i, C_i)

scaling_i = alpha_i / r_i
```

The final allocation for each client is:

```text
client_i -> {rank: r_i, alpha: alpha_i, scaling: alpha_i / r_i}
```

### Weighted Assignment

The Weighted Assignment pipeline is:

```text
client/domain signals -> score_i -> weight_i -> rank_i -> alpha_i -> scaling_i
```

For each client `i`, the weighted assignment score is:

```text
score_i = lambda_c C_i + lambda_e E_i + lambda_b B_i
```

where:

* `C_i` is the client/domain complexity score
* `E_i` is entropy
* `B_i` is data imbalance
* `lambda_c`, `lambda_e`, and `lambda_b` are weighting coefficients

The score is normalized into a weight:

```text
w_i = score_i / sum_j score_j
```

Then a fixed total-rank budget is allocated:

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

## Current Implementation

In the current implementation:

* global alpha is fixed at `alpha = 64`
* Uniform allocation uses rank 16 for all clients
* Random Matched-Budget uses heterogeneous random ranks with total rank fixed at 240
* Weighted Assignment uses a fixed total-rank budget of 240
* Dynamic Allocation uses a heterogeneous `4 / 8 / 16` adaptive-rank-style pattern
* Domain-Aware allocation is kept as the original complexity-based baseline

The matched-budget reference is:

```text
15 clients x rank 16 = total rank 240
```

This allows Uniform, Random Matched-Budget, and Weighted Assignment to be compared under the same total-rank capacity.

## Outputs

This experiment generates:

* `allocation_comparison.json`
* `allocation_convergence.png`
* `allocation_per_domain.png`

The JSON file includes:

* final average accuracy
* per-domain accuracy
* fairness metrics
* rank assignments
* total rank
* alpha value
* allocation metadata
* strategy metadata

## Results Summary

The current short-run results are:

| Strategy | Average Accuracy | Fairness Gap | Total Rank |
|---|---:|---:|---:|
| Uniform (r=16) | 0.0102 | 0.0055 | 240 |
| Random | 0.0150 | 0.0405 | 388 |
| Random Matched-Budget | 0.0087 | 0.0155 | 240 |
| Domain-Aware | 0.0183 | 0.0155 | 480 |
| Dynamic Allocation | 0.0118 | 0.0050 | 140 |
| Weighted Assignment | 0.0102 | 0.0055 | 240 |

## Interpretation

The results should be interpreted carefully.

This experiment extends the Experiment 04 allocation comparison pipeline by adding two controlled strategies: Random Matched-Budget and Weighted Assignment.

The current numbers are near chance-level for this setup, with average accuracies around 1-1.8%. Therefore, these short-run results should not be treated as final evidence that one allocation strategy is better than another.

Domain-Aware allocation achieves the highest short-run accuracy, but it also uses the largest total-rank budget:

```text
Domain-Aware total rank = 480
```

This is double the matched-budget reference of 240. Therefore, the Domain-Aware result should not be quoted as a fair allocation-policy advantage. The observed gain may come from larger total capacity rather than from the allocation rule itself.

Weighted Assignment is evaluated under the matched total-rank budget:

```text
Weighted Assignment total rank = 240
```

In this short-run evaluation, Weighted Assignment produces the same final average accuracy and fairness gap as Uniform allocation. This suggests that the current weighted formula and rank-rounding step may be too conservative, or that the current client/domain signals do not yet create enough useful rank diversity after mapping to allowed LoRA ranks.

Random Matched-Budget is included as a control baseline. Its purpose is to test whether heterogeneous rank diversity alone explains performance differences when the total-rank budget is fixed.

Overall, this version should be treated as a controlled integration step rather than final paper-level RQ4 evidence.

## Future Work

Future work should:

1. Refine the Weighted Assignment formula so it creates useful client-level rank diversity under the matched total-rank budget.
2. Run multi-seed evaluation.
3. Match communication or parameter budgets across all allocation strategies.
4. Check whether performance differences come from the allocation policy itself or mainly from rank diversity / total-rank differences.
5. Extend the alpha policy from a global `alpha = 64` setting to a more adaptive client-specific alpha policy.
6. Replace the current adaptive-rank-style pattern with actual per-client adaptive rank outputs when available.
