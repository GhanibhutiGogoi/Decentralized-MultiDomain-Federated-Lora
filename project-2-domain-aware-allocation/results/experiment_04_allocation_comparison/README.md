# Experiment 04: Allocation Strategy Comparison

## Purpose

This experiment compares different LoRA rank allocation strategies in heterogeneous federated learning.

The compared strategies are:

1. **Uniform allocation**
   All clients receive the same LoRA rank.

2. **Random allocation**
   Each client receives a randomly selected LoRA rank.

3. **Domain-Aware allocation**
   LoRA rank is allocated based on the estimated domain complexity score.

4. **Dynamic Allocation**
   Gabriel-style adaptive rank assignments are connected with the Project 2 alpha/scaling policy.

The Dynamic Allocation pipeline is:

```text
client_i → rank r_i → alpha α_i → scaling α_i / r_i
```

## Dynamic Allocation Equation

For each client `i`, the intended full dynamic allocation equation is:

```text
r_i = GabrielRankPolicy(s(G_i), batch_i)

α_i = AlphaPolicy(r_i, C_i)

scaling_i = α_i / r_i
```

The final allocation for each client is:

```text
client_i → {rank: r_i, alpha: α_i, scaling: α_i / r_i}
```

## Current Implementation

In the current implementation, Dynamic Allocation uses:

* a temporary Gabriel-style rank pattern: `4 / 8 / 16`
* global alpha: `α = 64`
* effective LoRA scaling: `scaling = α / rank`

This version is the first integration step between Gabriel's adaptive rank-selection direction and the Project 2 alpha/scaling policy.

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
* alpha value
* allocation metadata
* strategy metadata

## Interpretation

## Interpretation

The results should be interpreted carefully.

This experiment shows that the unified dynamic allocation policy can be integrated into the Experiment 04 allocation comparison pipeline and evaluated alongside Uniform, Random, and Domain-Aware allocation.

However, the current Experiment 04 numbers are near chance-level for this setup and should not be treated as a valid allocation-strategy comparison yet. The runs are undertrained / short-run evaluations, with average accuracies around 1–1.8%, so differences between strategies are not interpretable as evidence that one allocation policy is better.

The apparent lead of Domain-Aware allocation is also confounded with total rank / capacity. In the current results, Domain-Aware uses the largest total rank budget, while Dynamic Allocation, Uniform, and Random use different total-rank budgets. Therefore, the observed ordering may reflect capacity differences rather than allocation smartness. In particular, the current result should not be quoted as “Domain-Aware is the best allocation strategy”; a fair comparison requires multi-seed evaluation under matched parameter or communication budgets.

This version should therefore be treated as a first integration step rather than final paper-level RQ4 evidence.

## Future Work

Future work should:

1. Replace the temporary Gabriel-style `4 / 8 / 16` rank pattern with Gabriel's actual per-client adaptive rank output.
2. Run multi-seed evaluation.
3. Match communication or parameter budgets across allocation strategies.
4. Check whether performance differences come from the allocation policy itself or mainly from rank diversity / total-rank differences.
5. Extend the alpha policy from a global `α = 64` setting to a more adaptive client-specific alpha policy.
