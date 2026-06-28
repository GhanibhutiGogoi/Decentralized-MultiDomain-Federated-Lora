# Experiment 09: Dynamic Allocation Policy

## Problem

Project 1 provides adaptive LoRA rank selection, while Project 2 studies domain-aware alpha and scaling allocation.

Previously, rank selection and alpha/scaling analysis were treated separately. However, LoRA allocation should be defined jointly because the effective update magnitude depends on both rank and alpha through the scaling factor:

scaling = alpha / rank

This experiment creates the first unified dynamic allocation policy connecting Gabriel's adaptive rank-selection output with the Project 2 alpha/scaling policy.

## Approach

For each client, the allocation pipeline is:

client_i → rank r_i → alpha α_i → scaling α_i / r_i

The rank is provided by Gabriel's Project 1 adaptive rank-selection policy:

r_i = GabrielRankPolicy(s(G_i), batch_i)

where:

- s(G_i) is the stable rank of the client's gradient signal.
- batch_i is the client's batch size / capability proxy.
- R_i^max is the hardware-derived maximum rank.

Gabriel's configuration maps batch size to maximum rank:

- batch size 16 → max rank 4
- batch size 64 → max rank 8
- batch size 256 → max rank 16

For the first version of this experiment, alpha is set to a global value:

α_i = 64

This is based on the previous alpha-policy analysis, where alpha 64 was the best global alpha.

The final effective scaling is:

scaling_i = α_i / r_i

## Results

Generated file:

- dynamic_allocation_results.json

The first version produces the following allocation structure for each client:

client_i → rank, alpha, scaling

Example:

- client_0 → rank 4, alpha 64, scaling 16.0
- client_1 → rank 8, alpha 64, scaling 8.0
- client_2 → rank 16, alpha 64, scaling 4.0

## Findings

This experiment establishes the first full dynamic allocation pipeline connecting adaptive rank selection with alpha/scaling allocation.

The current version uses Gabriel's selected rank as the rank component and a global alpha of 64 as the alpha component. This gives a complete LoRA allocation for each client.

This is an implementation bridge rather than a final performance experiment. Future work should replace the sample rank assignments with Gabriel's actual rank output file and evaluate the dynamic allocation policy against uniform allocation and oracle-based baselines.

## Next Steps

- Integrate Gabriel's actual rank-selection result file.
- Extend alpha selection from fixed global alpha to rank-aware or complexity-aware alpha.
- Re-run Experiment 04 using the unified dynamic allocation policy.
- Compare against uniform allocation and oracle-based baselines under matched experimental settings.
