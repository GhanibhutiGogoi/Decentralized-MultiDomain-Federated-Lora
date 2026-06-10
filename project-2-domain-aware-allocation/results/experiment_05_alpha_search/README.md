# Experiment 05: Alpha Search with Fixed Oracle Rank

## Problem

Experiment 02 identified the oracle LoRA rank for each client. However, LoRA performance is influenced not only by rank but also by the scaling parameter alpha.

The objective of this experiment is to determine the optimal alpha value for each client while keeping the oracle rank fixed.

## Approach

For each client, the oracle rank obtained from Experiment 02 was fixed.

The following candidate alpha values were evaluated:

- 4
- 8
- 16
- 32
- 64
- 128

For every client-alpha combination, the model was trained and evaluated under the same experimental settings. The alpha value achieving the highest test accuracy was selected as the oracle alpha for that client.

## Results

Generated files:

- alpha_search_results.json
- best_alpha_per_client.json
- alpha_heatmap.png

Alpha distribution across clients:

| Best Alpha | Number of Clients |
|------------|------------------|
| 32 | 4 |
| 64 | 5 |
| 128 | 6 |

No client selected alpha values 4, 8, or 16 as the best configuration. :contentReference[oaicite:0]{index=0}

Best-performing client:

- Client 5 achieved 36.34% accuracy with rank 32 and alpha 64. :contentReference[oaicite:1]{index=1}

Lowest-performing client:

- Client 9 achieved 17.54% accuracy with rank 32 and alpha 32. :contentReference[oaicite:2]{index=2}

The heatmap shows a clear trend that performance generally improves as alpha increases from 4 to 32 and often continues improving up to 64 or 128. In many cases, the gain between alpha 64 and alpha 128 becomes very small or disappears completely. 

## Findings

The results indicate that larger alpha values are consistently more effective than small alpha values.

Across all 15 clients, the best alpha was always selected from the set {32, 64, 128}, while smaller alpha values never achieved the highest accuracy. :contentReference[oaicite:4]{index=4}

Most clients showed monotonic improvement as alpha increased, suggesting that stronger LoRA update scaling improves adaptation capacity. However, several clients experienced performance saturation beyond alpha 64, indicating diminishing returns at very large alpha values. 

These findings suggest that alpha is an important hyperparameter and should be considered together with rank when designing adaptive LoRA allocation strategies. The experiment provides oracle alpha labels that can be used in future work for alpha prediction or adaptive alpha allocation.
