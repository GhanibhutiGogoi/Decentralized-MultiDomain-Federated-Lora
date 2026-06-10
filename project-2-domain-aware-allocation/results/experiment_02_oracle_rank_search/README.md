# Experiment 02: Oracle Rank Search

## Problem

The complexity score alone does not directly tell us which LoRA rank should be assigned to a client. Therefore, we perform an oracle rank search to identify the rank that achieves the highest accuracy for each client.

## Approach

For each of the 15 clients, LoRA models were trained using candidate ranks:

- 4
- 8
- 16
- 32
- 64

All other training settings were kept fixed. The final test accuracy was recorded for every client-rank combination. The rank with the highest accuracy was selected as the oracle rank for that client.

## Results

Generated files:

- oracle_results.json
- oracle_heatmap.png

Oracle rank distribution:

- Rank 32 selected for 11 clients
- Rank 64 selected for 4 clients
- Rank 4, 8, and 16 were never selected as oracle ranks

Best-performing client:

- Client 5 achieved 35.74% accuracy with rank 32

Lowest-performing client:

- Client 9 achieved 17.54% accuracy with rank 32

In most clients, accuracy increased consistently as rank increased from 4 to 32. Further increasing rank from 32 to 64 produced only small improvements and sometimes reduced performance.

## Findings

The results indicate that larger LoRA ranks generally improve performance, but the relationship is not strictly monotonic.

Most clients achieved their best performance with rank 32, suggesting that rank 32 provides a good balance between model capacity and efficiency. Only four clients benefited from increasing rank to 64.

These oracle ranks serve as the target labels for later experiments. In particular, Experiment 03 investigates whether the complexity metric can predict these oracle ranks, while Experiment 04 evaluates domain-aware rank allocation strategies using the oracle results as a reference.
