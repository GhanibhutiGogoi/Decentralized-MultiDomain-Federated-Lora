# P1 adaptive-rank artifacts

This directory contains two adaptive-rank results from the documented GPU host, which must be kept distinct:

1. The completed five-task battery of the **original** controller against a fixed rank-32 reference (historical; gate failed; files listed below).
2. `adaptive_vs_matched_fashion.json`: the **revised** controller (two-round ceiling warm-up, half-capability floor, relative quality-drop restore) against a **capability-matched** fixed baseline at ranks [4, 8, 16] on Fashion-MNIST, 5 rounds, seed 42. Both finish at 82.85%; the adaptive run uses 10.0% fewer FLOPs, with a transient round-4 dip. One task, one seed: parity with a feasible baseline, not five-task parity.

Required files:

- `manifest.json`: command, source commit, environment, datasets, seeds and candidate-rank/capability settings.
- `summary.csv`: one row per task/seed/method with final accuracy, FLOPs, factor payload, average and selected ranks, number of rank changes, and budget residual/violations.
- `rank_history.csv`: per-client and per-round stable rank, EMA demand, selected rank, residual ratio and tail mass.
- `fig1_accuracy_curves.png`, `fig2_adaptive_rank_per_client.png`, `fig3_final_accuracy_bar.png`, `fig4_total_flops_per_round.png`, `fig5_flops_per_client.png`, `fig6_pareto_accuracy_flops.png`: rank trajectories and accuracy/FLOPs comparisons generated from the run.

Evaluate the preregistered gate (within one percentage point of oracle accuracy, at least 25% FLOP savings, >1 rank on at least two tasks, zero budget violations). The original controller failed it: rank 2 for every client and round, 93.75% FLOP savings against rank 32, accuracy loss on CIFAR, Fashion and Tabular. That result is retained here and in the paper as the historical record. Rank 32 exceeds every client's capability maximum, so it is a compute reference and not a feasible baseline; the revised controller is compared against the feasible [4, 8, 16] baseline instead.
