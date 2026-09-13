# P1 adaptive-rank artifacts

This directory contains the completed five-task adaptive-rank battery from the documented GPU host.

Required files:

- `manifest.json`: command, source commit, environment, datasets, seeds and candidate-rank/capability settings.
- `summary.csv`: one row per task/seed/method with final accuracy, FLOPs, factor payload, average and selected ranks, number of rank changes, and budget residual/violations.
- `rank_history.csv`: per-client and per-round stable rank, EMA demand, selected rank, residual ratio and tail mass.
- `fig1_accuracy_curves.png`, `fig2_adaptive_rank_per_client.png`, `fig3_final_accuracy_bar.png`, `fig4_total_flops_per_round.png`, `fig5_flops_per_client.png`, `fig6_pareto_accuracy_flops.png`: rank trajectories and accuracy/FLOPs comparisons generated from the run.

Evaluate the preregistered gate (within one percentage point of oracle accuracy, at least 25% FLOP savings, >1 rank on at least two tasks, zero budget violations). If the gate fails, report the controller as implemented/stable but not calibrated and retain that limitation in the paper.
