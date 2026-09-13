# P1 adaptive-rank artifacts

Populate this directory from the documented GPU host after running the adaptive controller on all five real Project 1 tasks.

Required files:

- `manifest.json`: command, source commit, environment, datasets, seeds and candidate-rank/capability settings.
- `summary.csv`: one row per task/seed/method with final accuracy, FLOPs, factor payload, average and selected ranks, number of rank changes, and budget residual/violations.
- `rank_history.csv`: per-client and per-round stable rank, EMA demand, selected rank, residual ratio and tail mass.
- `figures/`: rank trajectories and accuracy/FLOPs comparison generated from these CSVs.

Evaluate the preregistered gate (within one percentage point of oracle accuracy, at least 25% FLOP savings, >1 rank on at least two tasks, zero budget violations). If the gate fails, report the controller as implemented/stable but not calibrated and retain that limitation in the paper.
