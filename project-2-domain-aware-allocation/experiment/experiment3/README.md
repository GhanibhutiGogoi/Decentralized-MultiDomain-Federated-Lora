# Experiment 3 Engineering Pipeline

Experiment 3 is the future real-data comparison of three aggregation arms:
baseline `w_i q_i`, Form A `w_i q_i lambda_A,i`, and Form B
`w_i q_i lambda_B,i`. This directory currently contains software
infrastructure only. It does not execute or interpret a scientific Experiment 3
run.

Production execution depends on a corrected, validated Experiment 2 calibration
bundle. The current historical Experiment 2 outputs are stale and are rejected
because they are not the strict bundle schema required here.

## Calibration Bundle

The required schema is `exp3-calibration-bundle/v1`. A production bundle must
include `purpose: scientific_calibration`, `scientifically_valid: true`,
`is_synthetic: false`, `completion_status: complete`, an exact
`expected_task_set`, source Experiment 1 and Experiment 2 run IDs, source commit,
dataset provenance for every task, and both `form_a` and `form_b` calibrations.
Each form stores exact feature order, coefficients, intercept, gamma, clipping
bounds, feature means, and feature standard deviations. Missing, duplicate,
ambiguous, non-finite, synthetic, incomplete, mismatched, or stale values fail
closed.

## Seeds And Partitions

Experiment 3 records separate run, partition, model, data-loader, and
permutation seeds. Fresh Dirichlet partitions are generated from labels via the
shared Project 2 partitioning implementation, not from Experiment 1 or
Experiment 2 calibration outputs. Partition hashes are recorded for run identity
and statistical pairing.
Before each paired arm trains for the same task and round, the runner resets
Python, NumPy, PyTorch CPU/CUDA, and per-client DataLoader generator state from
the same arm-independent seed. Baseline, Form A, and Form B receive independent
copies of the same initial global model state; only lambda weighting differs.

## Metrics And Statistics

Implemented metrics include per-round global accuracy, final-round global
accuracy, convergence curves, a first-round-to-target convergence-speed measure,
per-domain accuracy, worst-domain accuracy, population variance of domain
accuracy, realized aggregation weights, and within-round rank agreement.
Statistical utilities pair arm-versus-baseline comparisons by task, seed, and
partition identity. Paired permutation tests use exact sign-flip enumeration for
small samples and deterministic Monte Carlo with the plus-one convention for
larger samples. MDE comparison requires an explicit configured MDE.
Production CLI runs require `--mde`; no default minimum detectable effect is
invented by the runner.

```powershell
python project-2-domain-aware-allocation/experiment/experiment3/run.py --calibration-bundle project-2-domain-aware-allocation/outputs/exp2/calibration_bundle.json --experiment1-run-id <exp1-run-id> --experiment2-run-id <exp2-run-id> --mde <minimum-detectable-effect>
```

Final artifacts include paired arm differences, paired permutation tests,
paired confidence intervals, task-level paired summaries, and configured MDE
comparisons in addition to per-round accuracy, per-domain accuracy, fairness,
realized-weight, and lambda-rank-agreement outputs.

## Checkpoint And Output Ownership

Checkpoints are JSON records with schema version, config hash, calibration
bundle hash, source commit, task, arm, seed, partition, round, client, status,
and data. Writes use a temporary file, fsync, and atomic replace. Resume rejects
malformed or incompatible checkpoints.

Future scientific artifacts belong only under `outputs/exp3`. The runner
validates configuration and calibration before preparing output directories, and
cleanup is confined to `outputs/exp3` or descendants. Experiment 1 and
Experiment 2 outputs are protected from overlap and deletion.

Engineering smoke validation uses synthetic in-memory data and a private test
fixture under `tests/fixtures/experiment3`; it is not a scientific run and does
not produce conclusions.

