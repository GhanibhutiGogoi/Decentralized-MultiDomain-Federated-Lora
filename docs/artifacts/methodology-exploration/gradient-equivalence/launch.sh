#!/bin/bash
set -uo pipefail
cd "$HOME/ahlora-exploration-20260915/project-3-hierarchical-gossip" || exit 1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$HOME/pystubs:$PWD"
experiment_root="$HOME/ahlora-runs/methodology-exploration-20260915"
"$HOME/ahlora-venv/bin/python" -m experiments.diagnose_gradient_equivalence \
  --data-dir "$HOME/ahlora-data" --feature-cache "$HOME/ahlora-data/features-v2" \
  --output "$experiment_root/gradient-equivalence-v1" \
  --seeds 42 43 44 --epochs 30 --proof-batches 64 --dtype float32 \
  > "$experiment_root/gradient-equivalence-v1.log" 2>&1
experiment_exit=$?
if [ "$experiment_exit" -eq 0 ]; then
  printf 'completed\n' > "$experiment_root/gradient-equivalence-v1.completed"
else
  printf '%s\n' "$experiment_exit" > "$experiment_root/gradient-equivalence-v1.failed"
fi
exit "$experiment_exit"
