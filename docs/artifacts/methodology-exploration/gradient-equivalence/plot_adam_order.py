"""Plot the fixed training-only Adam-order diagnostic, without inference tests."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--run-dir", type=Path, required=True)
args = p.parse_args()
record = json.loads((args.run_dir / "results.json").read_text())
assert record["status"] == "completed" and len(record["batches"]) == 24
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.7), layout="constrained")
seeds = sorted({row["seed"] for row in record["batches"]})
for index, seed in enumerate(seeds):
    rows = [row for row in record["batches"] if row["seed"] == seed]
    for shift, field, color, label in ((-.18, "global_adam_loss_change", "#2364aa", "Adam after gradient reduction"),
                                       (.18, "averaged_local_adam_loss_change", "#c94c35", "Mean of local Adam updates")):
        values = [-row[field] for row in rows]
        axes[0].bar(index + shift, np.mean(values), width=.32, color=color, alpha=.75, label=label if index == 0 else None)
        axes[0].scatter(index + shift + np.linspace(-.09, .09, len(values)), values, color=color, s=12, zorder=3)
    cosines = [row["effective_update_comparison"]["cosine"] for row in rows]
    axes[1].scatter(index + np.linspace(-.12, .12, len(rows)), cosines, color="#7549a5", s=24)
    axes[1].plot([index - .2, index + .2], [np.mean(cosines)] * 2, color="#7549a5", linewidth=2)
axes[0].set(ylabel="Same-batch cross-entropy decrease", title="One optimizer step from shared initialization")
axes[0].legend(frameon=False, fontsize=8, loc="upper right")
axes[1].axhline(1., color="grey", linestyle="--", linewidth=1, label="Identical update direction")
axes[1].set(ylabel="Cosine between effective updates", ylim=(0, 1.08), title="Equal raw gradients, different Adam updates")
axes[1].legend(frameon=False, fontsize=8, loc="upper right")
for axis in axes:
    axis.set_xticks(range(len(seeds)), [str(seed) for seed in seeds])
    axis.set_xlabel("Seed (8 training minibatches each)")
    axis.grid(axis="y", alpha=.2)
figure.savefig(args.run_dir / "adam_order.pdf", bbox_inches="tight")
figure.savefig(args.run_dir / "adam_order.png", dpi=180, bbox_inches="tight")
plt.close(figure)
