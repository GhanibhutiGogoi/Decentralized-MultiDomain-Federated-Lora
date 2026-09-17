"""Summarize completed paired controls and plot their recorded measurements."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    records = [json.loads((args.run_dir / f"seed{seed}.json").read_text()) for seed in (42, 43, 44)]
    if any(record["status"] != "completed" or len(record["rounds"]) != 30 for record in records):
        raise ValueError("require all three completed 30-epoch paired controls")
    args.output.mkdir(parents=True, exist_ok=True)
    final = []
    curves = []
    for record in records:
        last = record["rounds"][-1]
        final.append({
            "seed": record["seed"],
            "pooled_accuracy_percent": 100 * last["pooled_full_test_accuracy"],
            "peer_accuracy_percent": 100 * last["distributed_full_test_accuracy"],
            "peer_minus_pooled_pp": 100 * (last["distributed_full_test_accuracy"] - last["pooled_full_test_accuracy"]),
            "prediction_disagreements": last["test_prediction_disagreements"],
            "max_factor_abs_error": last["max_factor_abs_error"],
            "fp64_gradient_max_abs_error": record["max_proof_gradient_abs_error"],
            "peer_payload_bytes": record["total_peer_payload_bytes"],
            "peer_messages": record["total_peer_messages"],
            "wall_seconds": record["total_wall_seconds"],
        })
        for row in record["rounds"]:
            curves.append({"seed": record["seed"], **row})
    for filename, rows in (("per_seed.csv", final), ("curves.csv", curves)):
        with (args.output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    summary = {
        "status": "completed", "n_seeds": 3, "epochs": 30,
        "comparison": "paired fp32 pooled versus exact neighbor-tree factor-gradient synchronization, uniform rank16",
        "statistics": "sample standard deviation across seeds; numerical positive control, not population equivalence inference",
        "fp64_gradient_checks": sum(len(record["proof"]) for record in records),
        "max_fp64_gradient_abs_error": max(record["max_proof_gradient_abs_error"] for record in records),
        "max_recorded_prediction_disagreements": max(row["test_prediction_disagreements"] for row in curves),
        "per_seed": final,
    }
    for key in final[0]:
        if key != "seed":
            values = np.asarray([row[key] for row in final], dtype=float)
            summary[key] = {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    epochs = np.arange(1, 31)
    pooled = 100 * np.array([[r["pooled_full_test_accuracy"] for r in record["rounds"]] for record in records])
    peer = 100 * np.array([[r["distributed_full_test_accuracy"] for r in record["rounds"]] for record in records])
    axes[0].plot(epochs, pooled.mean(0), color="#2364aa", linewidth=2, label="Conventional pooled LoRA")
    axes[0].fill_between(epochs, pooled.mean(0) - pooled.std(0, ddof=1),
                         pooled.mean(0) + pooled.std(0, ddof=1), color="#2364aa", alpha=.16)
    axes[0].plot(epochs, peer.mean(0), color="#c94c35", linestyle="--", linewidth=1.5,
                 marker="o", markersize=3, markevery=5, label="Synchronized peer gradients")
    axes[0].set(xlabel="Training epochs", ylabel="Full-test accuracy (%)", ylim=(40, 62),
                title="Matching accuracy (3 seeds)")
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    cumulative = np.array([np.cumsum([r["peer_payload_bytes"] for r in record["rounds"]])
                           for record in records]) / 2**30
    axes[1].plot(epochs, cumulative.mean(0), color="#c94c35", linewidth=2)
    axes[1].plot(epochs, np.zeros(30), color="#2364aa", linewidth=1.5)
    axes[1].set(xlabel="Training epochs", ylabel="Cumulative peer payload (GiB)",
                title="Reduction + broadcast every minibatch", ylim=(-.3, 13))
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
    figure.savefig(args.output / "gradient_equivalence.pdf", bbox_inches="tight")
    figure.savefig(args.output / "gradient_equivalence.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({key: summary[key] for key in ("pooled_accuracy_percent", "peer_accuracy_percent",
                                                    "peer_minus_pooled_pp", "fp64_gradient_checks")}, indent=2))


if __name__ == "__main__":
    main()
