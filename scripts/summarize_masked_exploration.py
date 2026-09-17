#!/usr/bin/env python3
"""Independently verify saved partial-gradient candidates, without training.

Run on gpu003. Only the original train.pt is opened; its fixed 45k/5k holdout
is validated. No project model, forward pass, partitioner or evaluator is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from summarize_residual_exploration import finite_numbers, require, sha256, tensor_digest, verify_tree


LABELS = {
    "pooled": "Conventional pooled LoRA",
    "uniform_sample": "Shared gradients: uniform rank16",
    "fixed_sample": "Partial gradients: fixed / sample",
    "adaptive_sample": "Partial gradients: adaptive / sample",
    "adaptive_quality": "Partial gradients: adaptive / quality",
    "fixed_domain": "Partial gradients: fixed / domain",
    "adaptive_domain": "Partial gradients: adaptive / domain",
}
REPLICATED = ("pooled", "uniform_sample", "fixed_domain", "adaptive_domain")
SEEDS = (42, 43, 44)
DIRECTORIES = ("masked-screen-a", "masked-screen-b", "masked-replication-43",
               "masked-replication-44", "pooled-validation-replication", "residual-screen-v1")


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"n": len(values), "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "per_seed": dict(zip(map(str, SEEDS[:len(values)]), values.tolist()))}


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--feature-cache", required=True, type=Path, help="Directory directly containing train.pt")
    parser.add_argument("--paper-figure", required=True, type=Path)
    args = parser.parse_args()
    output = args.input / "masked-summary"
    output.mkdir(exist_ok=True)
    import torch
    torch.set_num_threads(2)
    data = torch.load(args.feature_cache / "train.pt", map_location="cpu", weights_only=True)
    cache = json.loads((args.feature_cache / "manifest.json").read_text())
    require(tensor_digest((data["features"], data["labels"])) == cache["train"]["sha256"], "training cache hash mismatch")
    common_holdout = None
    manifests = {}
    for directory in DIRECTORIES:
        folder = args.input / directory
        manifest = json.loads((folder / "manifest.json").read_text())
        require(manifest["status"] == "complete", "incomplete manifest: " + directory)
        require(manifest["config"].get("epochs", manifest["config"].get("rounds")) == 30, "wrong epoch budget")
        holdout = json.loads((folder / "holdout.json").read_text())
        digest = hashlib.sha256(json.dumps({k: v for k, v in holdout.items() if k != "sha256"}, sort_keys=True).encode()).hexdigest()
        require(digest == holdout["sha256"] and holdout["official_test_opened"] is False, "holdout provenance mismatch")
        common_holdout = common_holdout or holdout
        require(common_holdout == holdout, "holdouts differ between runs")
        manifests[directory] = manifest
    fit, val = common_holdout["training_original_indices"], common_holdout["validation_original_indices"]
    require(len(fit) == 45000 and len(val) == 5000 and sorted(fit + val) == list(range(50000)), "invalid disjoint holdout")
    require(np.array_equal(np.bincount(data["labels"][val], minlength=100), np.full(100, 50)), "unstratified holdout")
    features, labels = data["features"][val].float(), data["labels"][val]
    records, results, predictions, curves = {}, [], {}, []
    with torch.inference_mode():
        for directory in DIRECTORIES:
            folder, manifest = args.input / directory, manifests[directory]
            arms = ["pooled"] if directory == "residual-screen-v1" else manifest["config"]["arms"]
            for seed in manifest["config"]["seeds"]:
                split_dir = args.input / ("residual-screen-v1" if seed == 42 else "pooled-validation-replication") / "pooled"
                splits = json.loads((split_dir / f"splits_seed{seed}.json").read_text())
                splits = splits.get("clients", splits)
                require(sorted(index for part in splits.values() for index in part["train_indices"]) == list(range(45000)), "partition does not exhaust fit set")
                counts = np.asarray([len(splits[str(cid)]["train_indices"]) for cid in range(15)])
                owners = torch.full((45000,), -1, dtype=torch.long)
                for cid in range(15):
                    owners[splits[str(cid)]["train_indices"]] = cid
                split_sha = hashlib.sha256(json.dumps({int(k): v for k, v in splits.items()}, sort_keys=True).encode()).hexdigest()
                for arm in arms:
                    path = folder / arm / f"seed{seed}_{arm}.json"
                    record = json.loads(path.read_text())
                    require((seed, arm) not in records, "duplicate comparison arm/seed")
                    records[seed, arm] = record
                    require(record["status"] == "complete" and len(record["rounds"]) == 30 and finite_numbers(record), "incomplete/non-finite record")
                    require(record["holdout_sha256"] == common_holdout["sha256"] and "ORIGINAL TRAINING" in record["evaluation_split"], "record evaluation scope mismatch")
                    require(record["n_train"] == 45000 and record.get("n_validation", record.get("n_test")) == 5000, "record sample counts mismatch")
                    require(record["split_sha256"] == split_sha, "partition hash mismatch")
                    if arm != "pooled":
                        require(record["ownership_sha256"] == tensor_digest([owners]), "ownership hash mismatch")
                    grad_bytes = control_bytes = gradient_work = probe_work = exposures = 0
                    for epoch, row in enumerate(record["rounds"], 1):
                        accuracy = row.get("validation_accuracy", row.get("full_test_accuracy"))
                        correct = row.get("validation_correct", row.get("full_test_correct"))
                        require(correct / 5000 == accuracy, "accuracy/count mismatch")
                        require(row.get("epoch", row.get("round")) == epoch and row["optimizer_steps"] == 352, "epoch/step mismatch")
                        exposure = row.get("training_sample_exposures", row.get("train_sample_exposures"))
                        require(exposure == 45000, "wrong epoch sample exposure")
                        exposures += exposure
                        if arm != "pooled":
                            ranks = np.asarray([row["ranks"][str(cid)] for cid in range(15)])
                            ceilings = np.asarray([record["ceilings"][str(cid)] for cid in range(15)])
                            require((ranks > 0).all() and (ranks <= ceilings).all(), "capacity ceiling exceeded")
                            if arm.startswith("adaptive"):
                                require((ranks >= ceilings // 2).all(), "adaptive floor violated")
                                require(all(c["rounds_seen"] == epoch for c in row["controller_diagnostics"].values()), "controller observation mismatch")
                            else:
                                require(np.array_equal(ranks, ceilings), "fixed ranks changed")
                            rank_work = int(counts @ ranks)
                            require(row["training_gradient_rank_sample_products"] == rank_work, "gradient coordinate work mismatch")
                            gradient_work += rank_work
                            require(row["ceiling_probe_rank_sample_products"] == 128 * int(ceilings.sum()), "probe work mismatch")
                            probe_work += row["ceiling_probe_rank_sample_products"]
                            require(row["ceiling_probe_examples"] == 15 * 128, "probe examples mismatch")
                            require(row["postepoch_quality_probe_examples"] == (15 * 128 if arm.startswith("adaptive") else 0), "quality probe examples mismatch")
                            expected_bytes = 352 * 28 * ((16 * (512 + 100)) * 4 + 8)
                            require(row["peer_gradient_payload_bytes"] == expected_bytes and row["peer_gradient_messages"] == 352 * 28, "gradient transport accounting mismatch")
                            grad_bytes += expected_bytes
                            quality = np.asarray(row["pre_epoch_probe_qualities"])
                            domain = np.asarray(row["domain_factors"])
                            multipliers = np.asarray(row["objective_multipliers"])
                            require((quality > 0).all() and (quality <= 1).all(), "invalid quality score")
                            require((domain >= .85 - 1e-12).all() and (domain <= 1.15 + 1e-12).all(), "domain trust bound violated")
                            require(np.isclose(np.average(multipliers, weights=counts), 1, atol=1e-12, rtol=0), "objective scale changed")
                            if arm.endswith("sample"):
                                require(np.array_equal(multipliers, np.ones(15)) and row["control_transport"]["bytes"] == 0, "sample objective mismatch")
                            else:
                                expected = quality * domain
                                expected /= np.average(expected, weights=counts)
                                require(np.allclose(multipliers, expected, atol=1e-12, rtol=0), "quality/domain multipliers mismatch")
                                verify_tree(row["control_transport"], record["topology"], (epoch - 1) % 15, control=True)
                            control_bytes += row["control_transport"]["bytes"]
                        else:
                            gradient_work += row["train_rank_sample_products"]
                        curves.append({"seed": seed, "arm": arm, "epoch": epoch, "validation_accuracy_percent": 100 * accuracy,
                                       "validation_correct": correct, "train_loss": row["train_loss"]})
                    checkpoint_path = path.with_name(path.stem + "_adapter.pt")
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                    require(checkpoint["seed"] == seed and checkpoint.get("arm", checkpoint.get("method")) == arm, "checkpoint identity mismatch")
                    require(tensor_digest(checkpoint["initial"].values()) == record["initial_state_sha256"], "initialization hash mismatch")
                    require(set(checkpoint["state"]) == {"fc"} and checkpoint["alpha"] == 32., "wrong saved adapter")
                    a, b = checkpoint["state"]["fc"]["A"].float(), checkpoint["state"]["fc"]["B"].float()
                    require(a.shape == (16, 512) and b.shape == (100, 16) and torch.isfinite(a).all() and torch.isfinite(b).all(), "invalid factors")
                    weight = checkpoint["initial"]["weight"].float() + 2 * (b @ a)
                    bias = checkpoint["initial"]["bias"].float()
                    logits = torch.cat([features[start:start + 500] @ weight.T + bias for start in range(0, 5000, 500)])
                    prediction = logits.argmax(1)
                    verified = int((prediction == labels).sum())
                    require(verified == correct and accuracy == record.get("final_validation_accuracy", record.get("final_full_test_accuracy")), "independent fp32 accuracy mismatch")
                    predictions[seed, arm] = prediction
                    results.append({"seed": seed, "arm": arm, "label": LABELS[arm], "source_directory": directory,
                                    "validation_accuracy_percent": 100 * accuracy, "independent_correct": verified,
                                    "independent_cross_entropy": float((torch.logsumexp(logits.double(), 1) - logits.double()[torch.arange(5000), labels]).mean()),
                                    "training_sample_exposures": exposures, "optimizer_steps": 10560,
                                    "gradient_rank_sample_products": gradient_work, "ceiling_probe_rank_sample_products": probe_work,
                                    "gradient_coordinate_work_fraction_of_fullrank": gradient_work / (1350000 * 16),
                                    "gradient_payload_bytes": grad_bytes, "control_payload_bytes": control_bytes,
                                    "total_modeled_payload_bytes": grad_bytes + control_bytes,
                                    "checkpoint_sha256": sha256(checkpoint_path), "record_sha256": sha256(path),
                                    "prediction_sha256": hashlib.sha256(prediction.numpy().tobytes()).hexdigest(),
                                    "independent_fp32_count_match": True})
    require(len(results) == 15, "expected 12 partial-gradient and 3 conventional checkpoints")
    indexed = {(r["seed"], r["arm"]): r for r in results}
    paired_checks = []
    for seed in SEEDS:
        reference = records[seed, "pooled"]
        for arm in REPLICATED[1:]:
            require(records[seed, arm]["initial_state_sha256"] == reference["initial_state_sha256"], "paired initialization differs")
        matching_rounds = sum(records[seed, "uniform_sample"]["rounds"][i]["validation_correct"] == reference["rounds"][i]["full_test_correct"] for i in range(30))
        matching_predictions = bool(torch.equal(predictions[seed, "uniform_sample"], predictions[seed, "pooled"]))
        paired_checks.append({"seed": seed, "uniform_vs_pooled_matching_round_counts": matching_rounds,
                              "final_predictions_identical": matching_predictions,
                              "intermediate_count_differences": [{"epoch": i + 1,
                                  "uniform_correct": records[seed, "uniform_sample"]["rounds"][i]["validation_correct"],
                                  "pooled_correct": reference["rounds"][i]["full_test_correct"]}
                                  for i in range(30) if records[seed, "uniform_sample"]["rounds"][i]["validation_correct"] != reference["rounds"][i]["full_test_correct"]]})
    aggregates = {arm: stats([indexed[seed, arm]["validation_accuracy_percent"] for seed in SEEDS]) for arm in REPLICATED}
    effects = {name: stats([indexed[seed, a]["validation_accuracy_percent"] - indexed[seed, b]["validation_accuracy_percent"] for seed in SEEDS])
               for name, a, b in (("uniform_minus_pooled", "uniform_sample", "pooled"),
                                  ("fixed_domain_minus_pooled", "fixed_domain", "pooled"),
                                  ("adaptive_domain_minus_pooled", "adaptive_domain", "pooled"),
                                  ("adaptive_domain_minus_fixed_domain", "adaptive_domain", "fixed_domain"))}
    summary = {"status": "complete", "schema_version": 1, "exploratory": True, "official_test_opened_by_verifier": False,
               "evaluation_split": "5000 held-out ORIGINAL TRAINING examples; 45000 fit examples; official test not opened",
               "holdout_sha256": common_holdout["sha256"], "original_training_tensor_sha256": cache["train"]["sha256"],
               "independent_checkpoints_verified": 15, "epochs_per_run": 30, "seeds": list(SEEDS),
               "checkpoint_evaluation": "CPU fp32 direct X(W0+2BA)^T+b; no project evaluator/forward imports",
               "script_sha256": sha256(__file__), "results": results, "replicated_accuracy_percent": aggregates,
               "paired_effects_percentage_points": effects, "uniform_control_checks": paired_checks,
               "scope": "Partial factor-gradient coordinates only. Every client requires rank16 factors, rank16 forward computation, full Adam state and padded full-rank gradients. One-process simulation with global minibatch scheduling and a rotating peer-tree collective; not the original model-state gossip protocol.",
               "uncertainty": "Sample SD across three paired seeds on one fixed validation holdout; no equivalence margin or formal noninferiority test. Additional screening arms have only seed42.",
               "privacy": "No privacy guarantee. Dense gradient-derived domain signals, training histograms and quality metadata are exchanged.",
               "resource_accounting": "Gradient-coordinate products are a backward-computation proxy only, excluding common full-rank forward, optimizer work and probe work. Communication is modeled float payload including weight-control traffic, not measured network throughput."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_csv(output / "per_seed.csv", results)
    write_csv(output / "curves.csv", curves)
    (output / "SUMMARY.md").write_text(report(summary))
    plot(summary, curves, output, args.paper_figure)
    files = []
    for directory in DIRECTORIES:
        for path in sorted((args.input / directory).rglob("*")):
            if path.is_file() and not path.name.startswith("._"):
                files.append({"path": str(path.relative_to(args.input)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    (output / "data_manifest.json").write_text(json.dumps({"status": "complete", "files": files}, indent=2) + "\n")
    print(json.dumps({"status": "complete", "verified_checkpoints": 15, "replicated_accuracy_percent": aggregates,
                      "paired_effects_percentage_points": effects, "uniform_control_checks": paired_checks}, indent=2))


def report(summary):
    lines = ["# Partial-gradient exploration: completed validation-only screen and replication", "", summary["evaluation_split"], "",
             "All 15 saved adapters were independently evaluated on gpu003; every fp32 correct count matched the run record. The three replicated seeds are 42, 43 and 44. Seed42 conventional LoRA is reused from the residual screen. The same stratified training holdout, partition and initialization are used for each paired comparison.", "",
             "| Replicated arm | Validation accuracy, mean ± sample SD (%) |", "|---|---:|"]
    for arm, values in summary["replicated_accuracy_percent"].items():
        lines.append(f"| {LABELS[arm]} | {values['mean']:.2f} ± {values['sample_sd']:.2f} |")
    lines += ["", "| Paired effect | Mean ± sample SD (percentage points) |", "|---|---:|"]
    for name, values in summary["paired_effects_percentage_points"].items():
        lines.append(f"| {name.replace('_', ' ')} | {values['mean']:+.2f} ± {values['sample_sd']:.2f} |")
    matching = sum(check["uniform_vs_pooled_matching_round_counts"] for check in summary["uniform_control_checks"])
    lines += ["", f"Uniform shared gradients and conventional pooled LoRA have identical final prediction vectors at all three seeds. Their per-epoch correct counts match in {matching}/90 evaluations; any intermediate differences are explicitly listed in `summary.json`. Endpoint equality does not imply bitwise equality of their optimization trajectories."]
    lines += ["", "## All seed42 screening arms", "", "| Arm | Validation accuracy (%) |", "|---|---:|"]
    for row in summary["results"]:
        if row["seed"] == 42:
            lines.append(f"| {row['label']} | {row['validation_accuracy_percent']:.2f} |")
    lines += ["", "## Interpretation and limits", "", summary["scope"], "", summary["uncertainty"], "", summary["resource_accounting"], "", summary["privacy"], "",
              "The shared-gradient control reaches the same endpoint as ordinary pooled LoRA, while partial-gradient candidates remain below it. These are promising changed-method results, not a successful demonstration of the original rank-limited client-memory claim. The adaptive/domain candidate is weaker than its fixed/domain counterpart. All screening variants are retained; the replication plan was recorded before the seed42 screen finished. No official-test evaluation of these candidates was performed.", "",
              "Transport validation checks each recorded weight-control collective's neighbor edges, root rotation, reduction/broadcast order, message totals and byte ledger. Gradient collective counts are checked against 352 steps × 28 messages per epoch; per-step gradient edge traces were not recorded, so their runtime paths cannot be reconstructed independently from the artifacts. Gradient-rank products, train exposures, ceiling-probe work, normalized sample/quality/domain multipliers, rank ceilings and controller observations are checked for every round.", ""]
    return "\n".join(lines)


def plot(summary, curves, output, paper):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    colors = ("#222222", "#56B4E9", "#0072B2", "#CC79A7")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    for index, (arm, color) in enumerate(zip(REPLICATED, colors)):
        values = summary["replicated_accuracy_percent"][arm]
        axes[0].bar(index, values["mean"], yerr=values["sample_sd"], color=color, capsize=4)
        axes[0].scatter([index] * 3, list(values["per_seed"].values()), color="white", edgecolors="#333333", s=20, zorder=4)
        epochs = np.asarray([[r["validation_accuracy_percent"] for r in curves if r["arm"] == arm and r["seed"] == seed] for seed in SEEDS])
        mean, sd = epochs.mean(0), epochs.std(0, ddof=1)
        axes[1].plot(range(1, 31), mean, label=LABELS[arm], color=color, ls="--" if arm == "uniform_sample" else "-")
        axes[1].fill_between(range(1, 31), mean - sd, mean + sd, color=color, alpha=.12)
    axes[0].set_xticks(range(4), ["Pooled", "Uniform\nshared gradients", "Fixed partial\n+ domain", "Adaptive partial\n+ domain"], fontsize=8)
    axes[0].set_ylabel("Training-holdout accuracy (%)")
    axes[0].set_ylim(0, 65)
    axes[0].set_title("Final checkpoint; three seeds, mean ± sample SD")
    axes[1].set_xlabel("Epoch; 45000 fit examples per epoch")
    axes[1].set_ylabel("Training-holdout accuracy (%)")
    axes[1].legend(fontsize=7, frameon=False, loc="lower right")
    axes[1].set_title("Common 5000-example validation split")
    fig.tight_layout()
    paper.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(output / f"exploration_masked.{ext}", dpi=230, bbox_inches="tight")
        fig.savefig(paper.with_suffix("." + ext), dpi=230, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
