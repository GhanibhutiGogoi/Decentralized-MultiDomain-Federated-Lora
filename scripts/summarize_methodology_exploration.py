#!/usr/bin/env python3
"""Audit and summarize the seven exploratory optimization interventions.

All analysis, checkpoint evaluation and plotting must run on gpu003. This
entry point does not train. It independently evaluates saved fp32 checkpoints,
compares the unmodified controls with the earlier corrected experiment, and
reports paired-seed statistics without claiming formal equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = ("pooled_reset", "pooled_reset_svd", "pooled_reset_svd_center", "fedavg16",
        "fedavg16_center", "fedavg16_iid", "fedavg16_iid_center")
LABELS = ("Pooled; Adam reset", "Pooled + epoch SVD", "Pooled + centered SVD", "FedAvg; non-IID",
          "FedAvg; non-IID + centering", "FedAvg; IID", "FedAvg; IID + centering")
COLORS = ("#222222", "#756BB1", "#CC79A7", "#D55E00", "#E69F00", "#009E73", "#0072B2")
EFFECTS = {
    "pooled_epoch_svd_effect": ("pooled_reset_svd", "pooled_reset"),
    "pooled_centering_after_svd_effect": ("pooled_reset_svd_center", "pooled_reset_svd"),
    "noniid_fedavg_centering_effect": ("fedavg16_center", "fedavg16"),
    "iid_allocation_effect": ("fedavg16_iid", "fedavg16"),
    "iid_centering_effect": ("fedavg16_iid_center", "fedavg16_iid"),
    "iid_fedavg_gap_to_pooled": ("fedavg16_iid", "pooled_reset"),
    "iid_centered_fedavg_gap_to_pooled": ("fedavg16_iid_center", "pooled_reset"),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    values = np.asarray(list(values), dtype=float)
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError("exploration summary requires exactly three finite paired observations")
    return {"n": 3, "mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)),
            "per_seed": values.tolist()}


def scientific_fields(value):
    if isinstance(value, dict):
        return {key: scientific_fields(item) for key, item in value.items() if not key.endswith("seconds")}
    if isinstance(value, list):
        return [scientific_fields(item) for item in value]
    return value


def base_method(arm):
    return "pooled_reset" if arm.startswith("pooled") else "fedavg16"


def load_and_validate(folder, original, seeds):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["status"] != "complete" or not manifest["exploratory"]:
        raise ValueError("completed exploratory manifest required")
    if set(manifest["config"]["arms"]) != set(ARMS) or tuple(manifest["config"]["seeds"]) != tuple(seeds):
        raise ValueError("manifest arm/seed grid does not match the requested comparison")
    records, paths, checks = {}, [], []
    for seed in seeds:
        baseline_path = original / f"seed{seed}_pooled_reset.json"
        baseline = json.loads(baseline_path.read_text())
        original_splits = json.loads((original / f"splits_seed{seed}.json").read_text())["clients"]
        for arm in ARMS:
            base = base_method(arm)
            path = folder / arm / f"seed{seed}_{base}.json"
            paths.append(path)
            record = json.loads(path.read_text())
            if record["status"] != "complete" or record["intervention"]["arm"] != arm:
                raise ValueError(f"incomplete or mislabeled run: {seed}/{arm}")
            if len(record["rounds"]) != 30 or record["n_train"] != 50000 or record["n_test"] != 10000:
                raise ValueError(f"full-data 30-round comparison required: {seed}/{arm}")
            for key in ("alpha", "reference_rank", "initial_state_sha256", "feature_cache_identity_sha256",
                        "training_sample_exposures", "n_train", "n_test"):
                if record[key] != baseline[key]:
                    raise ValueError(f"paired {key} mismatch: {seed}/{arm}")
            if record["optimizer"] != baseline["optimizer"]:
                raise ValueError(f"paired optimizer settings mismatch: {seed}/{arm}")
            if not record["intervention"]["iid_preserving_client_sizes"] and record["split_sha256"] != baseline["split_sha256"]:
                raise ValueError(f"non-IID intervention unexpectedly changed split: {seed}/{arm}")
            splits = json.loads((folder / arm / f"splits_seed{seed}.json").read_text())["clients"]
            if set(splits) != set(original_splits):
                raise ValueError(f"client identifiers differ: {seed}/{arm}")
            for cid in splits:
                if (len(splits[cid]["train_indices"]) != len(original_splits[cid]["train_indices"])
                        or splits[cid]["test_indices"] != original_splits[cid]["test_indices"]):
                    raise ValueError(f"intervention changed client size or test assignment: {seed}/{arm}")
            if sorted(index for split in splits.values() for index in split["train_indices"]) != list(range(50000)):
                raise ValueError(f"training union is not the full disjoint dataset: {seed}/{arm}")
            for index, row in enumerate(record["rounds"], start=1):
                if row["round"] != index or row["train_sample_exposures"] != 50000:
                    raise ValueError(f"round/exposure mismatch: {seed}/{arm}")
                if row["full_test_correct"] / 10000 != row["full_test_accuracy"]:
                    raise ValueError(f"test denominator mismatch: {seed}/{arm}")
            if record["final_full_test_accuracy"] != record["rounds"][-1]["full_test_accuracy"]:
                raise ValueError(f"final metric mismatch: {seed}/{arm}")
            if arm in ("pooled_reset", "fedavg16"):
                previous = json.loads((original / f"seed{seed}_{base}.json").read_text())
                identical = scientific_fields(record["rounds"]) == scientific_fields(previous["rounds"])
                checks.append({"seed": seed, "arm": arm, "all_scientific_round_fields_equal": identical,
                               "n_rounds_compared": 30, "excluded_fields": "timing fields whose names end in seconds"})
                if not identical:
                    raise ValueError(f"unmodified control differs from original experiment: {seed}/{arm}")
            diagnostics = record["optimization_diagnostics"]
            if (arm == "pooled_reset" and diagnostics) or (arm != "pooled_reset" and len(diagnostics) != 30):
                raise ValueError(f"unexpected intervention diagnostic count: {seed}/{arm}")
            records[(seed, arm)] = record
    return records, paths, checks


def independent_checkpoint_verification(folder, records, seeds, cache):
    """Direct CPU fp32 forward pass; no original evaluator/LoRA helpers imported."""
    import torch
    torch.set_num_threads(2)
    data = torch.load(cache / "test.pt", map_location="cpu", weights_only=True)
    metadata = json.loads((cache / "manifest.json").read_text())
    digest = hashlib.sha256()
    for tensor in (data["features"], data["labels"]):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.numpy().tobytes())
    if digest.hexdigest() != metadata["test"]["sha256"] or len(data["labels"]) != 10000:
        raise ValueError("independent evaluator test cache integrity failure")
    verified = []
    with torch.inference_mode():
        for seed in seeds:
            for arm in ARMS:
                checkpoint_path = folder / arm / f"seed{seed}_{base_method(arm)}_adapter.pt"
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                if (checkpoint["seed"] != seed or checkpoint["method"] != base_method(arm)
                        or metadata["cache_identity_sha256"] != records[(seed, arm)]["feature_cache_identity_sha256"]):
                    raise ValueError("independent checkpoint identity or feature representation mismatch")
                if set(checkpoint["state"]) != {"fc"}:
                    raise ValueError("independent evaluator expects only the saved fc adapter")
                factors = checkpoint["state"]["fc"]
                a, b = factors["A"].float(), factors["B"].float()
                rank = a.shape[0]
                if rank != 16 or checkpoint["alpha"] != records[(seed, arm)]["alpha"]:
                    raise ValueError("independent checkpoint rank/alpha mismatch")
                weight = checkpoint["initial"]["weight"].float() + (float(checkpoint["alpha"]) / rank) * (b @ a)
                bias = checkpoint["initial"]["bias"].float()
                prediction = torch.cat([(data["features"][start:start + 1000].float() @ weight.T + bias).argmax(1)
                                        for start in range(0, 10000, 1000)])
                correct = int((prediction == data["labels"]).sum())
                expected = records[(seed, arm)]["rounds"][-1]["full_test_correct"]
                verified.append({"seed": seed, "arm": arm, "correct": correct, "expected_correct": expected,
                                 "n_test": 10000, "accuracy_percent": correct / 100,
                                 "exact_count_match": correct == expected,
                                 "checkpoint_sha256": sha256(checkpoint_path),
                                 "prediction_sha256": hashlib.sha256(prediction.numpy().tobytes()).hexdigest()})
                if correct != expected:
                    raise ValueError(f"independent fp32 count mismatch: {seed}/{arm}: {correct} != {expected}")
    return {"status": "passed", "n_checkpoints": len(verified), "execution": "gpu003 CPU float32 direct matrix multiply; 1000-example batches",
            "independence": "No imported benchmark evaluator, model, or LoRA reconstruction helpers; direct W=W0+(alpha/r)BA, logits=XW^T+b",
            "test_feature_sha256": digest.hexdigest(), "rows": verified}


def summarize(records, checks, verified, seeds):
    aggregate = []
    for arm in ARMS:
        rows = [records[(seed, arm)] for seed in seeds]
        result = {"arm": arm, "full_test_accuracy_percent": stats(100 * row["final_full_test_accuracy"] for row in rows),
                  "paired_gap_to_pooled_reset_pp": stats(100 * (records[(seed, arm)]["final_full_test_accuracy"] - records[(seed, "pooled_reset")]["final_full_test_accuracy"]) for seed in seeds),
                  "optimizer_steps": [row["total_optimizer_steps"] for row in rows],
                  "training_sample_exposures": [row["training_sample_exposures"] for row in rows]}
        if arm != "pooled_reset":
            diagnostics = [row["optimization_diagnostics"] for row in rows]
            for metric in ("common_logit_energy_fraction", "centered_tail16_fraction"):
                result[f"{metric}_round_mean"] = stats(np.mean([d[metric] for d in ds]) for ds in diagnostics)
                result[f"{metric}_final"] = stats(ds[-1][metric] for ds in diagnostics)
            if arm.startswith("pooled"):
                for metric in ("max_centered_train_logit_change", "svd_relative_reconstruction_energy"):
                    result[f"{metric}_maximum_per_seed"] = stats(max(d[metric] for d in ds) for ds in diagnostics)
                    result[f"{metric}_global_maximum"] = max(d[metric] for ds in diagnostics for d in ds)
        aggregate.append(result)
    effects = {name: {"first": first, "second": second,
                      **stats(100 * (records[(seed, first)]["final_full_test_accuracy"] - records[(seed, second)]["final_full_test_accuracy"]) for seed in seeds)}
               for name, (first, second) in EFFECTS.items()}
    return {"status": "complete", "schema_version": 1, "seeds": list(seeds), "arms": list(ARMS), "n_records": len(records),
            "exploratory": True, "scope": "Post-hoc mechanism controls on the same official CIFAR-100 full test set and frozen ResNet-18 features; no fresh held-out confirmation",
            "uncertainty": "Mean and sample SD across three seeds; differences calculated within matching seeds; no formal equivalence claim",
            "comparison_scope": "Same initialization, alpha, total training examples, per-client sample counts and optimizer hyperparameters; pooled and FedAvg optimizer step counts differ",
            "diagnostic_scope": "Common-logit fraction is measured before the intervention's projection; pooled immediate logit changes are centered logits on a fixed training probe only",
            "controls_exactly_reproduced": checks, "independent_checkpoint_verification": verified,
            "aggregate": aggregate, "paired_effects_pp": effects,
            "interpretation": [
                "Roundwise exact reproduction of the unmodified pooled and FedAvg controls ties these interventions to the earlier corrected experiment.",
                "Pooled SVD refactorization barely changes immediate centered training logits and final accuracy; it does not reproduce the large FedAvg failure by itself.",
                "IID allocation recovers much of the non-IID FedAvg deficit while preserving sample counts and the data union, implicating statistical heterogeneity/local-update behavior.",
                "Non-IID FedAvg has large common-logit energy, but removing that mode does not recover its accuracy in these controls; this statistic alone is not a demonstrated remedy.",
                "The IID FedAvg arms still fall below pooled accuracy; these controls are diagnostic, not a successful adaptive decentralized end-to-end result."]}


def plot(summary, records, output, paper):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    output.mkdir(parents=True, exist_ok=True)
    paper.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), gridspec_kw={"height_ratios": [1, 1.12]})
    seeds = summary["seeds"]
    for i, (row, color) in enumerate(zip(summary["aggregate"], COLORS)):
        for axis, field in zip(axes[0], ("full_test_accuracy_percent", "paired_gap_to_pooled_reset_pp")):
            stat = row[field]
            axis.errorbar(stat["mean"], i, xerr=stat["sample_sd"], fmt="o", color=color, capsize=3)
            axis.scatter(stat["per_seed"], i + np.linspace(-.12, .12, 3), color=color, alpha=.5, s=14)
        values = np.array([[100 * r["full_test_accuracy"] for r in records[(seed, row["arm"])]["rounds"]] for seed in seeds])
        reference = np.array([[100 * r["full_test_accuracy"] for r in records[(seed, "pooled_reset")]["rounds"]] for seed in seeds])
        for axis, matrix in zip(axes[1], (values, values - reference)):
            mean, sd = matrix.mean(0), matrix.std(0, ddof=1)
            axis.plot(np.arange(1, 31), mean, color=color, label=LABELS[i], lw=1.5,
                      ls="--" if "center" in row["arm"] else "-")
            axis.fill_between(np.arange(1, 31), mean - sd, mean + sd, color=color, alpha=.10)
    for axis in axes[0]:
        axis.set_yticks(range(7), LABELS)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=.15)
    axes[0, 0].set_xlabel("Final full-test accuracy (%)")
    axes[0, 0].set_title("Exploratory controls: mean ± sample SD")
    axes[0, 1].set_xlabel("Paired difference from pooled (pp)")
    axes[0, 1].set_title("Differences paired by seed")
    axes[0, 1].axvline(0, ls=":", color="#555555", lw=1)
    axes[1, 0].set_ylabel("Full-test accuracy (%)")
    axes[1, 1].set_ylabel("Paired difference from pooled (pp)")
    axes[1, 1].axhline(0, ls=":", color="#555555", lw=1)
    for axis in axes[1]:
        axis.set_xlabel("Training pass over each example")
        axis.grid(alpha=.15)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(handles, labels, fontsize=7.5, frameon=False, loc="lower center", ncol=3,
                  bbox_to_anchor=(.5, -.005))
    figure.tight_layout(rect=(0, .085, 1, 1))
    for ext in ("pdf", "png"):
        figure.savefig(output / f"exploration_optimization.{ext}", bbox_inches="tight", dpi=240)
        figure.savefig(paper.with_suffix("." + ext), bbox_inches="tight", dpi=240)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for arm, label, color in zip(ARMS[1:], LABELS[1:], COLORS[1:]):
        for axis, metric in zip(axes, ("common_logit_energy_fraction", "centered_tail16_fraction")):
            matrix = np.array([[100 * d[metric] for d in records[(seed, arm)]["optimization_diagnostics"]] for seed in seeds])
            axis.plot(np.arange(1, 31), matrix.mean(0), color=color, label=label, ls="--" if "center" in arm else "-")
            axis.fill_between(np.arange(1, 31), matrix.mean(0) - matrix.std(0, ddof=1), matrix.mean(0) + matrix.std(0, ddof=1), color=color, alpha=.10)
            axis.set_xlabel("Training round")
    axes[0].set_ylabel("Common-logit energy before projection (%)")
    axes[1].set_ylabel("Centered singular-value tail beyond rank16 (%)")
    axes[0].legend(fontsize=7, frameon=False)
    figure.tight_layout()
    for ext in ("pdf", "png"):
        figure.savefig(output / f"common_logit_diagnostics.{ext}", bbox_inches="tight", dpi=240)
    plt.close(figure)


def report(summary):
    def format_stat(stat):
        return f"{stat['mean']:.3f} ± {stat['sample_sd']:.3f}"
    lines = ["# Exploratory optimization controls", "", summary["scope"] + ".", "",
             "All 21 saved checkpoints were independently reevaluated on gpu003 with a direct CPU fp32 forward pass over all 10,000 test examples; every displayed final correct count matched. The unmodified pooled-reset and FedAvg controls exactly reproduced all 30 original scientific round records for all three seeds, excluding timings.", "",
             "| Arm | Final full-test accuracy (%) | Paired gap to pooled-reset (pp) |",
             "|---|---:|---:|"]
    for label, row in zip(LABELS, summary["aggregate"]):
        lines.append(f"| {label} | {format_stat(row['full_test_accuracy_percent'])} | {format_stat(row['paired_gap_to_pooled_reset_pp'])} |")
    lines += ["", "| Paired intervention effect | Accuracy difference (pp) |", "|---|---:|"]
    for name, effect in summary["paired_effects_pp"].items():
        lines.append(f"| {name.replace('_', ' ')} | {format_stat(effect)} |")
    lines += ["", summary["uncertainty"] + ".", "", "## What these controls show", ""]
    lines += ["- " + text for text in summary["interpretation"]]
    lines += ["", "## Immediate SVD and common-mode diagnostics", "",
              "SVD reconstruction changes the factor representation. Its immediate functional discrepancy here is measured using centered logits on a fixed training probe; the following small values are not a whole-test-set logit bound.", "",
              "| Pooled intervention | Maximum absolute centered training-logit change | Maximum relative reconstruction energy |",
              "|---|---:|---:|"]
    for row in summary["aggregate"][1:3]:
        lines.append(f"| {row['arm']} | {row['max_centered_train_logit_change_global_maximum']:.8g} | {row['svd_relative_reconstruction_energy_global_maximum']:.8g} |")
    lines += ["", "For common-logit energy, each seed is first averaged over its 30 rounds; uncertainty is then computed across the three seed averages. Measurements are taken before that round's intervention. Centered-tail energy excludes the class-common mode.", "",
              "| Arm | Mean common-logit energy (%) | Mean centered tail beyond rank16 (%) |", "|---|---:|---:|"]
    for row in summary["aggregate"][1:]:
        first, second = [dict(row[key]) for key in ("common_logit_energy_fraction_round_mean", "centered_tail16_fraction_round_mean")]
        for stat in (first, second):
            stat["mean"] *= 100
            stat["sample_sd"] *= 100
        lines.append(f"| {row['arm']} | {format_stat(first)} | {format_stat(second)} |")
    lines += ["", "## Interpretation limits", "", summary["comparison_scope"] + ".",
              "IID reallocation preserves each client's training sample count, the disjoint pooled training union and every test shard. Its changed split is intentional. These arms use centralized FedAvg at uniform rank16; they do not demonstrate adaptive-rank decentralized accuracy preservation. The interventions were selected after observing the original failure and reuse the official test set, so they are exploratory mechanism evidence. No privacy guarantee or new held-out confirmation is claimed.", "",
              "The current script hash, raw-input hashes, original-control matching checks, independent checkpoint/prediction hashes, per-seed final metrics, paired effects and diagnostic summaries are in `summary.json`. Publication figures are generated from the same records. The separate first failed launch is preserved by the parent experiment record; this completed directory does not overwrite it.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--paper-figure", type=Path, required=True, help="Stem without extension")
    args = parser.parse_args()
    seeds = (42, 43, 44)
    records, paths, checks = load_and_validate(args.input, args.original, seeds)
    verification = independent_checkpoint_verification(args.input, records, seeds, args.feature_cache)
    summary = summarize(records, checks, verification, seeds)
    summary["script_sha256"] = sha256(__file__)
    summary["input_files"] = [{"path": str(path.relative_to(args.input)), "sha256": sha256(path)} for path in paths]
    summary["source_manifest_sha256"] = sha256(args.input / "manifest.json")
    summary["original_manifest_sha256"] = sha256(args.original / "manifest.json")
    plot(summary, records, args.input / "figures", args.paper_figure)
    (args.input / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.input / "SUMMARY.md").write_text(report(summary))
    print(json.dumps({"status": summary["status"], "n_records": summary["n_records"],
                      "independently_verified_checkpoints": verification["n_checkpoints"],
                      "paired_effects_pp": summary["paired_effects_pp"]}, indent=2))


if __name__ == "__main__":
    main()
