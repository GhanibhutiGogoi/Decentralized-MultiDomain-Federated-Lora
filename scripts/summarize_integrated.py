#!/usr/bin/env python3
"""Summarize completed, paired integrated runs and render paper figures.

Run on the designated experiment host. The default requires all nine arms at
seeds 42/43/44, all 50,000 training examples and all 10,000 test examples. It
refuses failed, incomplete or mismatched comparisons instead of plotting a
plausible-looking result from unpaired runs. Error bars are sample SD across
seeds; this script makes no formal noninferiority or equivalence claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


METHODS = ("pooled", "pooled_reset", "fedavg16", "mh16", "mh_sample",
           "fixed_quality", "fixed_domain", "adaptive_quality", "adaptive_domain")
FACTORIAL = ("fixed_quality", "fixed_domain", "adaptive_quality", "adaptive_domain")
LABELS = {
    "pooled": "Pooled rank 16",
    "pooled_reset": "Pooled rank 16\nAdam reset",
    "fedavg16": "FedAvg rank 16",
    "mh16": "Peer rank 16",
    "mh_sample": "Peer ranks 4/8/16\nsample weights",
    "fixed_quality": "Fixed ranks\nquality weights",
    "fixed_domain": "Fixed ranks\ndomain weights",
    "adaptive_quality": "Adaptive ranks\nquality weights",
    "adaptive_domain": "Adaptive ranks\ndomain weights",
}
COLORS = dict(zip(METHODS, ("#242424", "#888888", "#9C755F", "#756BB1", "#56B4E9",
                           "#0072B2", "#009E73", "#D55E00", "#CC79A7")))


def stats(values):
    array = np.asarray(list(values), dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("statistics require nonempty finite observations")
    return {"n": len(array), "mean": float(array.mean()),
            "sample_sd": float(array.std(ddof=1)) if len(array) > 1 else None,
            "min": float(array.min()), "max": float(array.max())}


def summarize_records(records, *, methods=METHODS, seeds=(42, 43, 44),
                      expected_train=50000, expected_test=10000):
    """Validate pairing and produce seed-level, aggregate and factorial tables."""
    methods, seeds = tuple(methods), tuple(seeds)
    if "pooled" not in methods or len(set(methods)) != len(methods) or len(set(seeds)) != len(seeds):
        raise ValueError("unique methods/seeds and a pooled reference are required")
    indexed = {}
    for record in records:
        key = (record["seed"], record["method"])
        if key in indexed:
            raise ValueError(f"duplicate run: {key}")
        indexed[key] = record
    expected = {(seed, method) for seed in seeds for method in methods}
    if set(indexed) != expected:
        raise ValueError(f"run grid mismatch: missing={sorted(expected - set(indexed))}, extra={sorted(set(indexed) - expected)}")
    pairing_keys = ("alpha", "reference_rank", "initial_state_sha256", "split_sha256",
                    "n_train", "n_test", "feature_cache_identity_sha256", "training_sample_exposures")
    optimizer_keys = ("name", "lr", "weight_decay", "batch_size", "local_epochs")
    per_seed = []
    for seed in seeds:
        pooled = indexed[(seed, "pooled")]
        reference_rounds = [row["round"] for row in pooled["rounds"]]
        for method in methods:
            record = indexed[(seed, method)]
            rows = record["rounds"]
            if record.get("status") != "complete" or not rows:
                raise ValueError(f"run is not complete: {(seed, method)}")
            if reference_rounds != list(range(1, len(rows) + 1)) or [r["round"] for r in rows] != reference_rounds:
                raise ValueError(f"round sequence mismatch: {(seed, method)}")
            for key in pairing_keys:
                if record[key] != pooled[key]:
                    raise ValueError(f"pairing mismatch for {key}: {(seed, method)}")
            for key in optimizer_keys:
                if record["optimizer"][key] != pooled["optimizer"][key]:
                    raise ValueError(f"optimizer setting mismatch for {key}: {(seed, method)}")
            if record["optimizer"]["reset_each_round"] != (method != "pooled"):
                raise ValueError(f"unexpected optimizer moment policy: {(seed, method)}")
            if expected_train is not None and record["n_train"] != expected_train:
                raise ValueError(f"not the full training set: {(seed, method)}")
            if expected_test is not None and record["n_test"] != expected_test:
                raise ValueError(f"not the full test set: {(seed, method)}")
            if not np.isclose(record["final_full_test_accuracy"], rows[-1]["full_test_accuracy"], atol=1e-14, rtol=0):
                raise ValueError(f"final endpoint mismatch: {(seed, method)}")
            for row, reference in zip(rows, pooled["rounds"]):
                if row["train_sample_exposures"] != reference["train_sample_exposures"]:
                    raise ValueError(f"per-round sample exposure mismatch: {(seed, method)}")
                if not 0 <= row["full_test_accuracy"] <= 1:
                    raise ValueError(f"invalid test accuracy: {(seed, method)}")
                if not np.isclose(row["full_test_correct"] / record["n_test"], row["full_test_accuracy"], atol=1e-14, rtol=0):
                    raise ValueError(f"test count/accuracy mismatch: {(seed, method)}")
            factor_bytes = 4 * record["total_training_factor_floats"]
            control_bytes = record["total_control_bytes"]
            final_bytes = (record.get("final_assembly") or {}).get("bytes", 0)
            deployment_bytes = factor_bytes + control_bytes + final_bytes
            if deployment_bytes != record["total_deployment_payload_bytes"]:
                raise ValueError(f"communication accounting mismatch: {(seed, method)}")
            row = {
                "seed": seed, "method": method, "rounds": len(rows),
                "accuracy_percent": 100 * record["final_full_test_accuracy"],
                "paired_difference_vs_pooled_pp": 100 * (record["final_full_test_accuracy"] - pooled["final_full_test_accuracy"]),
                "training_sample_exposures": record["training_sample_exposures"],
                "optimizer_steps": record["total_optimizer_steps"],
                "training_factor_bytes": factor_bytes,
                "control_bytes": control_bytes,
                "final_assembly_bytes": final_bytes,
                "deployment_payload_bytes": deployment_bytes,
                "evaluation_only_assembly_bytes": record["evaluation_only_assembly_bytes"],
                "train_rank_sample_products": sum(r["train_rank_sample_products"] for r in rows),
                "gradient_probe_examples": sum(r.get("gradient_probe_examples", 0) for r in rows),
                "gradient_probe_rank_sample_products": sum(r.get("gradient_probe_rank_sample_products", 0) for r in rows),
                "quality_probe_examples": sum(r.get("quality_probe_examples", 0) for r in rows),
                "wall_seconds_including_evaluation": record["wall_seconds"],
                "local_train_seconds": sum(r["train_seconds"] for r in rows),
                "gradient_probe_seconds": sum(r.get("gradient_probe_seconds", 0) for r in rows),
                "quality_probe_seconds": sum(r.get("quality_probe_seconds", 0) for r in rows),
                "control_seconds": sum(r.get("control_seconds", 0) for r in rows),
                "gossip_seconds": sum(r.get("gossip_seconds", 0) for r in rows),
                "all_assembly_seconds_including_evaluation": sum(r.get("assembly_seconds", 0) for r in rows),
                "final_assembly_peak_peer_dense_bytes": (record.get("final_assembly") or {}).get("peak_peer_dense_bytes", 0),
                "final_assembly_relative_truncation_energy": (record.get("final_assembly") or {}).get("relative_truncation_energy", 0),
            }
            per_seed.append(row)
    fields = [field for field in per_seed[0] if field not in {"seed", "method", "rounds"}]
    aggregate = [{"method": method, "n_seeds": len(seeds), "rounds": len(indexed[(seeds[0], method)]["rounds"]),
                  **{field: stats(row[field] for row in per_seed if row["method"] == method) for field in fields}}
                 for method in methods]
    factorial_rows = []
    if set(FACTORIAL).issubset(methods):
        for seed in seeds:
            fq, fd, aq, ad = [100 * indexed[(seed, method)]["final_full_test_accuracy"] for method in FACTORIAL]
            factorial_rows.append({"seed": seed, "rank_effect_without_domain_pp": aq - fq,
                                   "rank_effect_with_domain_pp": ad - fd,
                                   "domain_effect_fixed_rank_pp": fd - fq,
                                   "domain_effect_adaptive_rank_pp": ad - aq,
                                   "rank_main_effect_pp": ((aq - fq) + (ad - fd)) / 2,
                                   "domain_main_effect_pp": ((fd - fq) + (ad - aq)) / 2,
                                   "interaction_pp": ad - aq - fd + fq})
    resource_pairs = []
    for seed in seeds:
        rows = {row["method"]: row for row in per_seed if row["seed"] == seed}
        for suffix in ("quality", "domain"):
            fixed, adaptive = f"fixed_{suffix}", f"adaptive_{suffix}"
            if fixed not in rows or adaptive not in rows:
                continue
            f, a = rows[fixed], rows[adaptive]
            resource_pairs.append({"seed": seed, "policy": suffix,
                                   "training_factor_saving_percent": 100 * (1 - a["training_factor_bytes"] / f["training_factor_bytes"]),
                                   "deployment_payload_saving_percent": 100 * (1 - a["deployment_payload_bytes"] / f["deployment_payload_bytes"]),
                                   "rank_sample_product_saving_percent": 100 * (1 - a["train_rank_sample_products"] / f["train_rank_sample_products"])})
    return {
        "schema_version": 1, "status": "complete", "seeds": list(seeds), "methods": list(methods),
        "primary_endpoint": "final full-test accuracy difference from pooled persistent-Adam rank16 LoRA, in percentage points",
        "uncertainty": "sample standard deviation across paired seeds (ddof=1), not a confidence interval",
        "inference": "descriptive paired comparison; no prespecified noninferiority/equivalence margin or formal equivalence claim",
        "scope": "CIFAR-100; frozen ResNet-18 features and a LoRA classification head; single-process peer-flow simulation",
        "optimizer_scope": "same training sample exposures, alpha, initialization, split and Adam hyperparameters; optimizer step counts and moment persistence differ",
        "privacy_scope": "no privacy guarantee; domain control exchange reveals training histograms and local dense adapter changes to all peers",
        "communication_scope": "float32 training factor payloads + float64 modeled domain-control payloads + one final dense assembly; setup/framing/headers excluded; earlier evaluation assemblies separate",
        "resource_scope": "factor bytes and rank-sample products are proxies; probe passes and measured timings are separate; no total-FLOP saving claim",
        "validated_pairing_keys": list(pairing_keys), "validated_optimizer_keys": list(optimizer_keys),
        "per_seed": per_seed, "aggregate": aggregate,
        "factorial_per_seed": factorial_rows,
        "factorial_aggregate": {key: stats(row[key] for row in factorial_rows)
                                for key in factorial_rows[0] if key != "seed"} if factorial_rows else {},
        "adaptive_resource_pairs": resource_pairs,
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flatten_aggregate(rows):
    output = []
    for row in rows:
        flat = {}
        for key, value in row.items():
            if isinstance(value, dict):
                flat.update({f"{key}_{stat}": number for stat, number in value.items()})
            else:
                flat[key] = value
        output.append(flat)
    return output


def curve_tables(records):
    curves, ranks = [], []
    for record in records:
        for row in record["rounds"]:
            curves.append({"seed": record["seed"], "method": record["method"], "round": row["round"],
                           "full_test_accuracy_percent": 100 * row["full_test_accuracy"],
                           "full_test_correct": row["full_test_correct"], "n_test": record["n_test"],
                           "train_loss": row["train_loss"], "training_factor_bytes": 4 * row["training_factor_floats"],
                           "control_bytes": row["control_bytes"], "assembly_bytes": (row.get("assembly") or {}).get("bytes", 0)})
            for cid, rank in row.get("ranks", {}).items():
                diagnostic = row.get("controller_diagnostics", {}).get(str(cid), {})
                ranks.append({"seed": record["seed"], "method": record["method"], "round": row["round"],
                              "client_id": int(cid), "rank": rank,
                              "capacity_ceiling": record["capacity_ceilings"][str(cid)],
                              "stable_rank": row.get("stable_ranks", {}).get(str(cid)),
                              "ema_demand": diagnostic.get("ema_demand"),
                              "quality_ema": diagnostic.get("quality_ema"),
                              "changed": diagnostic.get("changed")})
    return curves, ranks


def figures(summary, records, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 220})
    indexed = {(r["seed"], r["method"]): r for r in records}
    aggregate = {r["method"]: r for r in summary["aggregate"]}
    methods, seeds = summary["methods"], summary["seeds"]
    files = []

    def save(fig, stem):
        fig.tight_layout()
        for extension in ("pdf", "png"):
            path = output / f"integrated_{stem}.{extension}"
            fig.savefig(path, bbox_inches="tight")
            files.append(path.name)
        plt.close(fig)

    def mean_sd(method, field):
        item = aggregate[method][field]
        return item["mean"], item["sample_sd"] or 0.0

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)
    positions = np.arange(len(methods))
    for position, method in zip(positions, methods):
        for ax, field in zip(axes, ("accuracy_percent", "paired_difference_vs_pooled_pp")):
            mean, sd = mean_sd(method, field)
            ax.errorbar(mean, position, xerr=sd, fmt="o", color=COLORS[method], capsize=3)
            seed_values = [r[field] for r in summary["per_seed"] if r["method"] == method]
            ax.scatter(seed_values, np.full(len(seed_values), position) + np.linspace(-.11, .11, len(seed_values)),
                       color=COLORS[method], alpha=.5, s=12)
    axes[0].set_yticks(positions, [LABELS[m] for m in methods])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Final full-test accuracy (%)")
    axes[0].set_title("Assembled rank-16 model; mean ± sample SD")
    axes[1].axvline(0, color="#777777", lw=1, ls="--")
    axes[1].set_xlabel("Paired difference from pooled (percentage points)")
    axes[1].set_title("Differences paired by seed; mean ± sample SD")
    save(fig, "final_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), sharey=True)
    groups = ([m for m in methods if m not in FACTORIAL], ["pooled"] + [m for m in FACTORIAL if m in methods])
    for ax, group, title in zip(axes, groups, ("Pooled and communication controls", "Adaptive-rank × domain-weight ablation")):
        for method in group:
            matrix = np.array([[100 * row["full_test_accuracy"] for row in indexed[(seed, method)]["rounds"]] for seed in seeds])
            rounds = np.cumsum([row["train_sample_exposures"] for row in indexed[(seeds[0], method)]["rounds"]]) / indexed[(seeds[0], method)]["n_train"]
            mean, sd = matrix.mean(axis=0), matrix.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(matrix.shape[1])
            ax.plot(rounds, mean, color=COLORS[method], label=LABELS[method].replace("\n", " "), lw=1.7,
                    ls="--" if method == "pooled" else "-")
            ax.fill_between(rounds, mean - sd, mean + sd, color=COLORS[method], alpha=.10, linewidth=0)
        ax.set_xlabel("Training pass over each example")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best", frameon=False)
    axes[0].set_ylabel("Full-test accuracy (%)")
    save(fig, "accuracy_curves")

    if set(FACTORIAL).issubset(methods):
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
        for pair, label, color, marker in [(FACTORIAL[:2], "Fixed capacity ranks", "#0072B2", "o"),
                                           (FACTORIAL[2:], "P1 adaptive ranks", "#D55E00", "s")]:
            means, sds = zip(*(mean_sd(method, "accuracy_percent") for method in pair))
            axes[0].errorbar([0, 1], means, yerr=sds, color=color, marker=marker, capsize=4, label=label)
        axes[0].set_xticks([0, 1], ["Sample × quality", "Sample × quality × domain"])
        axes[0].set_ylabel("Final full-test accuracy (%)")
        axes[0].set_xlim(-.18, 1.18)
        axes[0].legend(frameon=False, fontsize=8)
        keys = ["rank_effect_without_domain_pp", "rank_effect_with_domain_pp", "domain_effect_fixed_rank_pp",
                "domain_effect_adaptive_rank_pp", "interaction_pp"]
        labels = ["Rank effect; quality only", "Rank effect; with domain", "Domain effect; fixed ranks",
                  "Domain effect; adaptive ranks", "Rank × domain interaction"]
        for i, key in enumerate(keys):
            value = summary["factorial_aggregate"][key]
            axes[1].errorbar(value["mean"], i, xerr=value["sample_sd"] or 0, fmt="o", color="#444444", capsize=3)
        axes[1].set_yticks(range(len(keys)), labels)
        axes[1].invert_yaxis()
        axes[1].axvline(0, color="#777777", lw=1, ls="--")
        axes[1].set_xlabel("Paired effect (percentage points)")
        save(fig, "factorial")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)
    left = np.zeros(len(methods))
    for field, label, color in [("training_factor_bytes", "Training factors", "#0072B2"),
                                 ("control_bytes", "Weight-control exchange", "#E69F00"),
                                 ("final_assembly_bytes", "One final assembly", "#009E73")]:
        values = np.array([aggregate[m][field]["mean"] for m in methods]) / 2 ** 20
        axes[0].barh(positions, values, left=left, label=label, color=color)
        left += values
    for i, method in enumerate(methods):
        mean, sd = mean_sd(method, "training_factor_bytes")
        axes[1].barh(i, mean / 2 ** 20, xerr=sd / 2 ** 20, color=COLORS[method], capsize=2)
    axes[0].set_yticks(positions, [LABELS[m] for m in methods])
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    axes[0].set_xlabel("Deployment payload (MiB)")
    axes[0].set_title("Training + control + final assembly")
    axes[1].set_xlabel("Training factor payload (MiB)")
    axes[1].set_title("Factor exchange component; mean ± sample SD")
    save(fig, "communication")

    adaptive_methods = [m for m in ("adaptive_quality", "adaptive_domain") if m in methods]
    if adaptive_methods:
        fig, axes = plt.subplots(1, len(adaptive_methods), figsize=(5.2 * len(adaptive_methods), 3.4), sharey=True, squeeze=False)
        for ax, method in zip(axes[0], adaptive_methods):
            for ceiling, color in [(4, "#0072B2"), (8, "#009E73"), (16, "#D55E00")]:
                trajectories = []
                for seed in seeds:
                    record = indexed[(seed, method)]
                    peers = [str(cid) for cid, maximum in record["capacity_ceilings"].items() if maximum == ceiling]
                    trajectories.append([np.mean([row["ranks"][cid] for cid in peers]) for row in record["rounds"]])
                matrix = np.asarray(trajectories)
                mean = matrix.mean(axis=0)
                sd = matrix.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(mean)
                rounds = np.arange(1, len(mean) + 1)
                ax.axhline(ceiling, color=color, alpha=.3, ls=":", lw=1)
                ax.plot(rounds, mean, color=color, label=f"Capability ceiling {ceiling}", lw=1.8)
                ax.fill_between(rounds, mean - sd, mean + sd, color=color, alpha=.13)
            ax.set_xlabel("Training round")
            ax.set_title(LABELS[method].replace("\n", ": "))
            ax.set_yticks([2, 4, 6, 8, 12, 16])
            ax.legend(fontsize=8, frameon=False)
        axes[0, 0].set_ylabel("Mean rank within capability tier")
        save(fig, "rank_trajectories")
    return files


def markdown_report(summary):
    def format_stat(item):
        return f"{item['mean']:.2f} ± {item['sample_sd']:.2f}" if item["sample_sd"] is not None else f"{item['mean']:.2f} (n=1)"
    lines = ["# Corrected integrated experiment: measured results", "",
             summary["scope"] + ".", "", summary["uncertainty"] + ". " + summary["inference"] + ".", "",
             "| Arm | Full-test accuracy (%) | Paired difference vs pooled (pp) | Seeds |",
             "|---|---:|---:|---:|"]
    for row in summary["aggregate"]:
        lines.append(f"| {LABELS[row['method']].replace(chr(10), ' ')} | {format_stat(row['accuracy_percent'])} | {format_stat(row['paired_difference_vs_pooled_pp'])} | {row['n_seeds']} |")
    lines += ["", "The differences use matching seeds before aggregation. Positive values favor the peer arm; negative values favor pooled training. A sample SD is not an equivalence confidence interval.", "",
              "| Arm | Training factors (MiB) | Control (MiB) | Final assembly (MiB) | Total deployment (MiB) | Evaluation-only assembly (MiB) |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in summary["aggregate"]:
        values = [row[k]["mean"] / 2 ** 20 for k in ("training_factor_bytes", "control_bytes", "final_assembly_bytes", "deployment_payload_bytes", "evaluation_only_assembly_bytes")]
        lines.append(f"| {LABELS[row['method']].replace(chr(10), ' ')} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    lines += ["", summary["communication_scope"] + ".", "", summary["optimizer_scope"] + ".", "",
              summary["resource_scope"] + ".", "", summary["privacy_scope"] + ".", ""]
    return "\n".join(lines)


def latex_table(summary):
    def format_stat(item, signed=False):
        mean = f"{item['mean']:+.2f}" if signed else f"{item['mean']:.2f}"
        return rf"${mean} \pm {item['sample_sd']:.2f}$" if item["sample_sd"] is not None else rf"${mean}$"
    rows = [r"% Generated by scripts/summarize_integrated.py from validated completed runs.",
            r"\begin{table*}[t]", r"\centering", r"\small", r"\begin{tabular}{lrrr}", r"\hline",
            r"Arm & Full-test accuracy (\%) & Paired gap to pooled (pp) & Seeds \\", r"\hline"]
    for row in summary["aggregate"]:
        label = LABELS[row["method"]].replace("\n", "; ")
        rows.append(f"{label} & {format_stat(row['accuracy_percent'])} & {format_stat(row['paired_difference_vs_pooled_pp'], True)} & {row['n_seeds']} " + r"\\")
    rows += [r"\hline", r"\end{tabular}",
             r"\caption{Corrected integrated comparison on the full CIFAR-100 test set. Each entry is the mean $\pm$ sample standard deviation across seeds. Differences are calculated within matching seeds against pooled rank-16 LoRA with persistent Adam moments. Peer results evaluate the final rank-16 assembly. The backbone is frozen; peer communication is simulated in one process. No noninferiority or equivalence margin was prespecified.}",
             r"\label{tab:integrated-corrected}", r"\end{table*}"]
    if summary["factorial_aggregate"]:
        rows += ["", r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lr}", r"\hline",
                 r"Paired contrast & Accuracy difference (pp) \\", r"\hline"]
        effects = [("rank_effect_without_domain_pp", "Adaptive minus fixed; quality only"),
                   ("rank_effect_with_domain_pp", "Adaptive minus fixed; domain weighted"),
                   ("domain_effect_fixed_rank_pp", "Domain minus quality; fixed ranks"),
                   ("domain_effect_adaptive_rank_pp", "Domain minus quality; adaptive ranks"),
                   ("interaction_pp", r"Rank $\times$ domain interaction")]
        for key, label in effects:
            rows.append(f"{label} & {format_stat(summary['factorial_aggregate'][key], True)} " + r"\\")
        rows += [r"\hline", r"\end{tabular}",
                 r"\caption{Paired factorial contrasts, mean $\pm$ sample standard deviation across seeds. Positive values favor the first condition. All four arms use the same graph and sample-times-quality base allocation. The interaction is $(A_D-F_D)-(A_Q-F_Q)$. These are descriptive contrasts, not equivalence tests.}",
                 r"\label{tab:integrated-factorial}", r"\end{table}"]
    return "\n".join(rows) + "\n"


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "docs/artifacts/integrated-corrected")
    parser.add_argument("--output", type=Path, help="Defaults to input directory")
    parser.add_argument("--figures", type=Path, default=root / "paper/figures")
    parser.add_argument("--latex", type=Path, default=root / "paper/integrated_results.tex")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--expected-train", type=int, default=50000)
    parser.add_argument("--expected-test", type=int, default=10000)
    args = parser.parse_args()
    output = args.output or args.input
    paths = sorted(args.input.glob("seed*_*.json"))
    records = [json.loads(path.read_text()) for path in paths]
    manifest_path = args.input / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete":
            raise ValueError("manifest is not complete; final research summaries require completed runs")
    summary = summarize_records(records, methods=args.methods, seeds=args.seeds,
                                expected_train=args.expected_train, expected_test=args.expected_test)
    summary["input_files"] = [{"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
    summary["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if manifest_path.exists():
        summary["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    summary["figure_files"] = figures(summary, records, args.figures)
    curves, ranks = curve_tables(records)
    write_csv(output / "summary.csv", flatten_aggregate(summary["aggregate"]))
    write_csv(output / "per_seed.csv", summary["per_seed"])
    write_csv(output / "factorial_per_seed.csv", summary["factorial_per_seed"])
    write_csv(output / "adaptive_resource_pairs.csv", summary["adaptive_resource_pairs"])
    write_csv(output / "curves.csv", curves)
    write_csv(output / "rank_trajectories.csv", ranks)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "RESULTS.md").write_text(markdown_report(summary))
    args.latex.parent.mkdir(parents=True, exist_ok=True)
    args.latex.write_text(latex_table(summary))
    print(f"Validated {len(records)} completed runs; wrote summary tables and {len(summary['figure_files'])} figures to {args.figures}")


if __name__ == "__main__":
    main()
