#!/usr/bin/env python3
"""Create four publication figures from the completed quantity-skew campaign.

Run on gpu003 only (all numerical analysis and plotting are remote)::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/ahlora-venv/bin/python \
      scripts/plot_quantity_paper.py \
      --summary ~/ahlora-quantity-20260917/campaign-summary/summary.json \
      --output ~/ahlora-quantity-20260917/final-figures

Requires the complete seven-arm, five-seed SST-2 quantity extension, identical
scientific budget/source, verified best/final checkpoints and exact seed pairing.
Smoke and the single-seed equal-size anchor are retained in provenance but never
pooled into these figures. Reads run-level summary and raw rounds; independently
recomputes validation accuracy from saved predictions/labels and rechecks all
input hashes against the audited campaign summary. Does not train or modify runs.

Outputs four PDF/PNG figure pairs, matching JSON data sidecars, tidy endpoint/
curve/effect/resource CSVs, statistics.json, README.md, and provenance.json.
Best validation accuracy is the preregistered paper-comparison endpoint; final
accuracy is secondary. Displayed uncertainty is sample SD or unadjusted paired
Student-t 95% intervals as labeled, never a claim of equivalence or hidden-test
generalization. Decimal GB includes serialized training tensors and metadata,
including classifier tensors; setup and evaluation/final assembly are separate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import socket

import numpy as np
from scipy.stats import t


SEEDS = (42, 43, 44, 45, 46)
ARMS = ("declora16", "declora4", "product16_sample", "fixed_uniform", "fixed_sample", "adaptive_uniform", "adaptive_sample")
LABELS = {
    "declora16": "Dec-LoRA r16*", "declora4": "Dec-LoRA r4",
    "product16_sample": "Product r16 + size*", "fixed_uniform": "Fixed + uniform",
    "fixed_sample": "Fixed + size", "adaptive_uniform": "Adaptive + uniform",
    "adaptive_sample": "Adaptive + size (ours)",
}
COLORS = dict(zip(ARMS, ("#222222", "#6B6B6B", "#0072B2", "#009E73", "#56B4E9", "#CC79A7", "#D55E00")))
COMPARISONS = {
    "ours_minus_declora16": {"adaptive_sample": 1, "declora16": -1},
    "ours_minus_declora4": {"adaptive_sample": 1, "declora4": -1},
    "ours_minus_product16": {"adaptive_sample": 1, "product16_sample": -1},
    "adaptation_size": {"adaptive_sample": 1, "fixed_sample": -1},
    "adaptation_uniform": {"adaptive_uniform": 1, "fixed_uniform": -1},
    "weighting_fixed": {"fixed_sample": 1, "fixed_uniform": -1},
    "weighting_adaptive": {"adaptive_sample": 1, "adaptive_uniform": -1},
    "adaptation_main": {"adaptive_sample": .5, "fixed_sample": -.5, "adaptive_uniform": .5, "fixed_uniform": -.5},
    "weighting_main": {"adaptive_sample": .5, "adaptive_uniform": -.5, "fixed_sample": .5, "fixed_uniform": -.5},
    "interaction": {"adaptive_sample": 1, "adaptive_uniform": -1, "fixed_sample": -1, "fixed_uniform": 1},
}
EFFECT_LABELS = {
    "ours_minus_declora16": "Ours − Dec-LoRA r16*", "ours_minus_declora4": "Ours − Dec-LoRA r4",
    "ours_minus_product16": "Ours − Product r16*", "adaptation_size": "Adaptive − fixed | size",
    "adaptation_uniform": "Adaptive − fixed | uniform", "weighting_fixed": "Size − uniform | fixed",
    "weighting_adaptive": "Size − uniform | adaptive", "interaction": "Rank × weighting interaction",
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def statistics(values):
    values = np.asarray(values, dtype=float)
    require(values.shape == (5,) and np.isfinite(values).all(), "exactly five finite paired seed values required")
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    halfwidth = float(t.ppf(.975, 4)) * sd / math.sqrt(5)
    return {"n": 5, "seeds": list(SEEDS), "values": values.tolist(), "mean": mean, "sample_sd": sd,
            "ci95_unadjusted_t": [mean - halfwidth, mean + halfwidth], "df": 4}


def load(summary_path):
    summary = read(summary_path)
    candidates = [group for group in summary["groups"]
                  if group["compatibility"]["classification"] == "full_configured_budget"
                  and group["compatibility"]["track"] == "quantity_skew_extension"
                  and group["compatibility"]["config"]["task"] == "sst2"]
    require(len(candidates) == 1, "one compatible completed SST-2 quantity group is required")
    group = candidates[0]
    require(group["seeds"] == list(SEEDS), "final figures require all predeclared seeds42–46")
    require(group["compatibility"]["config"]["rounds"] == 20, "expected frozen20-round study")
    selected = [run for run in summary["runs"] if run.get("comparison_group") == group["id"]]
    require(len(selected) == len(SEEDS) * len(ARMS), "seven arms × five seeds required, no missing or duplicate runs")
    records, hashes, endpoints, curves = {}, {}, [], []
    for run in selected:
        arm, seed = run["arm"], run["seed"]
        require(arm in ARMS and seed in SEEDS and (arm, seed) not in records, "unexpected or duplicate run")
        require(run["status"] == "complete" and not run["warnings"], "all selected runs must be complete and warning-free")
        verification = run["checkpoint_verification"]
        require(verification["status"] == "passed" and verification["kind"] == "standalone_independent_evaluator"
                and set(verification["verified_endpoints"]) == {"best", "final"}, "best/final independent checkpoint audits required")
        path = Path(run["path"])
        for filename, expected in run["input_artifacts_sha256"].items():
            actual = sha(path / filename)
            require(actual == expected, f"run artifact changed after campaign audit: {path}/{filename}")
            hashes[str(path / filename)] = actual
        audit_path = path / "independent_checkpoint_audit.json"
        require(sha(audit_path) == verification["audit_sha256"], "independent audit changed after campaign summary")
        hashes[str(audit_path)] = sha(audit_path)
        raw_summary, split = read(path / "summary.json"), read(path / "split.json")
        labels = np.asarray(split["validation_labels"], dtype=np.int64)
        require(len(labels) == 872, "full official SST-2 validation split required")
        rows = [json.loads(line) for line in (path / "rounds.jsonl").read_text().splitlines() if line.strip()]
        require([row["round"] for row in rows] == list(range(1, 21)), "all20 rounds in order required")
        accuracies, mean_ranks, rank_products = [], [], 0
        head_bytes = adapter_bytes = metadata_bytes = 0
        for row in rows:
            pred = np.asarray(row["validation"]["predictions"], dtype=np.int64)
            require(pred.shape == labels.shape and np.isin(pred, [0, 1]).all(), "invalid saved predictions")
            accuracy = 100 * int(np.count_nonzero(pred == labels)) / len(labels)
            require(abs(accuracy - row["validation"]["accuracy"]) < 1e-10, "saved validation accuracy disagrees with predictions")
            ranks = np.asarray(row["ranks"], dtype=int)
            require(ranks.shape == (10,), "expected ten peer ranks")
            examples = np.asarray([row["local"][str(cid)]["examples"] for cid in range(10)], dtype=int)
            rank_products += int(ranks @ examples)
            accuracies.append(accuracy)
            mean_ranks.append(float(ranks.mean()))
            # Every ring peer transmits to its two neighbors. Q/V adapters in
            # all12 blocks have24*(768+768)*r fp32 elements; the raw ledger
            # independently records the combined tensor payload.
            row_adapter_bytes = int(2 * 24 * (768 + 768) * 4 * ranks.sum())
            row_head_bytes = row["gossip"]["tensor_bytes"] - row_adapter_bytes
            require(row_head_bytes >= 0, "adapter traffic exceeds actual tensor payload")
            require(row_adapter_bytes == 2 * sum(state["adapter"] for state in row["state_bytes"].values())
                    and row_head_bytes == 2 * sum(state["head"] for state in row["state_bytes"].values()),
                    "traffic components do not match independently recorded factor/head tensor storage")
            adapter_bytes += row_adapter_bytes
            head_bytes += row_head_bytes
            metadata_bytes += row["gossip"]["metadata_bytes"]
            curves.append({"arm": arm, "seed": seed, "round": row["round"],
                           "validation_accuracy": accuracy, "mean_peer_rank": float(ranks.mean()),
                           "training_bytes_cumulative": row["cumulative_training_bytes"],
                           "training_adapter_bytes": row_adapter_bytes, "training_head_bytes": row_head_bytes,
                           "training_metadata_bytes": row["gossip"]["metadata_bytes"],
                           **{f"rank_peer{cid}": int(ranks[cid]) for cid in range(10)}})
        require(abs(max(accuracies) - run["best_primary"]) < 1e-10 and abs(accuracies[-1] - run["final_primary"]) < 1e-10,
                "independent best/final accuracy disagrees with campaign summary")
        require(adapter_bytes + head_bytes + metadata_bytes == raw_summary["training_bytes"], "traffic decomposition does not match actual serialized total")
        require(raw_summary["total_steps"] == 42200 and raw_summary["total_examples"] == 1350400, "matched study training budget changed")
        endpoint = {"arm": arm, "seed": seed, "best_accuracy": max(accuracies), "final_accuracy": accuracies[-1],
                    "best_round": int(np.argmax(accuracies)) + 1, "training_bytes": raw_summary["training_bytes"],
                    "production_bytes": raw_summary["production_bytes_setup_training_final_assembly"],
                    "training_adapter_bytes": adapter_bytes, "training_head_bytes": head_bytes, "training_metadata_bytes": metadata_bytes,
                    "total_steps": raw_summary["total_steps"], "total_examples": raw_summary["total_examples"],
                    "probe_examples": raw_summary["total_probe_examples"], "rank_example_products": rank_products,
                    "mean_final_rank": mean_ranks[-1], "path": str(path)}
        endpoints.append(endpoint)
        records[(arm, seed)] = {"run": run, "endpoint": endpoint, "accuracy": accuracies, "mean_rank": mean_ranks,
                                "training_bytes": [row["cumulative_training_bytes"] for row in rows]}
    for seed in SEEDS:
        require(len({records[(arm, seed)]["run"]["pair_identity_sha256"] for arm in ARMS}) == 1,
                "same-seed methods have different shards, graph/caps, or realized sample streams")
    for effect in group["paired_effects"].values():
        require(effect["matched_seeds"] == list(SEEDS) and not effect["pairing_failures"], "campaign factorial pairing failed")
    return summary, group, records, endpoints, curves, hashes


def export_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compute(records):
    by_arm, effects, resources = {}, {}, {}
    for arm in ARMS:
        by_arm[arm] = {field: statistics([records[(arm, seed)]["endpoint"][field] for seed in SEEDS])
                       for field in ("best_accuracy", "final_accuracy", "training_bytes", "production_bytes", "training_adapter_bytes", "training_head_bytes", "training_metadata_bytes", "probe_examples", "rank_example_products", "mean_final_rank")}
    for name, coefficients in COMPARISONS.items():
        effects[name] = {"coefficients": coefficients}
        for endpoint in ("best_accuracy", "final_accuracy"):
            effects[name][endpoint] = statistics([sum(weight * records[(arm, seed)]["endpoint"][endpoint] for arm, weight in coefficients.items()) for seed in SEEDS])
    for comparator in ("declora16", "declora4", "fixed_sample"):
        resources[comparator] = {field + "_reduction_percent": statistics([
            100 * (1 - records[("adaptive_sample", seed)]["endpoint"][field] / records[(comparator, seed)]["endpoint"][field])
            for seed in SEEDS]) for field in ("training_bytes", "production_bytes", "training_adapter_bytes", "rank_example_products")}
    return {"seeds": list(SEEDS), "arms": by_arm, "paired_effects_pp": effects,
            "ours_relative_resource_reductions": resources,
            "uncertainty": "sampleSD; displayed effect intervals are unadjusted two-sided Student-t95%,df4; multiple exploratory contrasts; no equivalence margin or hidden-test estimate"}


def save_figure(fig, output, stem, data, files):
    for suffix in ("pdf", "png"):
        path = output / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        files.append(path.name)
    write_json(output / f"{stem}.data.json", data)
    files.append(f"{stem}.data.json")


def plot(output, records, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 11,
                         "axes.labelsize": 10.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})
    files, jitter = [], np.linspace(-.13, .13, 5)
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 4.4), sharey=True)
    for axis, endpoint, title in zip(axes, ("best_accuracy", "final_accuracy"), ("(a) Best accuracy (primary)", "(b) Final accuracy (secondary)")):
        for index, arm in enumerate(ARMS):
            stat = result["arms"][arm][endpoint]
            axis.scatter(stat["values"], index + jitter, s=16, facecolors="none", edgecolors=COLORS[arm], linewidths=.8, zorder=2)
            axis.errorbar(stat["mean"], index, xerr=stat["sample_sd"], color=COLORS[arm], fmt="o", ms=5, capsize=3, lw=1.5, zorder=3)
        axis.set(title=title, xlabel="Validation accuracy (%)", xlim=(92.85, 95.1))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.grid(axis="x", alpha=.2)
        axis.set_yticks(range(len(ARMS)), [LABELS[arm] for arm in ARMS])
    axes[0].invert_yaxis()
    figure.text(.5, -.035, "Five matched seeds: mean ± sample SD and individual seed points.\n*Uniform r16 exceeds weaker clients' training rank caps.", ha="center", fontsize=9.5)
    figure.tight_layout(w_pad=2)
    save_figure(figure, output, "quantity-skew-accuracy", {"scope": "35 quantity runs only; best over20 rounds and final; mean±sampleSD", "arms": result["arms"]}, files)
    plt.close(figure)

    keys = ("ours_minus_declora16", "ours_minus_declora4", "ours_minus_product16", "adaptation_size", "adaptation_uniform", "weighting_fixed", "weighting_adaptive", "interaction")
    figure, axes = plt.subplots(1, 2, figsize=(8.7, 4.8), sharey=True)
    for axis, endpoint, title in zip(axes, ("best_accuracy", "final_accuracy"), ("(a) Best endpoint", "(b) Final endpoint")):
        axis.axvline(0, color="#888888", lw=1, ls="--")
        for index, key in enumerate(keys):
            stat = result["paired_effects_pp"][key][endpoint]
            color = "#D55E00" if index < 3 else "#0072B2"
            interval = np.asarray(stat["ci95_unadjusted_t"])
            axis.scatter(stat["values"], index + jitter, s=12, marker="x", color=color, alpha=.5)
            axis.errorbar(stat["mean"], index, xerr=np.array([[stat["mean"] - interval[0]], [interval[1] - stat["mean"]]]),
                          color=color, fmt="o", capsize=3, ms=4.5, lw=1.5)
        extrema = [value for key in keys for metric in ("best_accuracy", "final_accuracy")
                   for value in (result["paired_effects_pp"][key][metric]["values"]
                                 + result["paired_effects_pp"][key][metric]["ci95_unadjusted_t"])]
        lower, upper = min(extrema), max(extrema)
        padding = .07 * (upper - lower)
        axis.set(title=title, xlabel="Accuracy difference (pp)", xlim=(lower - padding, upper + padding))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.grid(axis="x", alpha=.2)
        axis.set_yticks(range(len(keys)), [EFFECT_LABELS[key] for key in keys])
    axes[0].invert_yaxis()
    figure.text(.5, -.04, "Mean ± unadjusted 95% paired t interval (n = 5, df = 4); crosses are seed differences.\nDifferences are percentage points. These are not equivalence tests.", ha="center", fontsize=9.3)
    figure.tight_layout(w_pad=2)
    save_figure(figure, output, "quantity-skew-paired-effects", {"displayed_effects": list(keys), "effects": result["paired_effects_pp"], "uncertainty": result["uncertainty"]}, files)
    plt.close(figure)

    figure, (axis, traffic_axis) = plt.subplots(1, 2, figsize=(8.8, 5.2), gridspec_kw={"width_ratios": [1.18, 1]})
    short_labels = {"declora16": "Dec-LoRA r16*", "declora4": "Dec-LoRA r4", "product16_sample": "Product r16*",
                    "fixed_uniform": "Fixed/uniform", "fixed_sample": "Fixed/size", "adaptive_uniform": "Adaptive/uniform", "adaptive_sample": "Ours"}
    offsets = {"declora16": (-74, -28), "declora4": (-2, 29), "product16_sample": (-92, 13),
               "fixed_uniform": (9, 8), "fixed_sample": (9, -16), "adaptive_uniform": (10, -23), "adaptive_sample": (10, -2)}
    for arm in ARMS:
        traffic, score = result["arms"][arm]["training_bytes"], result["arms"][arm]["best_accuracy"]
        x, y = np.asarray(traffic["values"]) / 1e9, np.asarray(score["values"])
        marker = "s" if arm in {"declora16", "product16_sample"} else "o"
        axis.scatter(x, y, marker=marker, s=16, alpha=.35, color=COLORS[arm])
        axis.errorbar(x.mean(), y.mean(), xerr=x.std(ddof=1), yerr=y.std(ddof=1), fmt=marker,
                      color=COLORS[arm], ms=6, capsize=3, lw=1.3)
        axis.annotate(short_labels[arm], (x.mean(), y.mean()), xytext=offsets[arm], textcoords="offset points",
                      color=COLORS[arm], fontsize=10)
    axis.set(title="(a) Accuracy–traffic tradeoff", xlabel="Training traffic (GB)",
             ylabel="Best validation accuracy (%)", xlim=(1.12, 2.01), ylim=(93.72, 95.1))
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.grid(alpha=.2)
    axis.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777", label="Within training rank caps"),
                         Line2D([0], [0], marker="s", color="none", markerfacecolor="#777777", label="Uniform r16 reference*")],
                fontsize=9.2, loc="upper left", frameon=False)
    head = np.array([result["arms"][arm]["training_head_bytes"]["mean"] / 1e9 for arm in ARMS])
    adapters = np.array([result["arms"][arm]["training_adapter_bytes"]["mean"] / 1e9 for arm in ARMS])
    metadata = np.array([result["arms"][arm]["training_metadata_bytes"]["mean"] / 1e9 for arm in ARMS])
    positions = np.arange(len(ARMS))
    traffic_axis.barh(positions, head, color="#B8BEC5", label="Classifier head")
    traffic_axis.barh(positions, adapters, left=head, color=[COLORS[arm] for arm in ARMS], label="LoRA factors")
    traffic_axis.barh(positions, metadata, left=head + adapters, color="#EFEFEF", label="Metadata")
    traffic_axis.set_yticks(positions, [short_labels[arm] for arm in ARMS])
    traffic_axis.invert_yaxis()
    traffic_axis.set(title="(b) Traffic components", xlabel="Mean training traffic (GB)", xlim=(0, 2.04))
    traffic_axis.set_ylim(8.35, -.65)
    traffic_axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    traffic_axis.grid(axis="x", alpha=.15)
    traffic_axis.legend(fontsize=9.2, loc="lower right", frameon=False)
    figure.text(.5, -.03, "Means ± sample SD; decimal GB includes classifier, LoRA, and metadata.\nSetup and assemblies are excluded. Ours uses 8.92% more bytes than rank 4.", ha="center", fontsize=9.3)
    figure.tight_layout(w_pad=3.0)
    save_figure(figure, output, "quantity-skew-tradeoff", {"arms": result["arms"], "relative_resource_reductions": result["ours_relative_resource_reductions"], "unit": "decimalGB", "scope": "actual serialized training traffic; excluding setup and assembly"}, files)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(8.7, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    selected_learning = ("declora16", "declora4", "product16_sample", "fixed_sample", "adaptive_sample")
    x = np.arange(1, 21)
    curve_data = {}
    for arm in ARMS:
        accuracy = np.asarray([records[(arm, seed)]["accuracy"] for seed in SEEDS])
        ranks = np.asarray([records[(arm, seed)]["mean_rank"] for seed in SEEDS])
        curve_data[arm] = {"rounds": x.tolist(), "accuracy_by_seed": accuracy.tolist(), "mean_rank_by_seed": ranks.tolist(),
                           "accuracy_mean": accuracy.mean(0).tolist(), "accuracy_sample_sd": accuracy.std(0, ddof=1).tolist(),
                           "mean_rank_mean": ranks.mean(0).tolist(), "mean_rank_sample_sd": ranks.std(0, ddof=1).tolist()}
        if arm in selected_learning:
            axes[0].plot(x, accuracy.mean(0), color=COLORS[arm], lw=1.6, label=LABELS[arm])
            axes[0].fill_between(x, accuracy.mean(0) - accuracy.std(0, ddof=1), accuracy.mean(0) + accuracy.std(0, ddof=1), color=COLORS[arm], alpha=.11)
        if arm in {"fixed_sample", "adaptive_uniform", "adaptive_sample"}:
            axes[1].plot(x, ranks.mean(0), color=COLORS[arm], lw=1.6, label=LABELS[arm], drawstyle="steps-mid")
            axes[1].fill_between(x, ranks.mean(0) - ranks.std(0, ddof=1), ranks.mean(0) + ranks.std(0, ddof=1), color=COLORS[arm], alpha=.13, step="mid")
    axes[0].set(title="(a) Learning curves", xlabel="Communication round", ylabel="Validation accuracy (%)")
    axes[1].set(title="(b) Training ranks", xlabel="Communication round", ylabel="Mean rank across ten peers", ylim=(3.5, 9.6))
    for axis in axes:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        axis.set_xlim(1, 20)
        axis.grid(alpha=.2)
        axis.legend(fontsize=9.2, loc="lower right" if axis is axes[0] else "upper right", frameon=False)
    figure.text(.5, -.035, "Five seeds: mean ± between-seed sample SD. Fixed heterogeneous mean rank is 8.8.\nAdaptive probes still run at the original capability ceiling; no peak-memory reduction is claimed.", ha="center", fontsize=9.1)
    figure.tight_layout(w_pad=2.5)
    save_figure(figure, output, "quantity-skew-trajectories", {"curves": curve_data, "scope": "all20 rounds; rank mean over10peers then seedmean/sampleSD; no peakmemory claim"}, files)
    plt.close(figure)
    return files, {"matplotlib": matplotlib.__version__, "numpy": np.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(socket.gethostname().split(".")[0] == "gpu003", "Scientific analysis and plotting must run on gpu003")
    args.output.mkdir(parents=True, exist_ok=True)
    summary, group, records, endpoints, curves, hashes = load(args.summary)
    result = compute(records)
    write_json(args.output / "statistics.json", result)
    export_csv(args.output / "endpoints.csv", sorted(endpoints, key=lambda row: (ARMS.index(row["arm"]), row["seed"])))
    export_csv(args.output / "curves.csv", sorted(curves, key=lambda row: (ARMS.index(row["arm"]), row["seed"], row["round"])))
    effect_rows = []
    for name, effect in result["paired_effects_pp"].items():
        for endpoint in ("best_accuracy", "final_accuracy"):
            stat = effect[endpoint]
            for seed, value in zip(SEEDS, stat["values"]):
                effect_rows.append({"effect": name, "endpoint": endpoint, "seed": seed, "difference_pp": value,
                                    "paired_mean": stat["mean"], "sample_sd": stat["sample_sd"],
                                    "ci95_low": stat["ci95_unadjusted_t"][0], "ci95_high": stat["ci95_unadjusted_t"][1]})
    export_csv(args.output / "paired-effects.csv", effect_rows)
    figure_files, versions = plot(args.output, records, result)
    excluded = [{"group_id": other["id"], "track": other["compatibility"]["track"],
                 "classification": other["compatibility"]["classification"], "n_runs": other["n_runs"]}
                for other in summary["groups"] if other["id"] != group["id"]]
    provenance = {"schema_version": 1, "host": socket.gethostname(), "python": platform.python_version(), "versions": versions,
                  "analysis_script": str(Path(__file__).resolve()), "analysis_script_sha256": sha(__file__),
                  "campaign_summary": str(args.summary.resolve()), "campaign_summary_sha256": sha(args.summary),
                  "included_group": group["id"], "included_seeds": list(SEEDS), "included_arms": list(ARMS),
                  "included_runs": 35, "rounds_independently_rescored": 700, "verified_checkpoint_audits": 70,
                  "source_sha256": group["compatibility"]["source_sha256"], "excluded_groups": excluded,
                  "input_files_sha256": hashes, "figure_files_sha256": {name: sha(args.output / name) for name in figure_files},
                  "independent_scope": "Recomputed700round accuracies from saved predictions/validation labels, rechecked immutable input hashes and paired identities;70best/final model-inference audits already completed independently. Shared standard model/install helper remains disclosed."}
    write_json(args.output / "provenance.json", provenance)
    lines = ["# Final quantity-skew figures", "", "Generated on gpu003 from 35 complete quantity-extension runs: seven methods × seeds 42–46. The seven smoke runs and single equal-size anchor are excluded from all figures. All 700 round accuracies were independently recomputed from predictions and labeled validation targets; 70 best/final checkpoint audit bindings and artifact hashes were rechecked.", "",
             "Four figures: accuracy (primary best / secondary final), paired effects (unadjusted 95% t intervals), accuracy/serialized-training-traffic tradeoff plus classifier/adapter decomposition, and learning/rank trajectories. Each PDF/PNG pair has a matching `.data.json`; raw derived numbers are in CSV and statistics.json. Exact source and artifact hashes are in provenance.json.", "",
             "All outcomes are official labeled SST-2 validation, not hidden-test generalization. Best accuracy selects the best of 20 rounds. Intervals are exploratory seed-paired t intervals with n = 5, df = 4; they are unadjusted for multiple contrasts and do not establish parity. Dec-LoRA is our independent paper-based reimplementation, not author code. Uniform rank-16 references exceed weaker client caps. Rank 4 is feasible and must remain visible: our method uses 8.92% more training bytes than rank 4 and has lower mean accuracy.", "",
             "Training traffic includes actual serialized LoRA factors, classifier tensors, and metadata; setup and evaluation/final assembly are excluded from this axis and separately logged. The classifier traffic is substantial and shared across arms. Rank trajectories measure persistent training rank, not total or peak client memory; adaptive probes run at the original capacity ceiling. This is a single-process explicit peer-payload simulation without formal privacy protection.", "",
             "Reproduce with `scripts/plot_quantity_paper.py --summary <campaign-summary/summary.json> --output <final-figures>` on gpu003.", ""]
    (args.output / "README.md").write_text("\n".join(lines))
    print(json.dumps({"included_runs": 35, "seeds": list(SEEDS), "figures": figure_files,
                      "ours_best": result["arms"]["adaptive_sample"]["best_accuracy"],
                      "ours_minus_declora16": result["paired_effects_pp"]["ours_minus_declora16"]}, indent=2))


if __name__ == "__main__":
    main()
