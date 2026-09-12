"""Aggregate completed Experiment 04 runs without selecting favorable results.

Usage:
    python experiments/summarize_completion.py --inputs results/main results/hetero \
        --output results/completion-report

Different training protocols remain separate comparison contexts. Within a
context, every method/merge/feedback arm is retained, including negative paired
differences. Standard deviations use the sample denominator (n-1). Single-seed
standard deviations are null; no significance or superiority test is implied.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import time


FINAL_METRICS = (
    "personalized_accuracy", "personalized_sample_weighted_accuracy", "consensus_accuracy",
    "accuracy_gap", "worst_domain_accuracy", "consensus_distance",
    "total_effective_messages", "total_effective_floats", "total_operational_messages",
    "total_operational_floats", "wall_seconds",
)
ROUND_METRICS = (
    "personalized_accuracy", "personalized_sample_weighted_accuracy", "consensus_accuracy",
    "accuracy_gap", "worst_domain_accuracy", "consensus_distance", "mean_train_loss",
    "mean_tail_mass", "max_tail_mass", "mean_residual_energy", "mean_merge_error_energy",
    "mean_relative_merge_error", "effective_messages", "effective_floats",
    "operational_messages", "operational_floats", "wall_seconds",
)
# These change the training/evaluation population or optimizer and cannot be
# silently pooled. Transport topology and merge choice identify separate arms.
CONTEXT_FIELDS = (
    "rounds", "ranks", "alpha", "consensus_rank", "n_domains", "clients_per_domain",
    "dirichlet_alpha", "local_epochs", "lr", "weight_decay", "batch_size",
    "image_size", "max_train_per_client", "max_test_per_domain",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def statistics_for(values):
    values = [float(value) for value in values if value is not None]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("non-finite numeric result cannot be aggregated")
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def context_for(manifest):
    config = manifest["config"]
    protocol = {key: config.get(key) for key in CONTEXT_FIELDS}
    protocol["consensus_rank"] = config.get("consensus_rank") or max(config["ranks"])
    cache = manifest.get("feature_cache", {})
    protocol["feature_sha256"] = {split: cache.get(split, {}).get("sha256") for split in ("train", "test")}
    protocol["feature_cache_identity_sha256"] = cache.get("cache_identity_sha256")
    # A report/diagnostic file added between runs should not split otherwise
    # identical contexts. The numerical runner/model and benchmark driver must
    # still match; a missing hash is explicitly retained as unknown provenance.
    source = manifest.get("source", {}).get("files_sha256", {})
    protocol["behavior_source_sha256"] = {key: value for key, value in source.items()
        if key in {"experiments/protocol_benchmark.py", "experiments/feature_cache.py"}
        or key.startswith("src/federated/") or key.startswith("src/models/")
        or key == "src/data/cifar100_domains.py"}
    return digest(protocol)[:12], protocol


def variant_for(record, manifest):
    config = manifest["config"]
    variant = {"method": record["method"], "merge": record["merge"],
               "error_feedback": bool(record["error_feedback"]),
               "topology": config.get("topology") if record["method"] == "mh" else None,
               "bridge_every": config.get("bridge_every") if record["method"] == "oracle" else None}
    method = {"local": "Local", "fedavg": "FedAvg", "mh": "MH", "oracle": "Oracle"}.get(record["method"], record["method"])
    label = f"{method} / {'ΔW' if record['merge'] == 'delta' else 'factor zero-pad'}"
    if variant["error_feedback"]:
        label += " / feedback"
    if variant["topology"]:
        label += f" / {variant['topology']}"
    if variant["bridge_every"]:
        label += f" / bridge {variant['bridge_every']}"
    return digest(variant)[:12], variant, label


def load_inputs(paths):
    entries, inventories, contexts, seen = [], [], {}, {}
    for directory in paths:
        directory = Path(directory).resolve()
        manifest_path, results_path = directory / "manifest.json", directory / "results.jsonl"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("experiment") == "05_signature_validation":
            raise ValueError(f"{directory} is Experiment 05; this report accepts Experiment 04 only")
        context_id, context = context_for(manifest)
        contexts[context_id] = context
        records = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()] if results_path.exists() else []
        inventory = {"directory": str(directory), "manifest_status": manifest.get("status"),
            "run_classification": manifest.get("run_classification"),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest() if results_path.exists() else None,
            "context_id": context_id, "expected_runs": len(manifest["config"]["seeds"]) * len(manifest["config"]["methods"]),
            "completed_runs": len(records), "config": manifest["config"],
            "partial_records": []}
        expected = set(itertools.product(manifest["config"]["seeds"], manifest["config"]["methods"]))
        observed = {(record.get("seed"), record.get("method")) for record in records}
        inventory["missing_runs"] = [{"seed": seed, "method": method}
                                     for seed, method in sorted(expected - observed)]
        inventory["completion_consistent"] = not (manifest.get("status") == "complete" and expected != observed)
        for path in sorted(directory.glob("seed*_*.json")):
            partial = json.loads(path.read_text())
            if partial.get("status") != "complete":
                inventory["partial_records"].append({"file": path.name, "status": partial.get("status"),
                    "completed_rounds": len(partial.get("rounds", []))})
        for record in records:
            if record.get("status") != "complete" or not record.get("rounds") or "summary" not in record:
                raise ValueError(f"{results_path} contains an incomplete/non-Experiment04 record")
            if len(record["rounds"]) != manifest["config"]["rounds"]:
                raise ValueError(f"{results_path}: completed round count differs from configuration")
            if record["seed"] not in manifest["config"]["seeds"] or record["method"] not in manifest["config"]["methods"]:
                raise ValueError(f"{results_path}: seed/method absent from manifest")
            variant_id, variant, label = variant_for(record, manifest)
            key = (context_id, variant_id, record["seed"])
            if key in seen:
                raise ValueError(f"duplicate context/arm/seed would double-count evidence: {seen[key]} and {directory}, {key}")
            seen[key] = str(directory)
            entries.append({"context_id": context_id, "variant_id": variant_id,
                "variant": variant, "label": label, "seed": record["seed"],
                "directory": str(directory), "classification": manifest.get("run_classification"),
                "record": record})
        inventories.append(inventory)
    if not entries:
        raise ValueError("no completed Experiment 04 runs were found; partial runs remain in their source directories")
    return entries, inventories, contexts


def aggregate(entries):
    groups, final_rows, round_rows, domain_rows = {}, [], [], []
    for entry in entries:
        key = (entry["context_id"], entry["variant_id"])
        groups.setdefault(key, []).append(entry)
        common = {key: entry[key] for key in ("context_id", "variant_id", "label", "seed", "directory", "classification")}
        final_rows.append({**common, **{metric: entry["record"]["summary"].get(metric) for metric in FINAL_METRICS},
                           "split_sha256": entry["record"]["split_sha256"],
                           "initial_state_sha256": entry["record"]["initial_state_sha256"]})
        cumulative = {name: 0 for name in ("effective_floats", "operational_floats")}
        for row in entry["record"]["rounds"]:
            for name in cumulative:
                cumulative[name] += row[name]
            round_rows.append({**common, "round": row["round"],
                **{metric: row.get(metric) for metric in ROUND_METRICS},
                **{f"cumulative_{name}": value for name, value in cumulative.items()}})
            for domain, accuracy in row["per_domain_accuracy"].items():
                domain_rows.append({**common, "round": row["round"], "domain_id": str(domain), "accuracy": accuracy})
    group_records, metric_rows = [], []
    for (context_id, variant_id), members in sorted(groups.items()):
        first = members[0]
        metrics = {metric: statistics_for([member["record"]["summary"].get(metric) for member in members]) for metric in FINAL_METRICS}
        group_records.append({"context_id": context_id, "variant_id": variant_id, "label": first["label"],
                              "variant": first["variant"], "seeds": sorted(member["seed"] for member in members),
                              "n_seeds": len(members), "metrics": metrics})
        metric_rows.extend({"context_id": context_id, "variant_id": variant_id, "label": first["label"],
                            "metric": metric, **summary} for metric, summary in metrics.items())
    return groups, group_records, metric_rows, final_rows, round_rows, domain_rows


def paired_differences(groups):
    """All within-context arm pairs, with exact split/init checks per seed."""
    rows, summaries, exclusions = [], [], []
    contexts = sorted({key[0] for key in groups})
    for context in contexts:
        variants = sorted(key[1] for key in groups if key[0] == context)
        for left, right in itertools.combinations(variants, 2):
            left_entries = {entry["seed"]: entry for entry in groups[(context, left)]}
            right_entries = {entry["seed"]: entry for entry in groups[(context, right)]}
            values = {metric: [] for metric in FINAL_METRICS}
            left_label, right_label = next(iter(left_entries.values()))["label"], next(iter(right_entries.values()))["label"]
            identity = {"context_id": context, "left_variant_id": left, "right_variant_id": right,
                        "left_label": left_label, "right_label": right_label}
            for seed in sorted(set(left_entries) | set(right_entries)):
                if seed not in left_entries or seed not in right_entries:
                    exclusions.append({**identity, "seed": seed, "reason": "seed missing from one arm"})
                    continue
                a, b = left_entries[seed]["record"], right_entries[seed]["record"]
                if any(a[field] != b[field] for field in ("split_sha256", "initial_state_sha256")):
                    exclusions.append({**identity, "seed": seed, "reason": "initialization or split checksum mismatch"})
                    continue
                for metric in FINAL_METRICS:
                    a_value, b_value = a["summary"].get(metric), b["summary"].get(metric)
                    if a_value is None or b_value is None:
                        continue
                    difference = b_value - a_value
                    values[metric].append(difference)
                    rows.append({**identity, "seed": seed, "metric": metric, "left_value": a_value,
                                 "right_value": b_value, "difference_right_minus_left": difference})
            for metric, differences in values.items():
                summaries.append({**identity, "metric": metric, **statistics_for(differences)})
    return rows, summaries, exclusions


def aggregate_trajectories(rows, metrics, domain=False):
    groups = {}
    for row in rows:
        key = (row["context_id"], row["variant_id"], row["label"], row["round"])
        if domain:
            key += (row["domain_id"],)
        groups.setdefault(key, []).append(row)
    result = []
    for key, members in sorted(groups.items()):
        for metric in metrics:
            identity = dict(zip(("context_id", "variant_id", "label", "round"), key[:4]))
            if domain:
                identity["domain_id"] = key[4]
            result.append({**identity, "metric": metric,
                           **statistics_for([member.get(metric) for member in members])})
    return result


def plot_reports(output, contexts, group_records, trajectories, domains):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 160})
    files = []
    for context_id, context in sorted(contexts.items()):
        relevant = [group for group in group_records if group["context_id"] == context_id]
        if not relevant:
            continue
        panels = [("personalized_accuracy", "Personalized client mean", "Accuracy"),
                  ("consensus_accuracy", "Merged consensus model", "Accuracy"),
                  ("cumulative_effective_floats", "Direct contributor payload", "Cumulative fp32 floats"),
                  ("cumulative_operational_floats", "Exact staged payload", "Cumulative fp32 floats")]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        colors = plt.get_cmap("tab10")
        for index, group in enumerate(relevant):
            for axis, (metric, title, ylabel) in zip(axes.flat, panels):
                rows = sorted([row for row in trajectories if row["context_id"] == context_id
                    and row["variant_id"] == group["variant_id"] and row["metric"] == metric], key=lambda x: x["round"])
                x = [row["round"] for row in rows]
                y = np.asarray([row["mean"] for row in rows])
                sd = np.asarray([row["sample_std"] or 0 for row in rows])
                axis.plot(x, y, label=f"{group['label']} (n={group['n_seeds']})", color=colors(index % 10))
                if group["n_seeds"] > 1:
                    axis.fill_between(x, y - sd, y + sd, alpha=0.12, color=colors(index % 10))
                axis.set(title=title, xlabel="Communication round", ylabel=ylabel)
                axis.grid(alpha=0.18)
                if "accuracy" in metric:
                    axis.set_ylim(0, 1)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.075), ncol=2, fontsize=8)
        fig.suptitle(f"Frozen-feature CIFAR-100 • ranks {context['ranks']} • {context['rounds']} rounds\n"
                     "Mean ± sample SD across seeds; communication is simulated payload")
        for suffix in ("png", "svg"):
            path = output / f"curves_{context_id}.{suffix}"
            fig.savefig(path, bbox_inches="tight")
            files.append(path.name)
        plt.close(fig)

        final_round = context["rounds"]
        domain_ids = sorted({row["domain_id"] for row in domains if row["context_id"] == context_id}, key=int)
        width = 0.8 / len(relevant)
        fig, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
        for index, group in enumerate(relevant):
            rows = {row["domain_id"]: row for row in domains if row["context_id"] == context_id
                    and row["variant_id"] == group["variant_id"] and row["round"] == final_round}
            values = [rows[domain]["mean"] for domain in domain_ids]
            errors = [rows[domain]["sample_std"] or 0 for domain in domain_ids]
            position = np.arange(len(domain_ids)) - 0.4 + width / 2 + index * width
            axis.bar(position, values, width, yerr=errors, capsize=2,
                     color=colors(index % 10), label=f"{group['label']} (n={group['n_seeds']})")
        axis.set(xticks=np.arange(len(domain_ids)), xticklabels=[f"Domain {domain}" for domain in domain_ids],
                 ylabel="Personalized domain accuracy", ylim=(0, 1),
                 title=f"Final round {final_round}: mean ± sample SD across seeds")
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
        for suffix in ("png", "svg"):
            path = output / f"domains_{context_id}.{suffix}"
            fig.savefig(path, bbox_inches="tight")
            files.append(path.name)
        plt.close(fig)
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(f"report output directory must be empty: {args.output}")
    entries, inputs, contexts = load_inputs(args.inputs)
    groups, group_records, metric_rows, final_rows, rounds, domains = aggregate(entries)
    paired_rows, paired_summary, exclusions = paired_differences(groups)
    trajectory = aggregate_trajectories(rounds, (*ROUND_METRICS, "cumulative_effective_floats", "cumulative_operational_floats"))
    domain_summary = aggregate_trajectories(domains, ("accuracy",), domain=True)
    common = ["context_id", "variant_id", "label", "seed", "directory", "classification"]
    stats = ["n", "mean", "sample_std", "min", "max"]
    pair = ["context_id", "left_variant_id", "right_variant_id", "left_label", "right_label"]
    write_csv(args.output / "seed_metrics.csv", final_rows, common + list(FINAL_METRICS) + ["split_sha256", "initial_state_sha256"])
    write_csv(args.output / "aggregate_metrics.csv", metric_rows, ["context_id", "variant_id", "label", "metric"] + stats)
    write_csv(args.output / "paired_differences.csv", paired_rows, pair + ["seed", "metric", "left_value", "right_value", "difference_right_minus_left"])
    write_csv(args.output / "paired_aggregate.csv", paired_summary, pair + ["metric"] + stats)
    write_csv(args.output / "round_metrics.csv", rounds, common + ["round"] + list(ROUND_METRICS) + ["cumulative_effective_floats", "cumulative_operational_floats"])
    write_csv(args.output / "round_aggregate.csv", trajectory, ["context_id", "variant_id", "label", "round", "metric"] + stats)
    write_csv(args.output / "per_domain.csv", domains, common + ["round", "domain_id", "accuracy"])
    write_csv(args.output / "per_domain_aggregate.csv", domain_summary, ["context_id", "variant_id", "label", "round", "domain_id", "metric"] + stats)
    report = {"schema_version": 1, "status": "aggregated", "generated_unix": time.time(),
        "inputs": inputs, "contexts": contexts, "groups": group_records,
        "paired_difference_definition": "right arm minus left arm, matched seed and identical split/initialization hashes",
        "paired_summary": paired_summary, "pairing_exclusions": exclusions,
        "units": {"accuracy": "fraction in [0,1]", "accuracy_gap": "fraction, lower is better",
                  "floats": "simulated scalar fp32 payload; multiply by four for bytes",
                  "wall_seconds": "measured serial process runtime; not distributed network latency"},
        "interpretation": ["All completed arms and seeds are included without favorable-result filtering.",
            "Comparison contexts differ in training, evaluation, representation or numerical source protocol; do not pool them.",
            "Statistics use sample SD (n-1); n=1 has null SD. No statistical significance or superiority claim is made.",
            "Paired differences are descriptive and retain both positive and negative values; lower cost/gap and higher accuracy have different preferred signs.",
            "Partial runs are inventoried but never represented as completed evidence.",
            "Oracle hierarchy uses supplied domains; results do not establish automatic domain discovery.",
            "Cache extraction time is recorded in the input manifest; seed runtime excludes original feature extraction."],
        "files": []}
    write_json(args.output / "aggregate.json", report)
    if not args.no_plots:
        report["files"].extend(plot_reports(args.output, contexts, group_records, trajectory, domain_summary))
    report["status"] = "complete"
    report["files"] += [path.name for path in sorted(args.output.glob("*.csv"))]
    write_json(args.output / "aggregate.json", report)
    (args.output / "README.md").write_text(
        "# Completion evidence report\n\n"
        "Start with `aggregate.json` for all contexts, arm statistics, provenance and pairing exclusions. "
        "`seed_metrics.csv` retains each final seed result; `paired_differences.csv` records right minus left "
        "for every compatible arm pair. Positive accuracy differences and negative cost/gap differences have "
        "different interpretations. No winners or unfavorable runs were filtered.\n\n"
        "`round_aggregate.csv` and `per_domain_aggregate.csv` supply chart-ready means and sample standard "
        "deviations; their matching raw CSVs retain every seed. PNG/SVG figures show mean ± sample SD. "
        "Separate context IDs indicate different protocols and must not be pooled. Partial runs and "
        "single-seed evidence remain labeled in `aggregate.json`.\n")
    print(f"Wrote {len(entries)} completed runs in {len(contexts)} comparison contexts to {args.output}")
    return report


if __name__ == "__main__":
    main()
