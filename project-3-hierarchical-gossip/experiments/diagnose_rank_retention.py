"""No-training stress diagnostic for repeated heterogeneous-rank projection.

Starts from oracle pooled rank-16 checkpoints. This is NOT a federated training
success experiment: it asks whether mixing/projection alone destroys an already
learned common adapter. No optimizer, backward pass or training data is used.
All evaluations use the official full CIFAR-100 test feature cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from experiments.feature_cache import cache_identity, tensor_digest, write_json
from experiments.integrated_benchmark import WeightedRunner, score_state
from experiments.protocol_benchmark import package_versions, source_provenance
from src.federated.domain_weights import weighted_metropolis
from src.federated.merge import factorize_delta, lora_to_delta
from src.federated.mixing import metropolis_hastings
from src.federated.peer_assembly import tree_weighted_assembly


METHODS = {
    "ring_uniform16": ("ring", (16, 16, 16)),
    "ring_capability4_8_16": ("ring", (4, 8, 16)),
    "ring_reduced2_4_8": ("ring", (2, 4, 8)),
    "central_uniform16": ("central", (16, 16, 16)),
    "central_capability4_8_16": ("central", (4, 8, 16)),
    "central_reduced2_4_8": ("central", (2, 4, 8)),
}
COLORS = ("#222222", "#0072B2", "#D55E00", "#888888", "#56B4E9", "#E69F00")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def progress(output, event, **payload):
    with (output / "progress.jsonl").open("a") as handle:
        handle.write(json.dumps({"event": event, "time_unix": time.time(), **payload}, allow_nan=False) + "\n")


def runner_for(matrix, weights, ranks, alpha):
    clients = [SimpleNamespace(client_id=i, domain_id=0) for i in range(len(ranks))]
    runner = WeightedRunner(clients, lambda _: matrix, dict(enumerate(ranks)), alpha)
    runner.set_weights(matrix, weights)
    return runner


def projected_states(delta, ranks, alpha):
    return [{"fc": factorize_delta(delta, rank, alpha)} for rank in ranks]


def common_basis_coefficients(delta, u, vh, components=16):
    return torch.diag(u[:, :components].T @ delta @ vh[:components].T)


def synthetic_formula_check():
    """Independent diagonal oracle for central attenuation and ring recurrence."""
    alpha, n, components = 32.0, 6, 16
    singular = torch.arange(components, 0, -1, dtype=torch.float64)
    delta = torch.diag(singular)
    ranks = [2, 4, 8, 2, 4, 8]
    weights = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    weights /= weights.sum()
    neighbors = {i: [(i - 1) % n, (i + 1) % n] for i in range(n)}
    proposal = metropolis_hastings(neighbors, list(range(n)))
    retain = np.arange(1, components + 1)[None, :] <= np.asarray(ranks)[:, None]
    mass = weights @ retain
    rows = []
    for topology in ("central", "ring"):
        matrix = np.tile(weights, (n, 1)) if topology == "central" else weighted_metropolis(proposal, weights)
        runner = runner_for(matrix, weights, ranks, alpha)
        states = projected_states(delta, ranks, alpha)
        coefficients = retain * singular.numpy()[None, :]
        for cycle in range(6):
            if cycle:
                states, _ = runner.gossip_round(cycle - 1, states)
                coefficients = retain * (matrix @ coefficients)
            observed = sum(float(w) * lora_to_delta(state, alpha)["fc"] for w, state in zip(weights, states))
            expected = torch.diag(torch.from_numpy(weights @ coefficients))
            error = float(torch.linalg.norm(observed - expected))
            if error > 1e-10:
                raise AssertionError(f"synthetic {topology} common-basis recurrence failed: {error}")
            if topology == "central":
                closed_form = singular.numpy() * mass ** (cycle + 1)
                if not np.allclose(weights @ coefficients, closed_form, atol=1e-12, rtol=1e-12):
                    raise AssertionError("central coefficient attenuation disagrees with p_k^(t+1)")
            rows.append({"topology": topology, "cycle": cycle,
                         "observed_coefficients": torch.diag(observed).tolist(),
                         "predicted_coefficients": (weights @ coefficients).tolist(),
                         "absolute_frobenius_error": error})
    return {"status": "passed", "dtype": "float64", "ranks": ranks, "weights": weights.tolist(),
            "oracle_singular_values": singular.tolist(), "retaining_weight_mass": mass.tolist(),
            "formula": "p_k = sum_i pi_i 1[r_i >= k]; initial aggregate coefficient sigma_k p_k; central cycle t coefficient sigma_k p_k^(t+1)",
            "ring_formula": "c_k(0)=sigma_k d_k; c_k(t+1)=diag(d_k) P c_k(t); aggregate coefficient=pi^T c_k(t)",
            "limiting_observation": "On a connected fixed-rank network, every component absent at any peer is killed by repeated mixing/projection; components retained by every peer survive. The shared SVD basis and no-training conditions are essential.",
            "rows": rows}


def evaluate_cycle(states, oracle, u, singular, vh, weights, coefficients, initial,
                   test, indices, graph, alpha, cycle, merge_diagnostics):
    n = len(states)
    evaluation_root = max(range(n), key=lambda i: states[i]["fc"]["A"].shape[0])
    assembled, assembly = tree_weighted_assembly(dict(enumerate(states)), dict(enumerate(weights)),
                                                graph, evaluation_root, 16, alpha)
    aggregate = lora_to_delta(assembled, alpha)["fc"]
    direct = sum(float(w) * lora_to_delta(state, alpha)["fc"] for w, state in zip(weights, states))
    target_coefficients = weights @ coefficients
    observed_coefficients = common_basis_coefficients(aggregate, u, vh).numpy()
    error = np.linalg.norm(observed_coefficients - target_coefficients) / float(torch.linalg.norm(singular[:16]))
    if error > 2e-4:
        raise AssertionError(f"common-basis coefficient recurrence diverged at cycle {cycle}: {error}")
    oracle_energy = float(torch.sum(oracle ** 2))
    aggregate_energy = float(torch.sum(aggregate ** 2))
    evaluation = score_state(assembled, initial, test, indices, alpha, 2048)
    client_norms = [float(torch.linalg.norm(lora_to_delta(state, alpha)["fc"])) for state in states]
    return {"cycle": cycle, "full_test_accuracy": evaluation["accuracy"],
            "full_test_correct": evaluation["correct"], "n_test": evaluation["n_test"],
            "frobenius_norm": aggregate_energy ** .5,
            "relative_frobenius_norm": (aggregate_energy / oracle_energy) ** .5,
            "retained_energy_fraction": aggregate_energy / oracle_energy,
            "relative_distance_to_oracle": float(torch.linalg.norm(aggregate - oracle)) / oracle_energy ** .5,
            "assembled_singular_values": torch.linalg.svdvals(aggregate).tolist(),
            "common_basis_coefficients": observed_coefficients.tolist(),
            "predicted_common_basis_coefficients": target_coefficients.tolist(),
            "coefficient_retention_ratios": (observed_coefficients / singular[:16].numpy()).tolist(),
            "common_basis_relative_prediction_error": float(error),
            "per_client_frobenius_norms": client_norms,
            "assembly_relative_truncation_energy": assembly["relative_truncation_energy"],
            "assembly_vs_direct_relative_error": float(torch.linalg.norm(aggregate - direct)) / oracle_energy ** .5,
            "mean_projection_residual_energy": merge_diagnostics.get("mean_residual_energy", 0.0),
            "mean_projection_tail_mass": merge_diagnostics.get("mean_tail_mass", 0.0),
            "max_projection_tail_mass": merge_diagnostics.get("max_tail_mass", 0.0),
            "merge_diagnostics": merge_diagnostics,
            "assembly": assembly,
            "n_peers": n}


def run_seed(args, seed, test, cache_metadata):
    checkpoint_path = args.checkpoints / f"seed{seed}_pooled_adapter.pt"
    pooled_record_path = args.checkpoints / f"seed{seed}_pooled.json"
    peer_record_path = args.checkpoints / f"seed{seed}_mh_sample.json"
    split_path = args.checkpoints / f"splits_seed{seed}.json"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pooled_record = json.loads(pooled_record_path.read_text())
    peer_record = json.loads(peer_record_path.read_text())
    splits = json.loads(split_path.read_text())["clients"]
    if pooled_record["status"] != "complete" or peer_record["status"] != "complete":
        raise ValueError("oracle and topology source experiments must be complete")
    if pooled_record["feature_cache_identity_sha256"] != cache_metadata["cache_identity_sha256"]:
        raise ValueError("feature cache identity differs from saved pooled model")
    alpha = float(checkpoint["alpha"])
    if alpha != pooled_record["alpha"] or alpha != peer_record["alpha"] or checkpoint["method"] != "pooled":
        raise ValueError("pooled checkpoint provenance mismatch")
    if set(checkpoint["state"]) != {"fc"} or checkpoint["state"]["fc"]["A"].shape[0] != 16:
        raise ValueError("diagnostic requires the pooled rank-16 fc adapter")
    oracle = lora_to_delta(checkpoint["state"], alpha)["fc"]
    u, singular, vh = torch.linalg.svd(oracle, full_matrices=False)
    graph = {int(cid): [int(peer) for peer in neighbors] for cid, neighbors in peer_record["topology"].items()}
    ids = sorted(graph)
    if ids != list(range(15)):
        raise ValueError("diagnostic expects exactly 15 peers with contiguous IDs")
    weights = np.asarray([len(splits[str(cid)]["train_indices"]) for cid in ids], dtype=float)
    weights /= weights.sum()
    proposal = metropolis_hastings(graph, ids)
    indices = torch.arange(len(test["labels"]), device=args.device)
    baseline = score_state(checkpoint["state"], checkpoint["initial"], test, indices, alpha, 2048)
    if baseline["correct"] != pooled_record["rounds"][-1]["full_test_correct"]:
        raise ValueError("saved pooled checkpoint does not reproduce its recorded full-test endpoint")
    direct_projections = {}
    for rank in (2, 4, 8, 16):
        projected = {"fc": factorize_delta(oracle, rank, alpha)}
        direct_projections[str(rank)] = score_state(projected, checkpoint["initial"], test, indices, alpha, 2048)
    seed_provenance = {"checkpoint_sha256": sha256(checkpoint_path),
                       "pooled_record_sha256": sha256(pooled_record_path),
                       "peer_topology_record_sha256": sha256(peer_record_path),
                       "split_file_sha256": sha256(split_path),
                       "feature_cache_identity_sha256": cache_metadata["cache_identity_sha256"]}
    completed = []
    for method, (topology, menu) in METHODS.items():
        ranks = [menu[cid % 3] for cid in ids]
        matrix = np.tile(weights, (len(ids), 1)) if topology == "central" else weighted_metropolis(proposal, weights)
        runner = runner_for(matrix, weights, ranks, alpha)
        states = projected_states(oracle, ranks, alpha)
        retain = np.arange(1, 17)[None, :] <= np.asarray(ranks)[:, None]
        coefficients = retain * singular[:16].numpy()[None, :]
        initial_residuals = [float(torch.sum((oracle - lora_to_delta(state, alpha)["fc"]) ** 2)) for state in states]
        record = {"schema_version": 1, "status": "running", "seed": seed, "method": method,
                  "scope": "Oracle pooled initialization; no optimizer or training; information-retention stress diagnostic, not federated training success",
                  "alpha": alpha, "ranks": ranks, "weights": weights.tolist(), "mixing_matrix": matrix.tolist(),
                  "topology": graph, "mixing_scope": "neighbor weighted-MH" if topology == "ring" else "centralized weighted model-state averaging control",
                  "oracle_full_test_accuracy": baseline["accuracy"], "oracle_full_test_correct": baseline["correct"],
                  "oracle_frobenius_norm": float(torch.linalg.norm(oracle)), "oracle_singular_values": singular.tolist(),
                  "oracle_direct_rank_projections": direct_projections,
                  "common_basis_retaining_weight_mass": (weights @ retain).tolist(),
                  "initial_mean_projection_residual_energy": float(np.mean(initial_residuals)),
                  "expected_common_basis_limiting_rank": min(ranks), "provenance": seed_provenance,
                  "evaluation_assembly_scope": "rank16 reconstruction measures retained information; the all-ranks-at-most8 arm does not claim a resource-feasible rank16 deployment peer",
                  "training_steps": 0, "training_sample_exposures": 0,
                  "cycles": []}
        path = args.output / f"seed{seed}_{method}.json"
        write_json(path, record)
        tick = time.perf_counter()
        for cycle in range(args.cycles + 1):
            diagnostics = {}
            if cycle:
                states, diagnostics = runner.gossip_round(cycle - 1, states)
                coefficients = retain * (matrix @ coefficients)
            row = evaluate_cycle(states, oracle, u, singular, vh, weights, coefficients,
                                 checkpoint["initial"], test, indices, graph, alpha, cycle, diagnostics)
            record["cycles"].append(row)
            write_json(path, record)
            progress(args.output, "cycle_completed", seed=seed, method=method, cycle=cycle,
                     full_test_accuracy=row["full_test_accuracy"], retained_energy_fraction=row["retained_energy_fraction"])
            if cycle % 5 == 0 or cycle == args.cycles:
                print(f"seed={seed} method={method} cycle={cycle} accuracy={row['full_test_accuracy']:.4f} retained_energy={row['retained_energy_fraction']:.4f}", flush=True)
        record["status"] = "complete"
        record["wall_seconds"] = time.perf_counter() - tick
        write_json(path, record)
        completed.append(record)
    return completed


def summarize_and_plot(records, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    per_seed = []
    for record in records:
        initial, final = record["cycles"][0], record["cycles"][-1]
        per_seed.append({"seed": record["seed"], "method": record["method"],
                         "oracle_accuracy_percent": 100 * record["oracle_full_test_accuracy"],
                         "initial_projected_accuracy_percent": 100 * initial["full_test_accuracy"],
                         "final_accuracy_percent": 100 * final["full_test_accuracy"],
                         "additional_cycle_accuracy_change_pp": 100 * (final["full_test_accuracy"] - initial["full_test_accuracy"]),
                         "gap_to_oracle_pp": 100 * (final["full_test_accuracy"] - record["oracle_full_test_accuracy"]),
                         "initial_retained_energy_percent": 100 * initial["retained_energy_fraction"],
                         "final_retained_energy_percent": 100 * final["retained_energy_fraction"],
                         "final_relative_frobenius_norm": final["relative_frobenius_norm"],
                         "max_common_basis_prediction_error": max(row["common_basis_relative_prediction_error"] for row in record["cycles"]),
                         "final_mean_projection_residual_energy": final["mean_projection_residual_energy"]})
    aggregate = []
    for method in METHODS:
        rows = [row for row in per_seed if row["method"] == method]
        values = {key: {"mean": float(np.mean([row[key] for row in rows])),
                        "sample_sd": float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else None}
                  for key in rows[0] if key not in {"seed", "method"}}
        aggregate.append({"method": method, "n_seeds": len(rows), **values})
    summary = {"status": "complete", "scope": "no-training oracle information-retention diagnostic", "n_records": len(records),
               "cycle_count": len(records[0]["cycles"]) - 1, "per_seed": per_seed, "aggregate": aggregate,
               "uncertainty": "sample SD across seeds; no equivalence claim", "training_steps": 0,
               "interpretation_boundary": "Information lost from repeated projection/mixing can explain a failure mechanism; these oracle-initialized results are not achievable federated training results."}
    write_json(output / "summary.json", summary)
    with (output / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for method, color in zip(METHODS, COLORS):
        runs = [record for record in records if record["method"] == method]
        cycles = [row["cycle"] for row in runs[0]["cycles"]]
        for ax, metric, factor in zip(axes, ("full_test_accuracy", "retained_energy_fraction", "relative_frobenius_norm"), (100, 100, 1)):
            matrix = np.array([[factor * row[metric] for row in record["cycles"]] for record in runs])
            mean = matrix.mean(axis=0)
            sd = matrix.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
            ax.plot(cycles, mean, color=color, ls="--" if method.startswith("central") else "-", label=method, lw=1.6)
            ax.fill_between(cycles, mean - sd, mean + sd, color=color, alpha=.10)
            ax.set_xlabel("Mix/project cycles; no training")
    for ax, label in zip(axes, ("Full-test accuracy (%)", "Adapter energy retained (%)", "Adapter Frobenius norm / oracle norm")):
        ax.set_ylabel(label)
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output / f"rank_retention_curves.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True, constrained_layout=True)
    for ax, method in zip(axes.flat, METHODS):
        runs = [record for record in records if record["method"] == method]
        matrix = np.mean([[row["coefficient_retention_ratios"] for row in record["cycles"]] for record in runs], axis=0).T
        plot = ax.imshow(matrix, origin="upper", aspect="auto", vmin=0, vmax=1, cmap="viridis",
                         extent=(-.5, matrix.shape[1] - .5, 16.5, .5))
        ax.set_title(method)
        ax.set_yticks([1, 4, 8, 12, 16])
        ax.set_xlabel("Mix/project cycles")
        ax.set_ylabel("Original singular component")
    fig.colorbar(plot, ax=axes, label="Coefficient / pooled oracle coefficient", shrink=.8)
    for ext in ("pdf", "png"):
        fig.savefig(output / f"rank_retention_coefficients.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)
    def number(row, field):
        item = row[field]
        return f"{item['mean']:.3f} ± {item['sample_sd']:.3f}" if item["sample_sd"] is not None else f"{item['mean']:.3f}"
    lines = ["# No-training rank-retention diagnostic", "",
             "This starts from oracle pooled rank-16 checkpoints. It tests whether communication and rank truncation preserve an already learned model, with zero optimizer steps and no training data. It cannot establish federated training success.", "",
             f"| Method | After initial projection (%) | After {summary['cycle_count']} cycles (%) | Extra accuracy change (pp) | Final energy retained (%) |",
             "|---|---:|---:|---:|---:|"]
    for row in aggregate:
        lines.append(f"| {row['method']} | {number(row, 'initial_projected_accuracy_percent')} | {number(row, 'final_accuracy_percent')} | {number(row, 'additional_cycle_accuracy_change_pp')} | {number(row, 'final_retained_energy_percent')} |")
    lines += ["", "Values are means ± sample SD across the paired seeds. Cycle zero already includes distributing rank projections and assembling them; subsequent cycles contain mixing and repeated projection only.", "",
              "For a common SVD basis, let $d_{ik}=1[r_i\geq k]$ and $p_k=\sum_i\pi_i d_{ik}$. The initial weighted assembly's kth coefficient is $\sigma_k p_k$. Under repeated centralized mixing and receiver truncation, it becomes $\sigma_k p_k^{t+1}$ after t cycles. For a ring, $c_k(0)=\sigma_k d_k$ and $c_k(t+1)=\mathrm{diag}(d_k)P c_k(t)$; assembly reads $\pi^T c_k(t)$. A synthetic double-precision check validates both formulas against the actual projection/merge kernels.", "",
              "The mechanism has a specific scope: fixed ranks, aligned singular bases, no new local learning, and model-state averaging followed by receiver truncation. On a connected graph, repeatedly deleting a component at some peers can eventually remove it across the network. The experiment does not prove that this is the only cause of the full training accuracy gap, and does not test a proposed repair.", "",
              "Full raw spectra, original-basis coefficients and independent recurrence predictions, Frobenius norms, projection residuals, assembly diagnostics, test counts and input/source hashes are preserved. Tests and plotting were executed on gpu003. No privacy result is implied.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True,
                        help="Exact cache directory containing manifest.json and test.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--cycles", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.cycles < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("positive cycles and unique seeds are required")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("use a fresh output directory to preserve every attempted diagnostic")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    manifest = {"schema_version": 1, "status": "running", "host": "gpu003",
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "source": source_provenance(), "script_sha256": sha256(__file__), "packages": package_versions(),
                "scope": "oracle initialized, zero-training information-retention diagnostic; no federated success or privacy claim",
                "records": []}
    write_json(args.output / "manifest.json", manifest)
    progress(args.output, "diagnostic_started", seeds=args.seeds, cycles=args.cycles)
    try:
        synthetic = synthetic_formula_check()
        write_json(args.output / "synthetic_formula_check.json", synthetic)
        metadata = json.loads((args.feature_cache / "manifest.json").read_text())
        if metadata["cache_identity_sha256"] != cache_identity(metadata):
            raise ValueError("cache metadata identity mismatch")
        test_cpu = torch.load(args.feature_cache / "test.pt", map_location="cpu", weights_only=True)
        if tensor_digest(test_cpu["features"], test_cpu["labels"]) != metadata["test"]["sha256"]:
            raise ValueError("full test feature cache digest mismatch")
        if len(test_cpu["labels"]) != 10000:
            raise ValueError("full official CIFAR-100 test set required")
        test = {name: tensor.to(args.device) for name, tensor in test_cpu.items()}
        manifest["test_feature_cache"] = metadata
        write_json(args.output / "manifest.json", manifest)
        records = []
        with torch.inference_mode():
            for seed in args.seeds:
                records.extend(run_seed(args, seed, test, metadata))
                manifest["records"] = [{"seed": record["seed"], "method": record["method"], "status": record["status"]} for record in records]
                write_json(args.output / "manifest.json", manifest)
        summarize_and_plot(records, args.output)
        manifest["status"] = "complete"
        write_json(args.output / "manifest.json", manifest)
        progress(args.output, "diagnostic_completed", n_records=len(records))
    except Exception as error:
        manifest.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        write_json(args.output / "manifest.json", manifest)
        progress(args.output, "diagnostic_failed", error=manifest["error"])
        raise


if __name__ == "__main__":
    main()
