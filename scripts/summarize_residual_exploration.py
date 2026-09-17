#!/usr/bin/env python3
"""Verify and summarize the seed-42 residual screen on a training holdout.

Run on gpu003. Only train.pt is opened: official test.pt is never loaded. Saved
checkpoints are independently evaluated using a direct CPU fp32 forward pass,
without any project model, evaluator, or LoRA reconstruction imports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ARMS = (
    "pooled", "fixed_domain", "adaptive_domain", "residual_uniform_sample_g1",
    "residual_fixed_sample_g1", "residual_adaptive_domain_g1", "residual_adaptive_domain_g5",
    "residual_adaptive_domain_g15", "residual_adaptive_sample_g5",
    "residual_adaptive_quality_g5", "residual_fixed_domain_g5",
)
LABELS = (
    "Pooled LoRA", "Original fixed / domain", "Original adaptive / domain",
    "Residual uniform / sample ×1", "Residual fixed / sample ×1",
    "Residual adaptive / domain ×1", "Residual adaptive / domain ×5",
    "Residual adaptive / domain ×15", "Residual adaptive / sample ×5",
    "Residual adaptive / quality ×5", "Residual fixed / domain ×5",
)
COLORS = ("#222222", "#0072B2", "#CC79A7", "#56B4E9", "#999999", "#009E73",
          "#E69F00", "#D55E00", "#4C78A8", "#8C6BB1", "#8C564B")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_numbers(value):
    if isinstance(value, dict):
        return all(finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_numbers(item) for item in value)
    return math.isfinite(value) if isinstance(value, (int, float)) else True


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        tensor = tensor.detach().cpu().contiguous()
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def verify_tree(transport, graph, root, *, broadcast=False, control=False):
    edges = transport["edges"] if broadcast else transport["tree_edges"]
    require(transport["messages"] == len(edges), "message count differs from ledger")
    require(len(edges) == (28 if control else 14), "unexpected peer collective message count")
    byte_key = "bytes" if broadcast else "payload_bytes"
    require(transport["bytes"] == sum(edge[byte_key] for edge in edges), "payload byte sum differs from ledger")
    for edge in edges:
        require(edge["receiver"] in graph[str(edge["sender"])], "transport uses a non-neighbor edge")
    if control:
        require(transport["root_id"] == root, "control root did not follow rotation")
        require(transport["bytes"] == 8 * (transport["numeric_values"] + transport["source_identifier_values"]),
                "control float64/source-ID byte model mismatch")
        groups = ([edge for edge in edges if edge["phase"] == "gather"],
                  [edge for edge in edges if edge["phase"] == "broadcast"])
        for group, is_broadcast in zip(groups, (False, True)):
            synthetic = {"messages": len(group), "bytes": sum(edge["payload_bytes"] for edge in group),
                         "tree_edges": group}
            if is_broadcast:
                synthetic["edges"] = [{**edge, "bytes": edge["payload_bytes"]} for edge in group]
            verify_tree(synthetic, graph, root, broadcast=is_broadcast)
        return
    if broadcast:
        reached = {root}
        for edge in edges:
            require(edge["sender"] in reached and edge["receiver"] not in reached,
                    "broadcast sender lacked the model or receiver was repeated")
            reached.add(edge["receiver"])
        require(reached == set(range(15)), "broadcast did not reach every peer")
    else:
        parents = {edge["sender"]: edge["receiver"] for edge in edges}
        require(set(parents) == set(range(15)) - {root}, "reduction omitted or repeated a sender")
        sent = set()
        for edge in edges:
            children = {cid for cid, parent in parents.items() if parent == edge["sender"]}
            require(children <= sent, "a parent reduced before receiving its child subtrees")
            sent.add(edge["sender"])
        for cid in parents:
            seen, cursor = set(), cid
            while cursor != root:
                require(cursor not in seen and cursor in parents, "reduction tree is cyclic or disconnected")
                seen.add(cursor)
                cursor = parents[cursor]


def validate_weights(row, counts, *, residual):
    weights = np.asarray(row["weights"], dtype=float)
    require(weights.shape == (15,) and np.isfinite(weights).all() and (weights > 0).all(), "invalid client weights")
    require(np.isclose(weights.sum(), 1, atol=1e-12, rtol=0), "weights are not normalized")
    weight_record = row["weight_record"] if residual else row
    quality = np.asarray(weight_record.get("quality", np.ones(15)), dtype=float)
    factors = np.asarray(weight_record.get("domain_factors", np.ones(15)), dtype=float)
    require((quality > 0).all() and (quality <= 1).all(), "quality score outside its stated loss-based range")
    require((factors >= .85 - 1e-12).all() and (factors <= 1.15 + 1e-12).all(), "domain factor outside trust region")
    require(np.isclose(np.average(factors, weights=counts * quality), 1, atol=1e-12, rtol=0),
            "domain factors do not preserve base-weight scale")
    expected = counts * quality * factors
    expected /= expected.sum()
    require(np.allclose(weights, expected, atol=1e-12, rtol=0), "logged weights differ from sample*quality*domain formula")
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True, help="Exact directory containing train.pt and manifest.json")
    parser.add_argument("--paper-figure", type=Path, required=True, help="Output stem without extension")
    args = parser.parse_args()
    manifest = json.loads((args.input / "manifest.json").read_text())
    require(manifest["status"] == "complete" and manifest["official_test_opened"] is False,
            "completed train-holdout-only manifest required")
    require(tuple(manifest["config"]["arms"]) == ARMS and manifest["config"]["seeds"] == [42]
            and manifest["config"]["rounds"] == 30, "unexpected screening grid")
    holdout = json.loads((args.input / "holdout.json").read_text())
    split_hash = hashlib.sha256(json.dumps({key: value for key, value in holdout.items() if key != "sha256"}, sort_keys=True).encode()).hexdigest()
    require(split_hash == holdout["sha256"] and holdout["official_test_opened"] is False, "holdout provenance hash mismatch")
    fit, validation = holdout["training_original_indices"], holdout["validation_original_indices"]
    require(len(fit) == 45000 and len(validation) == 5000 and sorted(fit + validation) == list(range(50000)),
            "holdout is not a disjoint exhaustive 45000/5000 split")

    import torch
    torch.set_num_threads(2)
    # This is deliberately the only dataset tensor opened by this evaluator.
    training = torch.load(args.feature_cache / "train.pt", map_location="cpu", weights_only=True)
    cache = json.loads((args.feature_cache / "manifest.json").read_text())
    require(tensor_digest([training["features"], training["labels"]]) == cache["train"]["sha256"],
            "original training tensor hash mismatch")
    require(np.array_equal(np.bincount(training["labels"][validation].numpy(), minlength=100), np.full(100, 50)),
            "validation split is not class-stratified at 50 examples per class")
    features = training["features"][validation].float()
    labels = training["labels"][validation]
    records, rows, checks = {}, [], []
    initial_hash = None
    common_split_hash = None
    with torch.inference_mode():
        for arm, label in zip(ARMS, LABELS):
            residual = arm.startswith("residual_")
            path = args.input / arm / f"seed42_{arm}.json"
            record = json.loads(path.read_text())
            records[arm] = record
            require(record["status"] == "complete" and len(record["rounds"]) == 30, f"incomplete arm: {arm}")
            require(finite_numbers(record), f"non-finite recorded values: {arm}")
            require(record["holdout_sha256"] == split_hash and "ORIGINAL TRAINING" in record["evaluation_split"],
                    f"holdout endpoint mismatch: {arm}")
            require(record["n_train"] == 45000 and record.get("n_validation", record.get("n_test")) == 5000,
                    f"wrong sample counts: {arm}")
            require(record["training_sample_exposures"] == 1350000, f"wrong total training exposures: {arm}")
            initial_hash = initial_hash or record["initial_state_sha256"]
            common_split_hash = common_split_hash or record["split_sha256"]
            require(record["initial_state_sha256"] == initial_hash and record["split_sha256"] == common_split_hash,
                    "arms do not share initial parameters and local partitions")
            split_payload = json.loads((args.input / arm / "splits_seed42.json").read_text())
            splits = split_payload.get("clients", split_payload)
            counts = np.asarray([len(splits[str(cid)]["train_indices"]) for cid in range(15)], dtype=float)
            require(sorted(index for item in splits.values() for index in item["train_indices"]) == list(range(45000)),
                    "local ownership does not cover the fit set exactly once")
            require(sorted(index for item in splits.values() for index in item["test_indices"]) == list(range(5000)),
                    "local validation shards do not cover the holdout exactly once")
            reduction_bytes = broadcast_bytes = control_bytes = factor_bytes = 0
            roots = []
            for epoch, row in enumerate(record["rounds"], start=1):
                require(row["round"] == epoch and row.get("training_sample_exposures", row.get("train_sample_exposures")) == 45000,
                        f"round/exposure mismatch: {arm}/{epoch}")
                accuracy = row["validation_accuracy"] if residual else row["full_test_accuracy"]
                correct = row["validation_correct"] if residual else row["full_test_correct"]
                require(correct / 5000 == accuracy, "validation accuracy/count mismatch")
                if arm == "pooled":
                    continue
                graph = record["topology"]
                for cid, rank in row["ranks"].items():
                    ceiling = record["capacity_ceilings"][cid]
                    require(isinstance(rank, int) and 0 < rank <= ceiling, "trainable rank exceeds capability")
                    if "adaptive" in arm:
                        require(rank >= ceiling // 2, "adaptive rank violates half-capability floor")
                        require(row["controller_diagnostics"][cid]["rounds_seen"] == epoch, "controller observation round mismatch")
                        if epoch <= 2:
                            require(rank == ceiling, "adaptive warmup did not retain ceiling")
                weights = validate_weights(row, counts, residual=residual)
                if residual:
                    root = (epoch - 1) % 15
                    roots.append(root)
                    reduction, broadcast = row["residual_reduction"], row["broadcast"]
                    require(reduction["root_id"] == root and reduction["target_rank"] == 100, "reduction root/rank differs from recipe")
                    verify_tree(reduction, graph, root)
                    verify_tree(broadcast, graph, root, broadcast=True)
                    require(reduction["bytes"] == 14 * (51200 * 4 + 8), "dense residual reduction byte model mismatch")
                    require(broadcast["bytes"] == 14 * 51200 * 4, "frozen-head broadcast byte model mismatch")
                    require(np.allclose([reduction["normalized_weights"][str(cid)] for cid in range(15)], weights,
                                        atol=1e-12, rtol=0), "tree reduction weights mismatch")
                    require(0 <= row["global_projection_relative_tail"] <= 1 + 1e-6, "invalid global projection residual")
                    control = row["weight_record"]["control_transport"]
                    if record["weighting"] == "sample":
                        require(control["bytes"] == control["messages"] == 0, "sample-only arm has unexpected weight-control traffic")
                    else:
                        verify_tree(control, graph, root, control=True)
                    reduction_bytes += reduction["bytes"]
                    broadcast_bytes += broadcast["bytes"]
                    control_bytes += control["bytes"]
                else:
                    matrix = np.asarray(row["mixing_matrix"])
                    require(np.allclose(matrix.sum(1), 1, atol=1e-12, rtol=0) and
                            np.allclose(weights @ matrix, weights, atol=1e-12, rtol=0), "baseline weighted mixer invariant failed")
                    expected_floats = 0
                    for receiver in range(15):
                        for sender in range(15):
                            if receiver != sender and matrix[receiver, sender] > 0:
                                require(sender in graph[str(receiver)], "baseline factor transfer crosses a non-neighbor edge")
                                expected_floats += row["ranks"][str(sender)] * (512 + 100)
                    require(expected_floats == row["training_factor_floats"], "baseline factor byte count mismatch")
                    factor_bytes += 4 * expected_floats
                    control_bytes += row["control_bytes"]
                    verify_tree(row["assembly"], graph, row["assembly"]["root_id"])
            if residual:
                require(roots == list(range(15)) * 2, "root rotation did not cover all peers twice")
                total_bytes = reduction_bytes + broadcast_bytes + control_bytes
                require(total_bytes == record["total_transport_bytes"], "residual transport totals mismatch")
                final_accuracy = record["final_validation_accuracy"]
                final_correct = record["rounds"][-1]["validation_correct"]
                assembly_bytes = 0
            else:
                total_bytes = record["total_deployment_payload_bytes"]
                final_accuracy = record["final_full_test_accuracy"]
                final_correct = record["rounds"][-1]["full_test_correct"]
                assembly_bytes = (record.get("final_assembly") or {}).get("bytes", 0)
                require(total_bytes == factor_bytes + control_bytes + assembly_bytes, "baseline deployment total mismatch")
            require(final_correct / 5000 == final_accuracy, "final endpoint differs from last round")
            checkpoint_path = args.input / arm / f"seed42_{arm}_adapter.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            require(checkpoint["seed"] == 42 and checkpoint.get("arm", checkpoint.get("method")) == arm,
                    "checkpoint arm/seed mismatch")
            require(tensor_digest(checkpoint["initial"].values()) == record["initial_state_sha256"], "checkpoint initialization hash mismatch")
            require(set(checkpoint["state"]) == {"fc"}, "independent evaluator expects an fc-only adapter")
            a, b = checkpoint["state"]["fc"]["A"].float(), checkpoint["state"]["fc"]["B"].float()
            require(a.shape[0] == 16 and checkpoint["alpha"] == 32.0 and torch.isfinite(a).all() and torch.isfinite(b).all(),
                    "saved deployment adapter is invalid, non-finite or wrong rank/alpha")
            weight = checkpoint["initial"]["weight"].float() + (32.0 / a.shape[0]) * (b @ a)
            bias = checkpoint["initial"]["bias"].float()
            logits = torch.cat([features[start:start + 500] @ weight.T + bias for start in range(0, 5000, 500)])
            require(torch.isfinite(logits).all(), "independent validation forward pass produced non-finite logits")
            predictions = logits.argmax(1)
            verified_correct = int((predictions == labels).sum())
            require(verified_correct == final_correct, f"independent fp32 validation count mismatch: {arm}")
            validation_loss = float((torch.logsumexp(logits.double(), dim=1) - logits.double()[torch.arange(5000), labels]).mean())
            losses = [row["train_loss"] for row in record["rounds"]]
            rows.append({"arm": arm, "label": label, "seed": 42, "status": record["status"],
                         "endpoint": "5000 held-out original-training examples", "validation_accuracy_percent": 100 * final_accuracy,
                         "validation_correct": final_correct, "independent_correct": verified_correct,
                         "independent_validation_cross_entropy": validation_loss,
                         "independent_max_absolute_logit": float(logits.abs().max()),
                         "training_sample_exposures": record["training_sample_exposures"],
                         "first_train_loss": losses[0], "final_train_loss": losses[-1], "maximum_train_loss": max(losses),
                         "final_to_first_train_loss_ratio": losses[-1] / losses[0],
                         "all_metrics_and_checkpoint_finite": True,
                         "training_factor_bytes": factor_bytes, "dense_reduction_bytes": reduction_bytes,
                         "dense_broadcast_bytes": broadcast_bytes, "control_bytes": control_bytes,
                         "one_final_assembly_bytes": assembly_bytes, "total_protocol_bytes": total_bytes,
                         "max_projection_tail": max(row.get("global_projection_relative_tail", 0) for row in record["rounds"]),
                         "frozen_head_bytes_per_peer": record.get("retained_global_head_bytes_per_peer", 0),
                         "checkpoint_sha256": sha256(checkpoint_path), "record_sha256": sha256(path),
                         "prediction_sha256": hashlib.sha256(predictions.numpy().tobytes()).hexdigest()})
            checks.append({"arm": arm, "status": "passed", "rounds_checked": 30,
                           "rank_caps_checked": arm != "pooled", "weights_checked": arm != "pooled",
                           "root_rotation_checked": residual, "neighbor_edge_cost_checked": arm != "pooled",
                           "independent_fp32_count_match": True})
    summary = {"status": "complete", "schema_version": 1, "seed": 42, "n_arms": 11,
               "exploratory": True, "n_seeds": 1, "uncertainty": "single screening seed; no between-seed error bars or formal inference",
               "evaluation_split": "5000 held-out ORIGINAL TRAINING examples; 45000 examples used for training",
               "official_test_opened_by_verifier": False, "official_test_opened_by_run_as_declared": False,
               "holdout_sha256": split_hash, "original_training_tensor_sha256": cache["train"]["sha256"],
               "script_sha256": sha256(__file__), "source_manifest_sha256": sha256(args.input / "manifest.json"),
               "independent_evaluation": "gpu003 CPU float32 direct X(W0+(alpha/r)BA)^T+b, 500-example batches; no project forward/evaluator imports",
               "checks": checks, "results": rows,
               "interpretation": "All eight retained-base residual recipes failed this screening comparison. The zero-residual preservation invariant does not guarantee that nonzero local learning updates will be useful.",
               "method_change_scope": "Multiple changes: retained dense frozen head, fresh local residual directions, rotating exact tree reduction/broadcast, class centering, rank16 global projection and declared gain. Not a one-variable repair or the original one-hop MH protocol.",
               "capacity_scope": "Small ranks constrain trainable residuals only; every root performs rank100 reduction factorization and rank16 global projection, and all peers hold the frozen global head. Local function rank can exceed the residual rank.",
               "privacy_scope": "No privacy guarantee; domain-control allgather exposes individual dense changes and metadata."}
    (args.input / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (args.input / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(summary, records, args.input / "figures", args.paper_figure)
    (args.input / "SUMMARY.md").write_text(report(summary))
    print(json.dumps({"status": summary["status"], "verified_checkpoints": len(rows), "official_test_opened": False,
                      "results": [{key: row[key] for key in ("arm", "validation_accuracy_percent", "maximum_train_loss",
                                                             "final_train_loss", "independent_validation_cross_entropy")} for row in rows]}, indent=2))


def plot(summary, records, output, paper):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    output.mkdir(parents=True, exist_ok=True)
    paper.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(7.5, 7.8), gridspec_kw={"height_ratios": [1.3, 1]})
    positions = np.arange(len(ARMS))
    values = [row["validation_accuracy_percent"] for row in summary["results"]]
    axes[0].barh(positions, values, color=COLORS, height=.65)
    axes[0].set_yticks(positions, LABELS)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 64)
    for position, value in zip(positions, values):
        axes[0].text(value + .6, position, f"{value:.2f}", va="center", fontsize=8)
    axes[0].set_xlabel("Validation accuracy (%); 5000 original-training holdout examples")
    axes[0].set_title("Residual screen: all eleven arms, seed42; official test unopened")
    for arm in ("pooled", "adaptive_domain", "residual_adaptive_domain_g1", "residual_adaptive_domain_g5",
                "residual_adaptive_domain_g15", "residual_adaptive_sample_g5"):
        index = ARMS.index(arm)
        axes[1].semilogy(range(1, 31), [row["train_loss"] for row in records[arm]["rounds"]],
                         color=COLORS[index], label=LABELS[index], lw=1.6)
    axes[1].set_xlabel("Training epoch; 45000 fit examples per epoch")
    axes[1].set_ylabel("Local training cross entropy (log scale)")
    axes[1].legend(fontsize=6.8, frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.02))
    axes[1].grid(alpha=.15)
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(output / f"exploration_residual.{extension}", bbox_inches="tight", dpi=230)
        figure.savefig(paper.with_suffix("." + extension), bbox_inches="tight", dpi=230)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for arm, label, color in zip(ARMS[1:], LABELS[1:], COLORS[1:]):
        metrics = [100 * row.get("validation_accuracy", row.get("full_test_accuracy")) for row in records[arm]["rounds"]]
        axes[0].plot(range(1, 31), metrics, label=label, color=color, lw=1.5,
                     ls="--" if arm.startswith("residual") else "-")
    axes[0].set_xlabel("Training epoch")
    axes[0].set_ylabel("Training-holdout accuracy (%)")
    axes[0].set_title("All peer recipes; pooled reference shown in main figure")
    axes[0].legend(fontsize=6, frameon=False, ncol=2)
    left = np.zeros(11)
    for field, label, color in (("training_factor_bytes", "Original factor exchange", "#0072B2"),
                                ("dense_reduction_bytes", "Dense residual reduction", "#009E73"),
                                ("dense_broadcast_bytes", "Dense head broadcast", "#56B4E9"),
                                ("control_bytes", "Weight-control exchange", "#E69F00"),
                                ("one_final_assembly_bytes", "Original final assembly", "#CC79A7")):
        widths = np.asarray([row[field] for row in summary["results"]]) / 2**20
        axes[1].barh(positions, widths, left=left, color=color, label=label)
        left += widths
    axes[1].set_yticks(positions, LABELS, fontsize=6)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Modeled protocol payload (MiB)")
    axes[1].legend(fontsize=6, frameon=False, loc="lower right")
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(output / f"all_residual_curves_and_costs.{extension}", bbox_inches="tight", dpi=230)
    plt.close(figure)


def report(summary):
    lines = ["# Residual protocol screen: all candidates failed", "", summary["evaluation_split"] + ". **These are validation results, not official-test results.**",
             "", "All 11 saved models were independently reevaluated on gpu003 with a direct CPU fp32 forward pass; every correct count matched. Every arm completed 30 epochs with exactly 1,350,000 training sample exposures. Screening used seed42 only; no between-seed SD is available.", "",
             "| Arm | Validation accuracy (%) | Final local training loss | Maximum local training loss | Protocol payload (MiB) |",
             "|---|---:|---:|---:|---:|"]
    for row in summary["results"]:
        lines.append(f"| {row['label']} | {row['validation_accuracy_percent']:.2f} | {row['final_train_loss']:.3f} | {row['maximum_train_loss']:.3f} | {row['total_protocol_bytes']/2**20:.3f} |")
    gain15 = next(row for row in summary["results"] if row["arm"] == "residual_adaptive_domain_g15")
    lines += ["", summary["interpretation"], "",
              f"The gain15 recipe remained numerically finite but was unstable: local training cross entropy increased from {gain15['first_train_loss']:.3f} to {gain15['final_train_loss']:.3f}, with a maximum of {gain15['maximum_train_loss']:.3f}. Independent final validation cross entropy was {gain15['independent_validation_cross_entropy']:.3f}. This is severe finite loss growth, not a NaN crash or a proof of asymptotic divergence.", "",
              "## Independent checks", "",
              "The verifier opened only the original `train.pt`, checked its SHA-256, reconstructed the 45,000/5,000 holdout from `holdout.json`, verified exhaustive disjoint indices and 50 validation examples per class, and bound every arm to the same holdout/initialization/partition. The reused baseline `full_test_*` field names refer exclusively to this 5,000-example training holdout.", "",
              "Every peer round was checked for normalized positive sample/quality/domain weights and trainable-rank ceilings. Adaptive floors and warmup were checked. Residual roots rotate 0–14 twice; all reduction, broadcast and control edges belong to the declared graph. Ledger message counts, per-edge byte sums, full protocol totals, reduction weights, and global projection residuals were verified. Every saved rank16 model and its logits remained finite, including gain15.", "",
              "## Changed-method and resource scope", "", summary["method_change_scope"], "", summary["capacity_scope"], "",
              "Each residual round transports 2,867,312 bytes of dense residual reduction plus 2,867,200 bytes of frozen-head broadcast. Quality/domain control traffic is added separately. Domain allgather dominates the reported payload. Original-model-state controls count their training factors, control traffic, and one final assembly; evaluation-only intermediate assemblies remain outside deployment cost, as in the original driver. Packet framing, graph setup and runtime transport overhead remain excluded.", "",
              "Each peer stores a 204,800-byte frozen head; folding the retained update into that existing weight avoids an extra persistent delta buffer, but aggregation and SVD workspace remain additional. A small trainable residual rank is not a guarantee that the full local function, frozen state, or root's aggregation fits the original rank-only capacity interpretation.", "",
              "The screen does not establish that retained-base residual methods cannot work. It rejects these eight disclosed recipes at this budget and seed. All failures, the high-gain instability, raw per-round diagnostics and checkpoints remain preserved. " + summary["privacy_scope"], ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
