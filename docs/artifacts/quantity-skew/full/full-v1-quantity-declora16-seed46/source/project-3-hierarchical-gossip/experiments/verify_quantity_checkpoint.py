#!/usr/bin/env python3
"""Audit quantity-skew checkpoints with an independent data/metric path.

This script does not import the training runner, its data reader, its evaluation
function, its partition validator, or its training/aggregation code. It reuses
the model's explicit LoRA architecture/install helper and Hugging Face model
implementation; that shared dependency is recorded rather than hidden.

Run on gpu003: python experiments/verify_quantity_checkpoint.py \
    --run /path/to/run --assets /path/to/assets
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.models.roberta_lora import inject_query_value, install_state


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def array_digest(value):
    return hashlib.sha256(np.asarray(value, dtype="<i8").tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(path.read_text())


def integer_indices(values, upper, name):
    require(isinstance(values, list) and all(type(value) is int for value in values),
            f"{name}: expected integer index list")
    array = np.asarray(values, dtype=np.int64)
    require(array.ndim == 1 and len(array) > 0, f"{name}: empty or non-vector index list")
    require(bool(np.all((array >= 0) & (array < upper))), f"{name}: index out of range")
    require(len(np.unique(array)) == len(array), f"{name}: duplicate indices")
    return array


def verify_artifacts(run, assets, config, summary, environment, split):
    import pyarrow.parquet as parquet

    source = read_json(run / "source_manifest.json")
    for relative, expected in source.items():
        path = (run / "source" / relative).resolve()
        require(path.is_relative_to((run / "source").resolve()), "unsafe source manifest path")
        require(path.is_file() and file_sha256(path) == expected, f"archived source hash mismatch: {relative}")
    source_digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    require(source_digest == summary["source_sha256"] == environment["source_sha256"],
            "source manifest aggregate hash mismatch")
    for filename, expected in environment["model_files"].items():
        require(Path(filename).name == filename, "unsafe model file name")
        require(file_sha256(assets / "roberta-base" / filename) == expected,
                f"model asset hash mismatch: {filename}")

    tables = {}
    source_indices = {}
    for name in ("train", "validation"):
        path = assets / "glue" / config["task"] / f"{name}-00000-of-00001.parquet"
        record = split["data_files"][name]
        require(file_sha256(path) == record["file_sha256"], f"{name} parquet hash mismatch")
        table = parquet.read_table(path).to_pydict()
        labels = np.asarray(table["label"], dtype=np.int64)
        require(len(labels) == record["rows_in_file"], f"{name} raw row count mismatch")
        require(set(np.unique(labels)).issubset({0, 1}), f"{name}: expected labeled binary data")
        indices = integer_indices(record["source_indices"], len(labels), name)
        require(len(indices) == record["rows_used"], f"{name} used row count mismatch")
        require(array_digest(indices) == record["source_indices_sha256"], f"{name} source index hash mismatch")
        if name == "validation" or not config["max_train"]:
            require(np.array_equal(indices, np.arange(len(labels))), f"{name}: full canonical split not used")
        else:
            expected = np.random.default_rng(config["seed"] + 901).permutation(len(labels))[:config["max_train"]]
            require(np.array_equal(indices, expected), "smoke subset does not match declared selection")
        tables[name], source_indices[name] = table, indices

    labels = np.asarray(tables["train"]["label"])[source_indices["train"]]
    classes, class_counts = np.unique(labels, return_counts=True)
    require(split["n_examples"] == len(labels), "partition example count mismatch")
    require(split["class_labels"] == classes.tolist(), "partition class labels mismatch")
    require(split["global_class_counts"] == class_counts.tolist(), "partition global class counts mismatch")
    require(split["label_sha256"] == json_digest(labels.tolist()), "partition label hash mismatch")
    require(set(split["clients"]) == {str(cid) for cid in range(split["n_clients"])}, "client ID set mismatch")
    owners = np.zeros(len(labels), dtype=np.int64)
    memberships, ordered, observed_classes, client_counts = {}, {}, {}, []
    for cid in range(split["n_clients"]):
        key, record = str(cid), split["clients"][str(cid)]
        indices = integer_indices(record["indices"], len(labels), f"client {cid}")
        owners[indices] += 1
        histogram = [int(np.sum(labels[indices] == label)) for label in classes]
        require(len(indices) == record["n_samples"], f"client {cid}: sample count mismatch")
        require(histogram == record["class_counts"], f"client {cid}: class histogram mismatch")
        require(math.isclose(record["sample_weight"], len(indices) / len(labels), abs_tol=1e-15),
                f"client {cid}: sample weight mismatch")
        # Quantity-only stratification must preserve each proportional class
        # quota up to the unavoidable integer floor/ceiling, independently of
        # the partition implementation that created the manifest.
        products = len(indices) * class_counts
        lower = products // len(labels)
        upper = lower + (products % len(labels) != 0)
        require(bool(np.all(np.asarray(histogram) >= lower) and np.all(np.asarray(histogram) <= upper)),
                f"client {cid}: label proportions violate quantity-only allocation")
        memberships[key], ordered[key] = sorted(indices.tolist()), indices.tolist()
        require(record["membership_sha256"] == json_digest(memberships[key]), f"client {cid}: membership hash mismatch")
        observed_classes[key] = int(np.count_nonzero(histogram))
        require(observed_classes[key] == record["classes_observed"], f"client {cid}: observed class count mismatch")
        client_counts.append(len(indices))
    require(bool(np.all(owners == 1)), "training shards are not disjoint and exhaustive")
    require(client_counts == split["client_counts"], "client count vector mismatch")
    require(json_digest(memberships) == split["partition_sha256"], "partition membership hash mismatch")
    require(json_digest(ordered) == split["index_order_sha256"], "partition order hash mismatch")
    require(len(np.unique(source_indices["train"][np.concatenate([np.asarray(ordered[str(cid)])
                for cid in range(split["n_clients"])])])) == len(labels),
            "mapped original training indices overlap")
    return tables["validation"], {
        "archived_source_files_verified": len(source), "source_sha256": source_digest,
        "model_asset_files_verified": len(environment["model_files"]),
        "parquet_files_verified": 2, "partition_examples": len(labels),
        "partition_sha256": split["partition_sha256"], "shards_disjoint_exhaustive": True,
        "quantity_only_class_quotas_verified": True, "classes_observed_per_client": observed_classes,
        "training_source_indices_sha256": array_digest(source_indices["train"]),
        "validation_rows": len(tables["validation"]["label"]),
        "split_identity_scope": "Separate canonical train/validation parquet assets; original source index mappings checked."
    }


def metrics(predictions, labels):
    predictions, labels = np.asarray(predictions, dtype=np.int64), np.asarray(labels, dtype=np.int64)
    require(predictions.shape == labels.shape, "prediction count differs from labels")
    require(set(np.unique(predictions)).issubset({0, 1}), "nonbinary prediction")
    # Compute from a confusion matrix, independently of the runner's masks.
    confusion = np.bincount(2 * labels + predictions, minlength=4).reshape(2, 2)
    tn, fp, fn, tp = map(int, confusion.ravel())
    denominator = 2 * tp + fp + fn
    return {"n": len(labels), "correct": tn + tp,
            "accuracy": 100 * (tn + tp) / len(labels),
            "f1": 100 * (2 * tp / denominator) if denominator else 0.0,
            "confusion_matrix": confusion.tolist(), "predictions_sha256": array_digest(predictions)}


def verify_metric_record(observed, expected, label):
    for name in ("n", "correct", "predictions_sha256"):
        require(observed[name] == expected[name], f"{label}: {name} mismatch")
    for name in ("accuracy", "f1"):
        require(math.isclose(observed[name], expected[name], rel_tol=0, abs_tol=1e-10),
                f"{label}: {name} mismatch")


@torch.no_grad()
def checkpoint_predictions(model, encoded, batch_size, device):
    model.eval()
    results = []
    total = len(encoded["input_ids"])
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        # Independently reproduce right-padding removal, to preserve exact
        # floating-point batch shapes when checking near-tied logits.
        tokens = {name: value[start:stop] for name, value in encoded.items()}
        used_columns = int(torch.count_nonzero(tokens["attention_mask"], dim=1).max())
        tokens = {name: value[:, :used_columns].to(device) for name, value in tokens.items()}
        logits = model(**tokens).logits
        require(bool(torch.isfinite(logits).all()), "checkpoint emits nonfinite logits")
        results.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    return np.asarray(results, dtype=np.int64)


def audit(run, assets):
    require(socket.gethostname().split(".")[0] == "gpu003", "Scientific execution is restricted to gpu003")
    config, summary = read_json(run / "config.json"), read_json(run / "summary.json")
    environment, split = read_json(run / "environment.json"), read_json(run / "split.json")
    require(summary["status"] == "complete", "run is not complete")
    require(config["task"] in ("sst2", "mrpc"), "unsupported task")
    validation, artifact_checks = verify_artifacts(run, assets, config, summary, environment, split)
    labels = np.asarray(validation["label"], dtype=np.int64)
    rounds = [json.loads(line) for line in (run / "rounds.jsonl").read_text().splitlines() if line.strip()]
    require(len(rounds) == config["rounds"] == summary["rounds"], "round history length mismatch")
    require([row["round"] for row in rounds] == list(range(1, len(rounds) + 1)), "rounds missing or reordered")
    for row in rounds:
        verify_metric_record(metrics(row["validation"]["predictions"], labels), row["validation"], f"round {row['round']}")
    primary = "f1" if config["task"] == "mrpc" else "accuracy"
    require(summary["primary_metric"] == primary, "incorrect primary metric for task")
    best = max(rounds, key=lambda row: row["validation"][primary])
    require(best["round"] == summary["best"]["round"], "best checkpoint selection differs from first maximum")
    verify_metric_record(metrics(rounds[-1]["validation"]["predictions"], labels), summary["final"], "summary final")
    verify_metric_record(metrics(best["validation"]["predictions"], labels), summary["best"]["validation"], "summary best")
    require(summary["total_steps"] == sum(peer["steps"] for row in rounds for peer in row["local"].values()),
            "total optimizer step count mismatch")
    require(summary["total_examples"] == sum(peer["examples"] for row in rounds for peer in row["local"].values()),
            "total example exposure mismatch")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    from transformers import AutoTokenizer, RobertaForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(assets / "roberta-base", local_files_only=True)
    text_args = (validation["sentence"],) if config["task"] == "sst2" else (validation["sentence1"], validation["sentence2"])
    encoded = tokenizer(*text_args, padding="max_length", truncation=True,
                        max_length=config["max_length"], return_tensors="pt")
    model = RobertaForSequenceClassification.from_pretrained(assets / "roberta-base", num_labels=2,
                local_files_only=True, attn_implementation="eager")
    model = inject_query_value(model, rank=16, alpha=config["alpha"], dropout=0).to("cuda")
    outputs = {}
    for name, reference in (("best", best), ("final", rounds[-1])):
        path = run / f"{name}.pt"
        require(file_sha256(path) == summary[f"{name}_checkpoint_sha256"], f"{name} checkpoint file hash mismatch")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        require(checkpoint["config"] == config, f"{name} checkpoint configuration mismatch")
        require(checkpoint["source_sha256"] == artifact_checks["source_sha256"], f"{name} checkpoint source hash mismatch")
        require(checkpoint["round"] == reference["round"], f"{name} checkpoint round mismatch")
        install_state(model, checkpoint["state"])
        prediction = checkpoint_predictions(model, encoded, config["batch_size"], torch.device("cuda"))
        require(np.array_equal(prediction, reference["validation"]["predictions"]), f"{name} raw checkpoint predictions differ")
        result = metrics(prediction, labels)
        verify_metric_record(result, reference["validation"], f"{name} checkpoint")
        outputs[name] = {"round": reference["round"], "checkpoint_sha256": file_sha256(path), "metrics": result,
                         "exact_raw_prediction_match": True}
    helper = PROJECT / "src/models/roberta_lora.py"
    return {"status": "passed", "host": socket.gethostname(), "run": str(run),
            "evaluator_sha256": file_sha256(Path(__file__)), "model_helper_sha256": file_sha256(helper),
            "independence_scope": "Independent parquet reader, split/hash validator, log checks and confusion-matrix metrics; shared Hugging Face architecture and roberta_lora install helper; no training runner imports.",
            "artifact_checks": artifact_checks, "round_metric_records_verified": len(rounds),
            "checkpoints": outputs, "precision": "fp32 deterministic CUDA, original evaluation batch shapes",
            "evaluation_scope": "Canonical labeled GLUE validation; not hidden test evaluation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run.resolve(), args.assets.resolve())
    destination = args.run / "independent_checkpoint_audit.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    print(json.dumps({"status": result["status"], "run": str(args.run),
                      "best": result["checkpoints"]["best"]["metrics"],
                      "final": result["checkpoints"]["final"]["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
