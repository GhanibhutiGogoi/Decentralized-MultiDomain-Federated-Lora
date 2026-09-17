"""Independent saved-checkpoint and traffic audit; execute on gpu003 only."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import torch
import torch.nn.functional as F


def digest(*tensors):
    result = hashlib.sha256()
    for value in tensors:
        value = value.detach().cpu().contiguous()
        result.update(str((tuple(value.shape), str(value.dtype))).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    cached = torch.load(args.cache_dir / "test.pt", map_location="cpu", weights_only=True)
    cache_manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    assert digest(cached["features"], cached["labels"]) == cache_manifest["test"]["sha256"]
    output = {"status": "completed", "execution_host": __import__("socket").gethostname(),
              "scope": "independent forward evaluation and recorded traffic/source invariants", "seeds": []}
    dependency_path = args.run_dir / "dependency_source_manifest.json"
    dependencies = None
    if dependency_path.exists():
        dependencies = json.loads(dependency_path.read_text())
        archive = args.run_dir / "dependency_source_snapshot.tar.gz"
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == dependencies["archive_sha256"]
        with tarfile.open(archive, "r:gz") as handle:
            for name, expected in dependencies["files"].items():
                assert hashlib.sha256(handle.extractfile(name).read()).hexdigest() == expected
        output["source_files_verified"] = len(dependencies["files"])
    for seed in args.seeds:
        record = json.loads((args.run_dir / f"seed{seed}.json").read_text())
        assert record["status"] == "completed"
        if dependencies:
            assert record["source_sha256"] == dependencies["files"]["project-3-hierarchical-gossip/experiments/diagnose_gradient_equivalence.py"]
        checkpoint_path = args.run_dir / f"seed{seed}_checkpoint.pt"
        assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == record["checkpoint_sha256"]
        state = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
        assert state["feature_cache_identity_sha256"] == cache_manifest["cache_identity_sha256"]
        assert state["seed"] == seed and state["rank"] == record["rank"] == 16
        assert state["alpha"] == record["alpha"] == 32
        results = {}
        for arm in ("pooled", "distributed"):
            model = state[arm]
            a, b = model["lora_A"], model["lora_B"]
            assert a.shape[0] == b.shape[1] == 16
            predictions = []
            with torch.no_grad():
                for begin in range(0, len(cached["labels"]), 1024):
                    x = cached["features"][begin:begin + 1024].to(device=args.device, dtype=a.dtype)
                    logits = F.linear(x, model["linear.weight"], model["linear.bias"])
                    logits = logits + ((x @ a.T) @ b.T) * (state["alpha"] / a.shape[0])
                    predictions.append(logits.argmax(1).cpu())
            prediction = torch.cat(predictions)
            correct = int((prediction == cached["labels"]).sum())
            assert correct == record["rounds"][-1][f"{arm}_full_test_correct"]
            results[arm] = {"full_test_correct": correct, "accuracy": correct / len(prediction)}
            results[arm]["prediction_sha256"] = digest(prediction)
        assert len(record["proof"]) == record["proof_tolerance"]["checked_batches"]
        assert all(r["max_gradient_abs_error"] <= 2e-11 for r in record["proof"])
        n_peers = len(record["topology"])
        steps = sum(r["optimizer_steps"] for r in record["rounds"])
        n_parameters = state["distributed"]["lora_A"].numel() + state["distributed"]["lora_B"].numel()
        bytes_per_message = n_parameters * state["distributed"]["lora_A"].element_size() + 8
        assert record["total_peer_messages"] == steps * 2 * (n_peers - 1)
        assert record["total_peer_payload_bytes"] == record["total_peer_messages"] * bytes_per_message
        for sender, receiver in record["reduction_edges"] + record["broadcast_edges"]:
            assert receiver in record["topology"][str(sender)]
        output["seeds"].append({"seed": seed, "arms": results, "steps_per_arm": steps,
                                "peer_messages": record["total_peer_messages"],
                                "peer_payload_bytes": record["total_peer_payload_bytes"],
                                "fp64_gradient_checks": len(record["proof"])})
    (args.run_dir / "independent_verification.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
