#!/usr/bin/env python3
"""Re-layout the historical P3 heterogeneous figure, preserving its evidence.

Run on gpu003, never retraining or modifying historical artifacts::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
      ~/ahlora-venv/bin/python scripts/plot_historical_paper_readable.py \
      --input docs/artifacts/p3-completion-report \
      --output ~/ahlora-quantity-20260917/historical-figure-qa-p3

All nine arms and four endpoints are retained for context b729ad24b6b0.
Every plotted mean and sample SD is checked against the archived seed records.
Outputs use distinct p3-prefixed metadata names so P1 metadata can coexist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import socket
import statistics


CONTEXT = "b729ad24b6b0"
STEM = "p3_heterogeneous_curves_readable"
PANELS = (
    ("personalized_accuracy", "Personalized client mean", "Accuracy", 1.0),
    ("consensus_accuracy", "Merged consensus model", "Accuracy", 1.0),
    ("cumulative_effective_floats", "Direct contributor payload", "Cumulative fp32 floats\n(billions)", 1e9),
    ("cumulative_operational_floats", "Exact staged payload", "Cumulative fp32 floats\n(billions)", 1e9),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(socket.gethostname().split(".")[0] == "gpu003", "Scientific plotting must run on gpu003")
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    inputs = [args.input / name for name in ("aggregate.json", "round_aggregate.csv", "round_metrics.csv")]
    aggregate = json.loads(inputs[0].read_text())
    context = aggregate["contexts"][CONTEXT]
    require(context["ranks"] == [4, 12, 32] and context["rounds"] == 50, "Wrong historical protocol")
    groups = [g for g in aggregate["groups"] if g["context_id"] == CONTEXT]
    require(len(groups) == 9, "Historical figure must retain all nine arms")
    rows = [r for r in read_csv(inputs[1]) if r["context_id"] == CONTEXT]
    raw = [r for r in read_csv(inputs[2]) if r["context_id"] == CONTEXT]
    raw_index = {}
    for row in raw:
        raw_index.setdefault((row["variant_id"], int(row["round"])), []).append(row)
    curves = []
    checks = 0
    for group in groups:
        for metric, title, ylabel, divisor in PANELS:
            points = sorted((r for r in rows if r["variant_id"] == group["variant_id"] and r["metric"] == metric), key=lambda r: int(r["round"]))
            require([int(r["round"]) for r in points] == list(range(1, 51)), "Incomplete round coverage")
            for point in points:
                seed_rows = raw_index[(group["variant_id"], int(point["round"]))]
                require(sorted(int(r["seed"]) for r in seed_rows) == [42, 43, 44], "Incorrect or duplicate seed coverage")
                require(int(point["n"]) == len(seed_rows) == group["n_seeds"], "Seed count mismatch")
                values = [float(r[metric]) for r in seed_rows]
                require(np.isclose(statistics.mean(values), float(point["mean"]), rtol=1e-12, atol=1e-12), "Archived mean mismatch")
                require(np.isclose(statistics.stdev(values), float(point["sample_std"]), rtol=1e-12, atol=1e-12), "Archived SD mismatch")
                checks += 1
            curves.append({"variant_id": group["variant_id"], "label": group["label"], "metric": metric,
                           "n": group["n_seeds"], "round": [int(r["round"]) for r in points],
                           "mean": [float(r["mean"]) for r in points],
                           "sample_std": [float(r["sample_std"]) for r in points], "display_divisor": divisor})

    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.titlesize": 11, "axes.labelsize": 10.5, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 8.5))
    fig.subplots_adjust(left=.11, right=.98, bottom=.265, top=.885, wspace=.31, hspace=.39)
    colors = plt.get_cmap("tab10")
    for index, group in enumerate(groups):
        for ax, (metric, title, ylabel, divisor) in zip(axes.flat, PANELS):
            curve = next(c for c in curves if c["variant_id"] == group["variant_id"] and c["metric"] == metric)
            mean, sd = np.array(curve["mean"]) / divisor, np.array(curve["sample_std"]) / divisor
            ax.plot(curve["round"], mean, color=colors(index), linewidth=1.4, label=group["label"])
            ax.fill_between(curve["round"], mean - sd, mean + sd, color=colors(index), alpha=.12, linewidth=0)
            ax.set(title=title, xlabel="Communication round", ylabel=ylabel, xlim=(1, 50))
            ax.set_xticks([1, 10, 20, 30, 40, 50])
            ax.grid(alpha=.2)
            if "accuracy" in metric:
                ax.set_ylim(0, 1)
    for ax in axes[1]:
        ax.set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.51, .037), ncol=2,
               fontsize=10, frameon=False, handlelength=2.3, columnspacing=1.7, labelspacing=.65)
    fig.suptitle("Historical heterogeneous-rank protocol", x=.52, y=.977, fontsize=14, fontweight="bold")
    fig.text(.52, .942, "Frozen-feature CIFAR-100 · ranks [4, 12, 32] · 50 rounds", ha="center", fontsize=11)
    fig.text(.52, .916, "Mean ± sample SD across seeds 42, 43, 44; simulated payload", ha="center", fontsize=10.5)
    fig.text(.51, .017, "All nine historical arms retained; no training or endpoint changes.", ha="center", fontsize=9.7)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"{STEM}.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)
    with Image.open(args.output / f"{STEM}.png") as im:
        preview = im.resize((650, round(im.height * 650 / im.width)), Image.Resampling.LANCZOS)
        preview.save(args.output / "p3_actual_width_preview.png")
    write_json(args.output / f"{STEM}.data.json", {"context_id": CONTEXT, "context": context,
               "scope": "Historical P3 only; original uncertainty and all arms preserved", "curves": curves})
    provenance = {"status": "complete", "hostname": socket.gethostname(), "context_id": CONTEXT,
                  "raw_reconstruction_checks": checks, "python": platform.python_version(),
                  "matplotlib": matplotlib.__version__, "numpy": np.__version__,
                  "script": {"path": "scripts/plot_historical_paper_readable.py", "sha256": sha(__file__)},
                  "inputs": [{"path": str(p), "sha256": sha(p)} for p in inputs],
                  "outputs": [{"path": p.name, "sha256": sha(p)} for p in sorted(args.output.glob("p3*")) if p.is_file() and p.suffix != ".md" and p.name != "p3_provenance.json"]}
    write_json(args.output / "p3_provenance.json", provenance)
    (args.output / "p3_README.md").write_text(
        "# Historical P3 figure readability update\n\n"
        "Generated on gpu003 from the frozen P3 completion report, context `b729ad24b6b0`. "
        "All nine arms, all 50 rounds, and all four endpoints are preserved. "
        "The 1,800 plotted mean/sample-SD pairs were independently checked against seeds 42–44. "
        "Accuracy stays on the original 0–1 axis. Payload is divided by 1e9 for the explicitly labeled billions axis; it is simulated fp32 float count, not measured throughput. "
        "The detached legend has its own reserved area below the axes. Original figures are unchanged.\n\n"
        "Reproduce with `scripts/plot_historical_paper_readable.py` on gpu003 using the command in its docstring. "
        "The JSON sidecar contains all exact plotted arrays, and `p3_provenance.json` binds inputs, script, and outputs by SHA-256. "
        "`p3_actual_width_preview.png` renders the figure at 6.5 inches × 100 dpi for paper-width QA.\n")
    print(json.dumps({"output": str(args.output), "curves": len(curves), "mean_sd_pairs_verified": checks}))


if __name__ == "__main__":
    main()
