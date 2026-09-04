"""Publication-oriented figures for Experiment 1 outputs."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT2_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT2_ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_label_heatmap(task_name: str, class_frequency, output_dir: Path):
    _ensure_dir(output_dir)
    matrix = np.asarray(class_frequency, dtype=float)
    fig, ax = plt.subplots(figsize=(max(7, matrix.shape[1] * 0.45), 4.5))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0)
    ax.set_title(f"{task_name}: client label distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Client")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(range(matrix.shape[1]), fontsize=8)
    fig.colorbar(im, ax=ax, label="Class frequency")
    fig.tight_layout()
    fig.savefig(output_dir / f"{task_name}_label_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_client_histograms(task_name: str, class_counts, output_dir: Path):
    _ensure_dir(output_dir)
    counts = np.asarray(class_counts, dtype=float)
    n_clients, n_classes = counts.shape
    fig, axes = plt.subplots(n_clients, 1, figsize=(max(7, n_classes * 0.45), 2.2 * n_clients), sharex=True)
    if n_clients == 1:
        axes = [axes]
    for client_id, ax in enumerate(axes):
        ax.bar(range(n_classes), counts[client_id], color="#377eb8", edgecolor="white", linewidth=0.4)
        ax.set_ylabel(f"C{client_id}")
        ax.grid(axis="y", alpha=0.2, linestyle="--")
    axes[-1].set_xlabel("Class")
    fig.suptitle(f"{task_name}: per-client label histograms", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / f"{task_name}_client_label_histograms.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sample_counts(task_name: str, class_counts, output_dir: Path):
    _ensure_dir(output_dir)
    counts = np.asarray(class_counts, dtype=float).sum(axis=1)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(range(len(counts)), counts, color="#4daf4a", edgecolor="white", linewidth=0.6)
    ax.set_title(f"{task_name}: samples per client")
    ax.set_xlabel("Client")
    ax.set_ylabel("Samples")
    ax.set_xticks(range(len(counts)))
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_dir / f"{task_name}_sample_counts.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_signal_vs_contribution(measurements, output_dir: Path):
    _ensure_dir(output_dir)
    signals = [
        ("js_to_global", "JS label divergence"),
        ("update_cosine_distance_to_mean", "Update cosine distance"),
        ("update_l2_distance_to_mean", "Update L2 distance"),
    ]
    for col, label in signals:
        if col not in measurements:
            continue
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for task, task_df in measurements.groupby("task"):
            ax.scatter(task_df[col], task_df["delta_accuracy"], s=34, alpha=0.75, label=task)
        ax.set_xlabel(label)
        ax.set_ylabel("Leave-one-client-out delta accuracy")
        ax.set_title(f"{label} vs marginal contribution")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output_dir / f"{col}_vs_delta_accuracy.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

