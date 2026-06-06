"""Experiment 1: capability-aware adaptive LoRA rank.

The adaptive rank is not brute force. Each client uses the closed-form rule

    s(G) = ||G||_F^2 / ||G||_2^2
    c_i  = capability_index(batch_i) / (num_capabilities - 1)
    r_i  = nearest_allowed_rank(max(r_min, c_i R_i^max, min(s(G), R_i^max)))

where s(G) is the median stable rank of LoRA adapter gradients.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    ADAPT_COL,
    BATCH_TO_MAX_RANK,
    CLIENT_BATCH_SIZES,
    CLIENT_COLORS,
    CLIENT_EPOCHS,
    FIXED_RANK,
    HOMO_COL,
    NUM_CLIENTS,
    NUM_ROUNDS,
)
from Federated.client import compute_quality_score, train_client  # noqa: E402
from Federated.fedavg_aggregation import fedavg_quality_weighted  # noqa: E402
from Federated.flops import compute_round_flops  # noqa: E402
from Federated.utilities import evaluate, split_dataset  # noqa: E402
from rank_allocation.LoRa_rank_projection import load_global_state  # noqa: E402
from rank_allocation.rank_selector import estimate_optimal_rank  # noqa: E402
from Source.Models import AudioCNN, CNN, LSTMModel, MLP, TabularMLP  # noqa: E402
from Source.datasets.audio import get_audio  # noqa: E402
from Source.datasets.image import get_cifar10, get_fashion_mnist  # noqa: E402
from Source.datasets.tabular import get_tabular  # noqa: E402
from Source.datasets.text import AGNewsDataset, get_agnews  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_DIR = PROJECT_ROOT / "result" / "exp1"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RANK_EQUATION = (
    "s(G)=||G||_F^2/||G||_2^2; "
    "c_i=capability_index(batch_i)/(num_capabilities-1); "
    "r_i=nearest_allowed_rank(max(r_min, c_i*R_i^max, min(s(G), R_i^max)))"
)


def make_loaders(trainset):
    clients = split_dataset(trainset, NUM_CLIENTS)
    return [
        DataLoader(client_data, batch_size=CLIENT_BATCH_SIZES[i], shuffle=True)
        for i, client_data in enumerate(clients)
    ]


def run_fixed_rank(name, model_fn, trainset, testloader, rank=FIXED_RANK):
    print(f"\n  [FIXED] {name} r={rank}")
    loss_fn = nn.CrossEntropyLoss()
    loaders = make_loaders(trainset)
    global_state = model_fn(rank).to(DEVICE).state_dict()

    acc_curve, flops_hist = [], []

    for rnd in range(NUM_ROUNDS):
        weights, samples, quality_scores, round_flops = [], [], [], []

        for i, loader in enumerate(loaders):
            local = model_fn(rank).to(DEVICE)
            load_global_state(local, global_state)
            round_flops.append(compute_round_flops(local, loader, rank, CLIENT_EPOCHS))

            state, sample_count = train_client(local, loader, CLIENT_EPOCHS, DEVICE)
            quality = compute_quality_score(local, loader, loss_fn, DEVICE)
            weights.append(state)
            samples.append(sample_count)
            quality_scores.append(quality)

        flops_hist.append(round_flops)
        ref_sd = model_fn(rank).to(DEVICE).state_dict()
        global_state = fedavg_quality_weighted(
            weights, samples, quality_scores, rank, ref_sd, DEVICE
        )

        eval_model = model_fn(rank).to(DEVICE)
        load_global_state(eval_model, global_state)
        acc = evaluate(eval_model, testloader, DEVICE)
        acc_curve.append(acc)

        print(
            f"    Round {rnd + 1}/{NUM_ROUNDS} | acc={acc:.2f}% | "
            f"total_flops={sum(round_flops):.2e}"
        )

    return acc_curve, acc_curve[-1], flops_hist


def run_adaptive_rank(name, model_fn, trainset, testloader):
    print(f"\n  [ADAPTIVE] {name}")
    loss_fn = nn.CrossEntropyLoss()
    loaders = make_loaders(trainset)
    global_state = model_fn(FIXED_RANK).to(DEVICE).state_dict()

    acc_curve, flops_hist, rank_hist = [], [], []

    for rnd in range(NUM_ROUNDS):
        weights, samples, quality_scores = [], [], []
        ranks, round_flops = [], []

        for i, loader in enumerate(loaders):
            batch_size = CLIENT_BATCH_SIZES[i]
            probe_rank = BATCH_TO_MAX_RANK[batch_size]
            probe = model_fn(probe_rank).to(DEVICE)
            load_global_state(probe, global_state)

            chosen_rank = estimate_optimal_rank(probe, loader, loss_fn, batch_size)
            ranks.append(chosen_rank)

            local = model_fn(chosen_rank).to(DEVICE)
            load_global_state(local, global_state)
            round_flops.append(
                compute_round_flops(local, loader, chosen_rank, CLIENT_EPOCHS)
            )

            state, sample_count = train_client(local, loader, CLIENT_EPOCHS, DEVICE)
            quality = compute_quality_score(local, loader, loss_fn, DEVICE)
            weights.append(state)
            samples.append(sample_count)
            quality_scores.append(quality)

        flops_hist.append(round_flops)
        rank_hist.append(ranks)

        ref_sd = model_fn(FIXED_RANK).to(DEVICE).state_dict()
        global_state = fedavg_quality_weighted(
            weights, samples, quality_scores, FIXED_RANK, ref_sd, DEVICE
        )

        eval_model = model_fn(FIXED_RANK).to(DEVICE)
        load_global_state(eval_model, global_state)
        acc = evaluate(eval_model, testloader, DEVICE)
        acc_curve.append(acc)

        detail = " | ".join(
            f"C{i}(r={ranks[i]},flops={round_flops[i]:.2e},q={quality_scores[i]:.3f})"
            for i in range(NUM_CLIENTS)
        )
        print(
            f"    Round {rnd + 1}/{NUM_ROUNDS} | acc={acc:.2f}% | "
            f"total_flops={sum(round_flops):.2e} | {detail}"
        )

    return acc_curve, acc_curve[-1], flops_hist, rank_hist


def load_experiments():
    print("\n=== Loading Datasets ===")
    cifar_train, cifar_test, cifar_testloader = get_cifar10()
    fashion_train, fashion_test, fashion_testloader = get_fashion_mnist()
    agnews_train, agnews_test, agnews_testloader = get_agnews()
    tabular_train, tabular_test, tabular_testloader = get_tabular()
    audio_train, audio_test, audio_testloader = get_audio()

    print(f"  CIFAR-10  train={len(cifar_train)}, test={len(cifar_test)}")
    print(f"  Fashion   train={len(fashion_train)}, test={len(fashion_test)}")
    print(f"  AG News   train={len(agnews_train)}, test={len(agnews_test)}")
    print(f"  Tabular   train={len(tabular_train)}, test={len(tabular_test)}")
    print(f"  Audio     train={len(audio_train)}, test={len(audio_test)}")

    vocab = AGNewsDataset.VOCAB_SIZE
    return [
        ("CIFAR-CNN", lambda r: CNN(3, 10, r), cifar_train, cifar_testloader),
        ("Fashion-MLP", lambda r: MLP(28 * 28, 10, r), fashion_train, fashion_testloader),
        ("AGNews-LSTM", lambda r: LSTMModel(vocab, 64, 128, 2, 4, r), agnews_train, agnews_testloader),
        ("Tabular-MLP", lambda r: TabularMLP(tabular_train.in_dim, tabular_train.num_classes, r), tabular_train, tabular_testloader),
        ("Audio-1DCNN", lambda r: AudioCNN(1, audio_train.NUM_CLASSES, r), audio_train, audio_testloader),
    ]


def plot_accuracy(results, names):
    x_rounds = np.arange(1, NUM_ROUNDS + 1)
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4))
    if len(names) == 1:
        axes = [axes]
    fig.suptitle("Fixed-rank vs equation-based adaptive LoRA", fontweight="bold")

    for ax, name in zip(axes, names):
        fixed_acc = results[name]["fixed"][0]
        adaptive_acc = results[name]["adaptive"][0]
        ax.plot(x_rounds, fixed_acc, color=HOMO_COL, marker="s", label=f"Fixed r={FIXED_RANK}")
        ax.plot(x_rounds, adaptive_acc, color=ADAPT_COL, marker="o", label="Adaptive equation")
        ax.set_title(name)
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULT_DIR / "fig1_accuracy_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ranks(results, names):
    x_rounds = np.arange(1, NUM_ROUNDS + 1)
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4))
    if len(names) == 1:
        axes = [axes]
    fig.suptitle("Adaptive rank per client from closed-form equation", fontweight="bold")

    for ax, name in zip(axes, names):
        rank_hist = results[name]["adaptive"][3]
        for client_idx in range(NUM_CLIENTS):
            vals = [rank_hist[rnd][client_idx] for rnd in range(NUM_ROUNDS)]
            batch_size = CLIENT_BATCH_SIZES[client_idx]
            ax.plot(
                x_rounds,
                vals,
                color=CLIENT_COLORS[client_idx],
                marker="o",
                label=f"C{client_idx} bs={batch_size}",
            )
            ax.axhline(
                BATCH_TO_MAX_RANK[batch_size],
                color=CLIENT_COLORS[client_idx],
                linestyle=":",
                alpha=0.45,
            )
        ax.axhline(FIXED_RANK, color=HOMO_COL, linestyle="--", alpha=0.75)
        ax.set_title(name)
        ax.set_xlabel("Round")
        ax.set_ylabel("LoRA rank")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(RESULT_DIR / "fig2_adaptive_rank_per_client.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_final_accuracy(results, names):
    x_exp = np.arange(len(names))
    width = 0.35
    fixed = [results[name]["fixed"][1] for name in names]
    adaptive = [results[name]["adaptive"][1] for name in names]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(
        x_exp - width / 2,
        fixed,
        width,
        color=HOMO_COL,
        alpha=0.85,
        label=f"Fixed r={FIXED_RANK}",
        edgecolor="white",
    )
    ax.bar(
        x_exp + width / 2,
        adaptive,
        width,
        color=ADAPT_COL,
        alpha=0.85,
        label="Adaptive equation",
        edgecolor="white",
    )

    for idx, (fixed_acc, adaptive_acc) in enumerate(zip(fixed, adaptive)):
        delta = adaptive_acc - fixed_acc
        ax.text(
            idx,
            max(fixed_acc, adaptive_acc) + 0.8,
            f"{delta:+.1f}%",
            ha="center",
            fontsize=9,
            color=ADAPT_COL if delta >= 0 else HOMO_COL,
            fontweight="bold",
        )

    ax.set_title(f"Final accuracy after {NUM_ROUNDS} rounds")
    ax.set_xticks(x_exp)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Final accuracy (%)")
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "fig3_final_accuracy_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_total_flops(results, names):
    x_rounds = np.arange(1, NUM_ROUNDS + 1)
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4))
    if len(names) == 1:
        axes = [axes]
    fig.suptitle("Total FLOPs per round", fontweight="bold")

    for ax, name in zip(axes, names):
        fixed = [sum(results[name]["fixed"][2][rnd]) for rnd in range(NUM_ROUNDS)]
        adaptive = [sum(results[name]["adaptive"][2][rnd]) for rnd in range(NUM_ROUNDS)]
        ax.plot(x_rounds, fixed, color=HOMO_COL, marker="s", label=f"Fixed r={FIXED_RANK}")
        ax.plot(x_rounds, adaptive, color=ADAPT_COL, marker="o", label="Adaptive equation")
        ax.fill_between(
            x_rounds,
            adaptive,
            fixed,
            where=[f >= a for f, a in zip(fixed, adaptive)],
            alpha=0.15,
            color=ADAPT_COL,
            label="FLOPs saved",
        )
        ax.set_title(name)
        ax.set_xlabel("Round")
        ax.set_ylabel("Total FLOPs")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(RESULT_DIR / "fig4_total_flops_per_round.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_client_flops(results, names):
    x_rounds = np.arange(1, NUM_ROUNDS + 1)
    fig, axes = plt.subplots(2, len(names), figsize=(4.5 * len(names), 8))
    if len(names) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    fig.suptitle("FLOPs contribution per client", fontweight="bold")

    for exp_idx, name in enumerate(names):
        for row_idx, mode in enumerate(["fixed", "adaptive"]):
            ax = axes[row_idx][exp_idx]
            bottoms = np.zeros(NUM_ROUNDS)
            flops_hist = results[name][mode][2]

            for client_idx in range(NUM_CLIENTS):
                vals = np.array(
                    [flops_hist[rnd][client_idx] for rnd in range(NUM_ROUNDS)],
                    dtype=float,
                )
                ax.bar(
                    x_rounds,
                    vals,
                    bottom=bottoms,
                    color=CLIENT_COLORS[client_idx],
                    alpha=0.88,
                    label=f"C{client_idx} bs={CLIENT_BATCH_SIZES[client_idx]}",
                    width=0.6,
                    edgecolor="white",
                    linewidth=0.5,
                )
                bottoms += vals

            title = "Fixed" if mode == "fixed" else "Adaptive"
            ax.set_title(f"{name} - {title}")
            ax.set_xlabel("Round")
            ax.set_ylabel("FLOPs")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.grid(axis="y", alpha=0.2, linestyle="--")
            ax.legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(RESULT_DIR / "fig5_flops_per_client.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(results, names):
    fig, ax = plt.subplots(figsize=(9, 6))
    for name in names:
        fixed_flops = sum(sum(row) for row in results[name]["fixed"][2])
        adaptive_flops = sum(sum(row) for row in results[name]["adaptive"][2])
        fixed_acc = results[name]["fixed"][1]
        adaptive_acc = results[name]["adaptive"][1]

        ax.scatter(fixed_flops, fixed_acc, color=HOMO_COL, marker="s", s=100)
        ax.scatter(adaptive_flops, adaptive_acc, color=ADAPT_COL, marker="o", s=100)
        ax.annotate("", xy=(adaptive_flops, adaptive_acc), xytext=(fixed_flops, fixed_acc),
                    arrowprops=dict(arrowstyle="->", color="gray"))
        ax.text((fixed_flops + adaptive_flops) / 2, (fixed_acc + adaptive_acc) / 2, name, fontsize=8)

    ax.set_title("Accuracy vs total FLOPs")
    ax.set_xlabel("Total FLOPs")
    ax.set_ylabel("Final accuracy (%)")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "fig6_pareto_accuracy_flops.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary(results, names):
    rows = []
    for name in names:
        fixed_acc = results[name]["fixed"][1]
        adaptive_acc = results[name]["adaptive"][1]
        fixed_flops = sum(sum(row) for row in results[name]["fixed"][2])
        adaptive_flops = sum(sum(row) for row in results[name]["adaptive"][2])
        rank_hist = results[name]["adaptive"][3]
        all_ranks = [r for row in rank_hist for r in row]
        saved = fixed_flops - adaptive_flops
        rows.append(
            {
                "Experiment": name,
                "Rank Equation": RANK_EQUATION,
                "Fixed Rank": FIXED_RANK,
                "Fixed Final Acc (%)": round(fixed_acc, 2),
                "Adaptive Final Acc (%)": round(adaptive_acc, 2),
                "Accuracy Delta (%)": round(adaptive_acc - fixed_acc, 2),
                "Fixed Total FLOPs": int(fixed_flops),
                "Adaptive Total FLOPs": int(adaptive_flops),
                "FLOPs Saved (%)": round(100.0 * saved / fixed_flops, 2) if fixed_flops else 0.0,
                "Average Adaptive Rank": round(float(np.mean(all_ranks)), 2),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "federated_lora_summary.csv", index=False)
    (RESULT_DIR / "rank_equation.txt").write_text(RANK_EQUATION + "\n", encoding="utf-8")
    print("\n" + df.to_string(index=False))


def main():
    os.chdir(PROJECT_ROOT)
    print(f"Using device: {DEVICE}")
    print(f"Rank equation: {RANK_EQUATION}")
    print(f"Client capabilities: {BATCH_TO_MAX_RANK}")

    experiments = load_experiments()
    names = [item[0] for item in experiments]
    results = {}

    print("\n=== Running Experiments ===")
    for name, model_fn, trainset, testloader in experiments:
        print(f"\n{'=' * 60}\n  EXPERIMENT: {name}\n{'=' * 60}")
        results[name] = {
            "fixed": run_fixed_rank(name, model_fn, trainset, testloader),
            "adaptive": run_adaptive_rank(name, model_fn, trainset, testloader),
        }

    plot_accuracy(results, names)
    plot_ranks(results, names)
    plot_final_accuracy(results, names)
    plot_total_flops(results, names)
    plot_client_flops(results, names)
    plot_pareto(results, names)
    write_summary(results, names)
    print(f"\nSaved outputs to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
