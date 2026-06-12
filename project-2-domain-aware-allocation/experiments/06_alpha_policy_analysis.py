import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.allocation.alpha_policy import AlphaPolicyAnalyzer


INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "experiment_05_alpha_search"
    / "best_alpha_per_client.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "experiment_06_alpha_policy_analysis"
)

OUTPUT_FILE = OUTPUT_DIR / "alpha_policy_report.json"

def load_results():
    with open(INPUT_FILE, "r") as f:
        return json.load(f)


def compute_alpha_distribution(data):
    distribution = defaultdict(int)

    for client in data.values():
        distribution[client["best_alpha"]] += 1

    return dict(sorted(distribution.items()))



    return dict(sorted(result.items()))


def compute_tolerance_policy(data, tolerance=0.005):
    """
    Smallest alpha within tolerance of oracle accuracy.
    """

    result = {}

    for client_id, client in data.items():

        best_acc = client["best_accuracy"]

        candidates = []

        for alpha, acc in client["all_alpha_results"].items():

            alpha = int(alpha)

            if acc >= best_acc - tolerance:
                candidates.append((alpha, acc))

        chosen_alpha, chosen_acc = min(
            candidates,
            key=lambda x: x[0]
        )

        result[client_id] = {
            "fixed_rank": client["fixed_rank"],
            "oracle_best_alpha": client["best_alpha"],
            "oracle_best_accuracy": best_acc,
            "policy_alpha": chosen_alpha,
            "policy_accuracy": chosen_acc,
            "accuracy_drop": best_acc - chosen_acc
        }

    return result


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_results()

    analyzer = AlphaPolicyAnalyzer(INPUT_FILE)

    report = {
        "best_global_alpha":
            analyzer.best_global_alpha(),

        "best_alpha_distribution":
            compute_alpha_distribution(data),

        "best_alpha_per_rank":
            analyzer.best_alpha_per_rank(),

        "smallest_alpha_within_0_005_tolerance":
            compute_tolerance_policy(
                data,
                tolerance=0.005
            )
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("EXPERIMENT 06 COMPLETE")
    print("=" * 60)

    print("\nBest Global Alpha:")
    print(report["best_global_alpha"])

    print("\nAlpha Distribution:")
    print(report["best_alpha_distribution"])

    print("\nBest Alpha Per Rank:")
    print(report["best_alpha_per_rank"])

    print(f"\nSaved to: {OUTPUT_FILE}")




if __name__ == "__main__":
    main()
