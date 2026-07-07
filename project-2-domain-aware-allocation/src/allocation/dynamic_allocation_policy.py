"""
Dynamic allocation policy for Project 2.

This module connects Gabriel's adaptive rank selection output with
the alpha/scaling allocation policy.

Pipeline:
    client_i -> rank r_i -> alpha alpha_i -> scaling alpha_i / r_i
"""


def compute_scaling(alpha, rank):
    """
    Compute the effective LoRA scaling factor.

    LoRA uses scaling = alpha / rank.
    """
    if rank <= 0:
        raise ValueError("Rank must be positive.")

    return alpha / rank


def dynamic_allocation(rank_assignments, alpha=64):
    """
    Build full dynamic allocation from client-specific ranks.

    Args:
        rank_assignments:
            Dictionary mapping client_id to selected LoRA rank.
            Example:
                {
                    "client_0": 4,
                    "client_1": 8,
                    "client_2": 16
                }

        alpha:
            Global alpha value used for the first version of the policy.
            Default is 64 based on Experiment 06 alpha-policy analysis.

    Returns:
        Dictionary mapping each client to rank, alpha, and scaling.
    """
    allocations = {}

    for client_id, rank in rank_assignments.items():
        allocations[client_id] = {
            "rank": rank,
            "alpha": alpha,
            "scaling": compute_scaling(alpha, rank),
        }

    return allocations


def dynamic_allocation_from_records(client_records, alpha=64):
    """
    Build dynamic allocation from richer client records.

    Args:
        client_records:
            List of dictionaries. Each dictionary should contain:
                - client_id
                - selected_rank

            Example:
                [
                    {"client_id": "client_0", "selected_rank": 4},
                    {"client_id": "client_1", "selected_rank": 8},
                    {"client_id": "client_2", "selected_rank": 16}
                ]

        alpha:
            Global alpha value.

    Returns:
        Dictionary mapping each client to rank, alpha, and scaling.
    """
    rank_assignments = {
        record["client_id"]: record["selected_rank"]
        for record in client_records
    }

    return dynamic_allocation(rank_assignments, alpha=alpha)
