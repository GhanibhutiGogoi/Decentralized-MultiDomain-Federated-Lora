"""IID client partitioning."""


def iid_partition_indices(dataset, num_clients: int) -> list[list[int]]:
    """Project 1-compatible contiguous split."""
    n = len(dataset)
    size = n // num_clients
    return [
        list(range(client_id * size, (client_id + 1) * size))
        for client_id in range(num_clients)
    ]

