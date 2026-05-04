"""

Rank Allocator Module

---------------------

This module provides a decoupled RankAllocator class to manage LoRA rank 

assignment across a federated network. It supports:

1. Homogeneous Allocation: Assigns a fixed rank to all clients.

2. Adaptive Allocation: Assigns ranks based on client data volume 

   (or other heuristic metrics).

"""

import numpy as np

class RankAllocator:
    def __init__(self, strategy="homogeneous", base_rank=4):
        self.strategy = strategy
        self.base_rank = base_rank

    def get_allocation(self, client_ids, client_data_stats=None):
        if self.strategy == "homogeneous":
            return {cid: self.base_rank for cid in client_ids}

        elif self.strategy == "adaptive":
            if client_data_stats is None:
                raise ValueError("Adaptive strategy requires client_data_stats.")
            
            allocation = {}
            data_sizes = [client_data_stats[cid] for cid in client_ids]
            min_size, max_size = min(data_sizes), max(data_sizes)
            
            for cid in client_ids:
                norm = (np.log(client_data_stats[cid]) - np.log(min_size)) / \
                       (np.log(max_size) - np.log(min_size) + 1e-6)
                
                rank = int(2 ** round(norm * 3 + 1)) 
                allocation[cid] = rank
            return allocation
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

client_list = ["client_01", "client_02", "client_03", "client_04"]
data_volume = {"client_01": 100, "client_02": 500, "client_03": 1000, "client_04": 2000}

homo_allocator = RankAllocator(strategy="homogeneous", base_rank=4)
homo_mapping = homo_allocator.get_allocation(client_list)
print(f"Homogeneous Mapping: {homo_mapping}")

adaptive_allocator = RankAllocator(strategy="adaptive")
adaptive_mapping = adaptive_allocator.get_allocation(client_list, client_data_stats=data_volume)
print(f"Adaptive Mapping: {adaptive_mapping}")

def init_clients(allocation_map):
    clients = {}
    for cid, rank in allocation_map.items():
        print(f"Initializing {cid} with Rank: {rank}")
        clients[cid] = rank 
    return clients
