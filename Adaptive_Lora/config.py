"""Shared configuration for Project 1: capability-aware adaptive LoRA rank."""

NUM_ROUNDS = 5
NUM_CLIENTS = 3
CLIENT_EPOCHS = 3

CLIENT_BATCH_SIZES = (16, 64, 256)
FIXED_RANK = 32

BATCH_TO_MAX_RANK = {16: 4, 64: 8, 256: 16}
ALL_CANDIDATE_RANKS = [2, 4, 6, 8, 12, 16, 24, 32]

CLIENT_COLORS = ["#378ADD", "#1D9E75", "#EF9F27"]
HOMO_COL = "#C0392B"
ADAPT_COL = "#1A6B9A"

LORA_A_SUFFIXES = (".A", ".lora_q_A", ".lora_k_A", ".lora_v_A")
LORA_B_SUFFIXES = (".B", ".lora_q_B", ".lora_k_B", ".lora_v_B")
LORA_SUFFIXES = LORA_A_SUFFIXES + LORA_B_SUFFIXES

