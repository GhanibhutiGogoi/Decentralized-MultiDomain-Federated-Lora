"""LoRA FLOPs accounting utilities."""

from Source.Models.lora import LoRALinear, LoRAMultiheadAttention


def estimate_lora_flops(model, batch_size, rank):
    """
    Estimate LoRA adapter FLOPs for one forward+backward batch.

    For each LoRA linear adapter, forward cost is approximately:
        batch_size * (in_features * rank + rank * out_features)
    We multiply by 3 to approximate forward plus backward.
    """
    flops = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            in_f = module.A.shape[1]
            out_f = module.B.shape[0]
            flops += batch_size * (in_f * rank + rank * out_f)
        elif isinstance(module, LoRAMultiheadAttention):
            d = module.lora_q_A.shape[1]
            flops += 3 * batch_size * (d * rank + rank * d)
    return flops * 3


def compute_round_flops(model, loader, rank, epochs):
    """Total client FLOPs for one federated round."""
    return estimate_lora_flops(model, loader.batch_size, rank) * len(loader) * epochs

