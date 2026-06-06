# LoRA layer definitions
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    output = xW + (xA)B + b
    A: [r, in_f]   B: [out_f, r]
    """
    def __init__(self, in_f, out_f, r=4):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_f, in_f) * 0.01, requires_grad=False)
        self.b = nn.Parameter(torch.zeros(out_f), requires_grad=False)
        self.A = nn.Parameter(torch.randn(r, in_f) * 0.01)
        self.B = nn.Parameter(torch.randn(out_f, r) * 0.01)

    def forward(self, x):
        return x @ self.W.t() + (x @ self.A.t()) @ self.B.t() + self.b


class LoRAMultiheadAttention(nn.Module):
    """
    Multihead attention with LoRA on Q, K, V projections.
    """
    def __init__(self, d_model, num_heads, r=4, batch_first=True):
        super().__init__()
        self.attn     = nn.MultiheadAttention(d_model, num_heads, batch_first=batch_first)
        self.lora_q_A = nn.Parameter(torch.randn(r, d_model) * 0.01)
        self.lora_q_B = nn.Parameter(torch.randn(d_model, r) * 0.01)
        self.lora_k_A = nn.Parameter(torch.randn(r, d_model) * 0.01)
        self.lora_k_B = nn.Parameter(torch.randn(d_model, r) * 0.01)
        self.lora_v_A = nn.Parameter(torch.randn(r, d_model) * 0.01)
        self.lora_v_B = nn.Parameter(torch.randn(d_model, r) * 0.01)

    def forward(self, query, key, value, key_padding_mask=None):
        q = query + (query @ self.lora_q_A.t()) @ self.lora_q_B.t()
        k = key   + (key   @ self.lora_k_A.t()) @ self.lora_k_B.t()
        v = value + (value @ self.lora_v_A.t()) @ self.lora_v_B.t()
        out, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask)
        return out
