import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r=4):
        super().__init__()
        # Main weight and bias
        self.W = nn.Parameter(torch.randn(out_f, in_f) * 0.01)
        self.b = nn.Parameter(torch.zeros(out_f))
        
        # Low-rank adapters
        self.A = nn.Parameter(torch.randn(r, in_f) * 0.01)
        self.B = nn.Parameter(torch.randn(out_f, r) * 0.01)

    def forward(self, x):
        # Result = xW^T + (xA^T)B^T + bias
        return x @ self.W.t() + (x @ self.A.t()) @ self.B.t() + self.b

class LoRAMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, r=4, batch_first=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=batch_first)
        
        # LoRA adapters for Query, Key, and Value projections
        self.lora_q_A = nn.Parameter(torch.randn(r, d_model) * 0.01)
        self.lora_q_B = nn.Parameter(torch.randn(d_model, r) * 0.01)
        self.lora_k_A = nn.Parameter(torch.randn(r, d_model) * 0.01)
        self.lora_k_B = nn.Parameter(torch.randn(d_model, r) * 0.01)
        self.lora_v_A = nn.Parameter(torch.randn(r, d_model) * 0.01)
        self.lora_v_B = nn.Parameter(torch.randn(d_model, r) * 0.01)

    def forward(self, query, key, value, key_padding_mask=None):
        # Apply low-rank deltas before the standard attention operation
        q_delta = (query @ self.lora_q_A.t()) @ self.lora_q_B.t()
        k_delta = (key   @ self.lora_k_A.t()) @ self.lora_k_B.t()
        v_delta = (value @ self.lora_v_A.t()) @ self.lora_v_B.t()
        
        out, _ = self.attn(query + q_delta, key + k_delta, value + v_delta, 
                           key_padding_mask=key_padding_mask)
        return out
