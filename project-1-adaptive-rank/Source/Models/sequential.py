# sequence model definitions


import torch
import torch.nn as nn
from .lora import LoRALinear, LoRAMultiheadAttention


class LSTMModel(nn.Module):
    """
    LSTM-based text classifier (e.g. AG News).
    LoRA applied to the output linear layer.
    """
    def __init__(self, vocab_size, embed=64, hidden=128,
                 num_layers=2, num_classes=4, r=4):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed, padding_idx=0)
        self.lstm    = nn.LSTM(embed, hidden, num_layers=num_layers,
                               batch_first=True,
                               dropout=0.2 if num_layers > 1 else 0)
        self.dropout = nn.Dropout(0.3)
        self.fc      = LoRALinear(hidden, num_classes, r)

    def forward(self, x):
        _, (h, _) = self.lstm(self.embed(x))
        return self.fc(self.dropout(h[-1]))


class BERTStyleModel(nn.Module):
    """
    Encoder-only transformer (BERT-style) text classifier.
    LoRA applied to attention projections and feed-forward layers.
    """
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=2,
                 max_len=128, num_classes=4, r=4):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed   = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([nn.ModuleDict({
            "attn":  LoRAMultiheadAttention(d_model, nhead, r=r),
            "norm1": nn.LayerNorm(d_model),
            "ff1":   LoRALinear(d_model, d_model * 4, r),
            "ff2":   LoRALinear(d_model * 4, d_model, r),
            "norm2": nn.LayerNorm(d_model),
        }) for _ in range(num_layers)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head      = LoRALinear(d_model, num_classes, r)
        self.dropout   = nn.Dropout(0.1)
        self.max_len   = max_len

    def forward(self, x):
        B, T = x.shape
        T = min(T, self.max_len - 1)
        x = x[:, :T]
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h   = self.dropout(self.token_embed(x) + self.pos_embed(pos))
        h   = torch.cat([self.cls_token.expand(B, -1, -1), h], dim=1)
        for l in self.layers:
            h = l["norm1"](h + l["attn"](h, h, h))
            h = l["norm2"](h + l["ff2"](torch.relu(l["ff1"](h))))
        return self.head(h[:, 0, :])


class GPTStyleModel(nn.Module):
    """
    Decoder-style transformer text classifier.
    LoRA applied to attention projections and feed-forward layers.
    """
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=2,
                 max_len=128, num_classes=4, r=4):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed   = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([nn.ModuleDict({
            "attn":  LoRAMultiheadAttention(d_model, nhead, r=r),
            "norm1": nn.LayerNorm(d_model),
            "ff1":   LoRALinear(d_model, d_model * 4, r),
            "ff2":   LoRALinear(d_model * 4, d_model, r),
            "norm2": nn.LayerNorm(d_model),
        }) for _ in range(num_layers)])
        self.head    = LoRALinear(d_model, num_classes, r)
        self.dropout = nn.Dropout(0.1)
        self.max_len = max_len

    def forward(self, x):
        B, T = x.shape
        T = min(T, self.max_len)
        x = x[:, :T]
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h   = self.dropout(self.token_embed(x) + self.pos_embed(pos))
        for l in self.layers:
            h = l["norm1"](h + l["attn"](h, h, h))
            h = l["norm2"](h + l["ff2"](torch.relu(l["ff1"](h))))
        return self.head(h[:, -1, :])
