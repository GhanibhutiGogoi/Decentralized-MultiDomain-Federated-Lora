import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CONFIG

NUM_ROUNDS    = 5
NUM_CLIENTS   = 3

# Computational capability proxy: batch size
# C0=weak (small batch), C1=medium, C2=strong (large batch)
CLIENT_BATCH_SIZES = (16, 64, 256)

# Homo baseline: high rank to stress weak clients
FIXED_RANK = 32

# Adaptive: each client's max rank is gated by its batch size
BATCH_TO_MAX_RANK   = {16: 4, 64: 8, 256: 16}
ALL_CANDIDATE_RANKS = [2, 4, 6, 8, 12, 16, 24, 32]

CLIENT_COLORS = ["#378ADD", "#1D9E75", "#EF9F27"]
HOMO_COL      = "#C0392B"
ADAPT_COL     = "#1A6B9A"

# LoRA key suffixes
LORA_A_SUFFIXES = (".A", ".lora_q_A", ".lora_k_A", ".lora_v_A")
LORA_B_SUFFIXES = (".B", ".lora_q_B", ".lora_k_B", ".lora_v_B")
LORA_SUFFIXES   = LORA_A_SUFFIXES + LORA_B_SUFFIXES

CLIENT_EPOCHS = 3   

# FLOPS ESTIMATION (computational waste)

def estimate_lora_flops(model, batch_size, rank):
    """
    Estimate FLOPs for one forward+backward pass through LoRA layers.
    LoRA adds (xA)B to each linear: costs batch_size * (in_f*r + r*out_f) per layer.
    Multiply by 3 to approximate backward pass (≈2× forward) + forward.
    """
    flops = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            in_f  = module.A.shape[1]  
            out_f = module.B.shape[0]   
          
            flops += batch_size * (in_f * rank + rank * out_f)
        elif isinstance(module, LoRAMultiheadAttention):
            d = module.lora_q_A.shape[1]  
            r = module.lora_q_A.shape[0]
            
            flops += 3 * batch_size * (d * rank + rank * d)
    return flops * 3  


def compute_round_flops(model, loader, rank, epochs):
    """Total FLOPs for one client in one round = flops_per_batch * num_batches * epochs."""
    bs          = loader.batch_size
    flops_batch = estimate_lora_flops(model, bs, rank)
    num_batches = len(loader)
    return flops_batch * num_batches * epochs

# LoRA MODULES

class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r=4):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_f, in_f) * 0.01)
        self.b = nn.Parameter(torch.zeros(out_f))
        self.A = nn.Parameter(torch.randn(r, in_f) * 0.01)   # [r, in_f]
        self.B = nn.Parameter(torch.randn(out_f, r) * 0.01)  # [out_f, r]

    def forward(self, x):
        return x @ self.W.t() + (x @ self.A.t()) @ self.B.t() + self.b


class LoRAMultiheadAttention(nn.Module):
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

# MODELS

class CNN(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, r=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),    nn.ReLU(), nn.MaxPool2d(2))
        feat = 64 * 8 * 8 if in_ch == 3 else 64 * 7 * 7
        self.fc1 = LoRALinear(feat, 128, r)
        self.fc2 = LoRALinear(128, num_classes, r)

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        return self.fc2(torch.relu(self.fc1(x)))


class MLP(nn.Module):
    def __init__(self, in_dim, num_classes=10, r=4):
        super().__init__()
        self.fc1 = LoRALinear(in_dim, 128, r)
        self.fc2 = LoRALinear(128, num_classes, r)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x.view(x.size(0), -1))))


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed=64, hidden=128, num_layers=2, num_classes=4, r=4):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed, padding_idx=0)
        self.lstm    = nn.LSTM(embed, hidden, num_layers=num_layers,
                               batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.dropout = nn.Dropout(0.3)
        self.fc      = LoRALinear(hidden, num_classes, r)

    def forward(self, x):
        _, (h, _) = self.lstm(self.embed(x))
        return self.fc(self.dropout(h[-1]))


class BERTStyleModel(nn.Module):
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
        B, T = x.shape; T = min(T, self.max_len - 1); x = x[:, :T]
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h   = self.dropout(self.token_embed(x) + self.pos_embed(pos))
        h   = torch.cat([self.cls_token.expand(B, -1, -1), h], dim=1)
        for l in self.layers:
            h = l["norm1"](h + l["attn"](h, h, h))
            h = l["norm2"](h + l["ff2"](torch.relu(l["ff1"](h))))
        return self.head(h[:, 0, :])


class GPTStyleModel(nn.Module):
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
        B, T = x.shape; T = min(T, self.max_len); x = x[:, :T]
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h   = self.dropout(self.token_embed(x) + self.pos_embed(pos))
        for l in self.layers:
            h = l["norm1"](h + l["attn"](h, h, h))
            h = l["norm2"](h + l["ff2"](torch.relu(l["ff1"](h))))
        return self.head(h[:, -1, :])


class TabularMLP(nn.Module):
    def __init__(self, in_dim, num_classes=2, r=4):
        super().__init__()
        self.net  = nn.Sequential(
            LoRALinear(in_dim, 256, r), nn.ReLU(), nn.Dropout(0.3),
            LoRALinear(256, 128, r),   nn.ReLU(), nn.Dropout(0.2),
            LoRALinear(128, 64, r),    nn.ReLU())
        self.head = LoRALinear(64, num_classes, r)

    def forward(self, x): return self.head(self.net(x))


class AudioCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=35, r=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=80, stride=16), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),           nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),          nn.ReLU(),
            nn.AdaptiveAvgPool1d(8))
        self.fc      = LoRALinear(128 * 8, 256, r)
        self.head    = LoRALinear(256, num_classes, r)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        return self.head(self.dropout(torch.relu(self.fc(x))))

# DATASETS

class AGNewsDataset(Dataset):
    VOCAB_SIZE = 10000; MAX_LEN = 64

    def __init__(self, split="train"):
        super().__init__()
        try:
            from torchtext.datasets import AG_NEWS
            from torchtext.data.utils import get_tokenizer
            from torchtext.vocab import build_vocab_from_iterator
            tokenizer = get_tokenizer("basic_english")
            raw  = list(AG_NEWS(split=split))
            vocab = build_vocab_from_iterator(
                (tokenizer(t) for _, t in raw),
                specials=["<pad>", "<unk>"], max_tokens=self.VOCAB_SIZE)
            vocab.set_default_index(vocab["<unk>"])
            self.data = []
            for label, text in raw:
                ids = vocab(tokenizer(text)[:self.MAX_LEN])
                ids += [0] * (self.MAX_LEN - len(ids))
                self.data.append((torch.tensor(ids, dtype=torch.long), int(label) - 1))
        except Exception as e:
            print(f"[AGNews] {e}. Using synthetic data.")
            rng = np.random.RandomState(42 if split == "train" else 7)
            n   = 5000 if split == "train" else 1000
            self.data = [(torch.randint(1, self.VOCAB_SIZE, (self.MAX_LEN,)),
                          rng.randint(0, 4)) for _ in range(n)]

    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        x, y = self.data[i]; return x, int(y)


class TabularDataset(Dataset):
    def __init__(self, split="train"):
        super().__init__()
        try:
            import urllib.request, os
            url  = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                    "heart-disease/processed.cleveland.data")
            path = "./data/heart.csv"
            os.makedirs("./data", exist_ok=True)
            if not os.path.exists(path):
                urllib.request.urlretrieve(url, path)
            df = pd.read_csv(path, header=None, na_values="?").dropna()
            X  = df.iloc[:, :-1].values.astype(np.float32)
            y  = (df.iloc[:, -1].values > 0).astype(np.int64)
        except Exception as e:
            print(f"[Tabular] {e}. Using synthetic data.")
            rng = np.random.RandomState(0)
            n   = 4000 if split == "train" else 800
            centers = rng.randn(4, 20) * 2
            X = np.vstack([rng.randn(n//4, 20) + centers[i] for i in range(4)]).astype(np.float32)
            y = np.concatenate([np.full(n//4, i, dtype=np.int64) for i in range(4)])
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        rng = np.random.RandomState(1); idx = rng.permutation(len(X))
        sp  = int(0.8 * len(X))
        idx = idx[:sp] if split == "train" else idx[sp:]
        self.X = torch.from_numpy(X[idx]); self.y = torch.from_numpy(y[idx])
        self.in_dim = X.shape[1]; self.num_classes = int(y.max()) + 1

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class AudioDataset(Dataset):
    SAMPLE_RATE = 16000; NUM_CLASSES = 35

    def __init__(self, split="train"):
        super().__init__()
        self._loaded = False
        try:
            import torchaudio
            subset = "training" if split == "train" else "validation"
            ds = torchaudio.datasets.SPEECHCOMMANDS("./data", download=True, subset=subset)
            all_labels = sorted({ds[i][2] for i in range(min(500, len(ds)))})
            self.label2idx   = {l: i for i, l in enumerate(all_labels)}
            self.NUM_CLASSES = len(self.label2idx)
            self.data = ds; self._loaded = True
        except Exception as e:
            print(f"[Audio] {e}. Using synthetic data.")
            rng = np.random.RandomState(3 if split == "train" else 9)
            n   = 3000 if split == "train" else 600
            self.synth = [(torch.from_numpy(rng.randn(1, self.SAMPLE_RATE).astype(np.float32)),
                           rng.randint(0, self.NUM_CLASSES)) for _ in range(n)]

    def __len__(self):
        return len(self.data) if self._loaded else len(self.synth)

    def __getitem__(self, i):
        if self._loaded:
            waveform, sr, label, *_ = self.data[i]
            t = self.SAMPLE_RATE
            waveform = (torch.nn.functional.pad(waveform, (0, t - waveform.shape[-1]))
                        if waveform.shape[-1] < t else waveform[:, :t])
            return waveform, self.label2idx.get(label, 0)
        return self.synth[i]

# HELPERS

def split_dataset(dataset, num_clients=3):
    n    = len(dataset); size = n // num_clients
    return [Subset(dataset, list(range(i * size, (i + 1) * size)))
            for i in range(num_clients)]


def is_lora_key(k):
    return any(k.endswith(s) for s in LORA_SUFFIXES)


def is_lora_B_key(k):
    """B matrices have shape [out_f, r] — rank lives on dim-1."""
    return any(k.endswith(s) for s in LORA_B_SUFFIXES)


def project_tensor_to_rank(t, target_rank, rank_dim=0):
    """
    Project a 2-D LoRA matrix to target_rank along rank_dim.
      rank_dim=0  →  A matrices, shape [r, in_f]
      rank_dim=1  →  B matrices, shape [out_f, r]
    Truncation via SVD. Expansion via zero-padding.

    Fix: SVD of [m×n] with m<n yields Vh with only m rows. If those rows
    are fewer than target_rank, zero-pad up to target_rank before transposing
    so the returned tensor always has the correct shape along rank_dim.
    """
    cur_rank = t.shape[rank_dim]
    if cur_rank == target_rank:
        return t.clone()

    if cur_rank > target_rank:
        # Normalise so rank is always on dim-0 before SVD
        mat = t.float() if rank_dim == 0 else t.float().t()   
        _, _, Vh = torch.linalg.svd(mat, full_matrices=False)  
        actual_rows = Vh.shape[0]
        if actual_rows >= target_rank:
            compressed = Vh[:target_rank, :]                   
        else:
            pad = torch.zeros(target_rank - actual_rows, Vh.shape[1],
                              dtype=Vh.dtype, device=Vh.device)
            compressed = torch.cat([Vh, pad], dim=0)        
        result = compressed if rank_dim == 0 else compressed.t()
        return result.to(t.dtype)

    pad_shape = list(t.shape)
    pad_shape[rank_dim] = target_rank - cur_rank
    pad = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=rank_dim)


def load_global_state(model, global_state):
    """
    Load global_state into model, projecting LoRA matrices to the
    model's rank when sizes differ.
      A keys: rank on dim-0
      B keys: rank on dim-1
    """
    local = model.state_dict()
    for k in local:
        if k not in global_state:
            continue
        g = global_state[k]
        if g.shape == local[k].shape:
            local[k] = g.clone()
        elif is_lora_key(k) and g.dim() == 2:
            rank_dim    = 1 if is_lora_B_key(k) else 0
            target_rank = local[k].shape[rank_dim]
            local[k]    = project_tensor_to_rank(g, target_rank, rank_dim)
    model.load_state_dict(local)


def compute_quality_score(model, loader, loss_fn, num_batches=5):
    """Quality score = 1 / (1 + avg_train_loss). Higher = better."""
    model.eval()
    total_loss = 0.0
    count      = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= num_batches:
                break
            x, y = x.to(device), y.to(device)
            total_loss += loss_fn(model(x), y).item()
            count      += 1
    avg_loss = total_loss / max(count, 1)
    return 1.0 / (1.0 + avg_loss)


def fedavg_quality_weighted(weights, samples, quality_scores, target_rank, ref_sd):
    """FedAvg weighted by samples × quality_score."""
    raw    = [s * q for s, q in zip(samples, quality_scores)]
    total  = sum(raw)
    norm_w = [r / total for r in raw]

    agg = {}
    for k in ref_sd:
        if is_lora_key(k):
            rank_dim  = 1 if is_lora_B_key(k) else 0
            projected = []
            for w, nw in zip(weights, norm_w):
                if k not in w or w[k].dim() != 2:
                    continue
                t = project_tensor_to_rank(w[k].to(device), target_rank, rank_dim)
                projected.append((t, nw))
            agg[k] = (sum(t * nw for t, nw in projected)
                      if projected else ref_sd[k])
        else:
            contribs = [(w[k], nw) for w, nw in zip(weights, norm_w)
                        if k in w and w[k].shape == ref_sd[k].shape]
            agg[k] = (sum(t * nw for t, nw in contribs)
                      if contribs else ref_sd[k])

    return agg


def train_client(model, loader, epochs):
    model.train()
    opt     = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    total   = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
            total += y.size(0)
    return model.state_dict(), total


def evaluate(model, loader):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total if total else 0.0

# ADAPTIVE RANK SELECTOR

def estimate_optimal_rank(model, loader, loss_fn, batch_size, num_batches=3):
    """
    Gradient-signal rank probe with batch-size-based floor.
    Candidate ranks capped by BATCH_TO_MAX_RANK.
    """
    max_rank   = BATCH_TO_MAX_RANK.get(batch_size, 4)
    candidates = [r for r in ALL_CANDIDATE_RANKS if r <= max_rank]

    sorted_bs       = sorted(BATCH_TO_MAX_RANK.keys())
    bs_idx          = sorted_bs.index(batch_size)
    capability_bias = bs_idx / (len(sorted_bs) - 1)

    model.train()
    opt          = optim.Adam(model.parameters(), lr=0.001)
    stable_ranks = []

    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= num_batches:
            break
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        for _, param in model.named_parameters():
            if param.grad is not None and param.dim() == 2:
                G           = param.grad.float()
                frob_sq     = (G ** 2).sum().item()
                spectral_sq = torch.linalg.matrix_norm(G, ord=2).item() ** 2
                if spectral_sq > 1e-12:
                    stable_ranks.append(frob_sq / spectral_sq)

    if not stable_ranks:
        return candidates[-1]

    median_sr   = float(np.median(stable_ranks))
    signal_frac = min(median_sr / max(ALL_CANDIDATE_RANKS), 1.0)
    combined    = max(signal_frac, capability_bias)
    idx         = int(round(combined * (len(candidates) - 1)))
    return candidates[min(idx, len(candidates) - 1)]
  
# EXPERIMENT RUNNERS

def run_homo(name, model_fn, trainset, testloader):
    """All clients use FIXED_RANK=32. Tracks FLOPs per client per round."""
    print(f"\n  [HOMO] {name}")
    loss_fn = nn.CrossEntropyLoss()
    clients = split_dataset(trainset, NUM_CLIENTS)
    loaders = [DataLoader(c, batch_size=CLIENT_BATCH_SIZES[i], shuffle=True)
               for i, c in enumerate(clients)]

    global_state = model_fn(FIXED_RANK).to(device).state_dict()
    acc_curve    = []
    flops_hist   = []

    for rnd in range(NUM_ROUNDS):
        weights, samples, quality_scores = [], [], []
        round_flops = []

        for i in range(NUM_CLIENTS):
            local = model_fn(FIXED_RANK).to(device)
            load_global_state(local, global_state)

            # Compute FLOPs before training (rank is fixed)
            f = compute_round_flops(local, loaders[i], FIXED_RANK, CLIENT_EPOCHS)
            round_flops.append(f)

            w, s = train_client(local, loaders[i], CLIENT_EPOCHS)
            q    = compute_quality_score(local, loaders[i], loss_fn)
            weights.append(w); samples.append(s); quality_scores.append(q)

        flops_hist.append(round_flops)

        ref_sd       = model_fn(FIXED_RANK).to(device).state_dict()
        global_state = fedavg_quality_weighted(
            weights, samples, quality_scores, FIXED_RANK, ref_sd)

        eval_m = model_fn(FIXED_RANK).to(device)
        load_global_state(eval_m, global_state)
        acc = evaluate(eval_m, testloader)
        acc_curve.append(acc)

        total_f = sum(round_flops)
        print(f"    Round {rnd+1}/{NUM_ROUNDS} | acc={acc:.2f}% | "
              f"total_flops={total_f:.2e} | "
              f"{' | '.join(f'C{i}(r={FIXED_RANK},flops={round_flops[i]:.2e},q={quality_scores[i]:.3f})' for i in range(NUM_CLIENTS))}")

    return acc_curve, acc_curve[-1], flops_hist


def run_adaptive(name, model_fn, trainset, testloader):
    """Each client picks rank suited to its capability. Tracks FLOPs per client per round."""
    print(f"\n  [ADAPTIVE] {name}")
    loss_fn = nn.CrossEntropyLoss()
    clients = split_dataset(trainset, NUM_CLIENTS)
    loaders = [DataLoader(c, batch_size=CLIENT_BATCH_SIZES[i], shuffle=True)
               for i, c in enumerate(clients)]

    global_state = model_fn(FIXED_RANK).to(device).state_dict()
    acc_curve    = []
    flops_hist   = []
    rank_hist    = []

    for rnd in range(NUM_ROUNDS):
        weights, samples, quality_scores, ranks = [], [], [], []
        round_flops = []

        for i in range(NUM_CLIENTS):
            bs      = CLIENT_BATCH_SIZES[i]
            probe_r = BATCH_TO_MAX_RANK[bs]

            probe = model_fn(probe_r).to(device)
            load_global_state(probe, global_state)
            chosen_r = estimate_optimal_rank(probe, loaders[i], loss_fn, bs)
            ranks.append(chosen_r)

            local = model_fn(chosen_r).to(device)
            load_global_state(local, global_state)

            # Compute FLOPs using chosen rank
            f = compute_round_flops(local, loaders[i], chosen_r, CLIENT_EPOCHS)
            round_flops.append(f)

            w, s = train_client(local, loaders[i], CLIENT_EPOCHS)
            q    = compute_quality_score(local, loaders[i], loss_fn)
            weights.append(w); samples.append(s); quality_scores.append(q)

        flops_hist.append(round_flops)
        rank_hist.append(ranks[:])

        ref_sd       = model_fn(FIXED_RANK).to(device).state_dict()
        global_state = fedavg_quality_weighted(
            weights, samples, quality_scores, FIXED_RANK, ref_sd)

        eval_m = model_fn(FIXED_RANK).to(device)
        load_global_state(eval_m, global_state)
        acc = evaluate(eval_m, testloader)
        acc_curve.append(acc)

        total_f = sum(round_flops)
        print(f"    Round {rnd+1}/{NUM_ROUNDS} | acc={acc:.2f}% | "
              f"total_flops={total_f:.2e} | "
              f"{' | '.join(f'C{i}(r={ranks[i]},flops={round_flops[i]:.2e},q={quality_scores[i]:.3f})' for i in range(NUM_CLIENTS))}")

    return acc_curve, acc_curve[-1], flops_hist, rank_hist

# LOAD DATASETS

print("\n=== Loading Datasets ===")

t_cifar = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,)*3, (0.5,)*3)])
cifar_train      = torchvision.datasets.CIFAR10("./data", train=True,  download=True, transform=t_cifar)
cifar_test       = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=t_cifar)
cifar_testloader = DataLoader(cifar_test, batch_size=64)

t_fashion          = transforms.Compose([transforms.ToTensor()])
fashion_train      = torchvision.datasets.FashionMNIST("./data", train=True,  download=True, transform=t_fashion)
fashion_test       = torchvision.datasets.FashionMNIST("./data", train=False, download=True, transform=t_fashion)
fashion_testloader = DataLoader(fashion_test, batch_size=64)

print("Loading AG News...")
agnews_train      = AGNewsDataset("train")
agnews_test       = AGNewsDataset("test")
agnews_testloader = DataLoader(agnews_test, batch_size=64)

print("Loading Tabular...")
tabular_train      = TabularDataset("train")
tabular_test       = TabularDataset("test")
tabular_testloader = DataLoader(tabular_test, batch_size=64)
TAB_IN_DIM         = tabular_train.in_dim
TAB_N_CLASSES      = tabular_train.num_classes

print("Loading Audio...")
audio_train      = AudioDataset("train")
audio_test       = AudioDataset("test")
audio_testloader = DataLoader(audio_test, batch_size=32)
AUDIO_N_CLASSES  = audio_train.NUM_CLASSES

VOCAB = AGNewsDataset.VOCAB_SIZE

print(f"\n  CIFAR-10  train={len(cifar_train)}, test={len(cifar_test)}")
print(f"  Fashion   train={len(fashion_train)}, test={len(fashion_test)}")
print(f"  AG News   train={len(agnews_train)}, test={len(agnews_test)}")
print(f"  Tabular   train={len(tabular_train)}, test={len(tabular_test)}")
print(f"  Audio     train={len(audio_train)}, test={len(audio_test)}")

# MODEL FACTORIES

EXPERIMENTS = [
    ("CIFAR-CNN",   lambda r: CNN(3, 10, r),                            cifar_train,   cifar_testloader),
    ("Fashion-MLP", lambda r: MLP(28*28, 10, r),                        fashion_train, fashion_testloader),
    ("AGNews-LSTM", lambda r: LSTMModel(VOCAB, 64, 128, 2, 4, r),       agnews_train,  agnews_testloader),
    ("Tabular-MLP", lambda r: TabularMLP(TAB_IN_DIM, TAB_N_CLASSES, r), tabular_train, tabular_testloader),
    ("Audio-1DCNN", lambda r: AudioCNN(1, AUDIO_N_CLASSES, r),          audio_train,   audio_testloader),
]

EXP_NAMES = [e[0] for e in EXPERIMENTS]


print("\n=== Running Experiments ===")
print(f"  5 experiments × 2 modes × {NUM_ROUNDS} rounds = {5*2*NUM_ROUNDS} training rounds\n")
print(f"  Capability proxy : batch size {CLIENT_BATCH_SIZES} (C0=weak, C1=medium, C2=strong)")
print(f"  Homo rank        : FIXED_RANK={FIXED_RANK} (stresses weak clients)")
print(f"  Adaptive ceilings: {BATCH_TO_MAX_RANK}\n")

results = {}
for name, model_fn, trainset, testloader in EXPERIMENTS:
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {name}")
    print(f"{'='*60}")
    results[name] = {
        "homo":     run_homo(name,     model_fn, trainset, testloader),
        "adaptive": run_adaptive(name, model_fn, trainset, testloader),
    }

x_rounds = np.arange(1, NUM_ROUNDS + 1)

# FIGURE 1 — Accuracy curves: homo vs adaptive

print("\nGenerating Figure 1: Accuracy curves...")
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
fig.suptitle(
    f"Federated LoRA — Accuracy over {NUM_ROUNDS} Rounds\n"
    f"Red = Homogeneous (r={FIXED_RANK})  |  Blue = Adaptive (rank per capability)",
    fontsize=11, fontweight="bold")

for ax, name in zip(axes, EXP_NAMES):
    h_acc = results[name]["homo"][0]
    a_acc = results[name]["adaptive"][0]
    ax.plot(x_rounds, h_acc, color=HOMO_COL,  lw=2.5, marker="s",
            markersize=6, label=f"Homo r={FIXED_RANK}")
    ax.plot(x_rounds, a_acc, color=ADAPT_COL, lw=2.5, marker="o",
            markersize=6, label="Adaptive")
    ax.fill_between(x_rounds, h_acc, a_acc,
                    where=[a >= h for a, h in zip(a_acc, h_acc)],
                    alpha=0.15, color=ADAPT_COL)
    ax.fill_between(x_rounds, h_acc, a_acc,
                    where=[h > a for a, h in zip(a_acc, h_acc)],
                    alpha=0.10, color=HOMO_COL)
    ax.set_title(name, fontsize=9, fontweight="bold")
    ax.set_xlabel("Round", fontsize=8); ax.set_ylabel("Accuracy (%)", fontsize=8)
    ax.set_xticks(x_rounds); ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=7); ax.set_ylim(bottom=0)
    ax.annotate(f"{h_acc[-1]:.1f}%", xy=(NUM_ROUNDS, h_acc[-1]),
                xytext=(3, -10), textcoords="offset points",
                fontsize=7, color=HOMO_COL, fontweight="bold")
    ax.annotate(f"{a_acc[-1]:.1f}%", xy=(NUM_ROUNDS, a_acc[-1]),
                xytext=(3, 3), textcoords="offset points",
                fontsize=7, color=ADAPT_COL, fontweight="bold")

plt.tight_layout()
plt.savefig("fig1_accuracy_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig1_accuracy_curves.png")

# FIGURE 2 — Adaptive rank per client per round

print("Generating Figure 2: Adaptive rank per client...")
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
fig.suptitle(
    "Capability-Aware Adaptive LoRA — Rank Chosen per Client per Round\n"
    "Dotted = client ceiling  |  Dashed red = homo baseline (r=32)",
    fontsize=12, fontweight="bold")

cap_labels = [f"C{i}(bs={CLIENT_BATCH_SIZES[i]},max_r={BATCH_TO_MAX_RANK[CLIENT_BATCH_SIZES[i]]})"
              for i in range(NUM_CLIENTS)]

for ax, name in zip(axes, EXP_NAMES):
    rank_hist = results[name]["adaptive"][3]
    for ci in range(NUM_CLIENTS):
        vals = [rank_hist[rnd][ci] for rnd in range(NUM_ROUNDS)]
        ax.plot(x_rounds, vals, color=CLIENT_COLORS[ci], lw=2,
                marker="o", markersize=5, label=cap_labels[ci])
        ax.hlines(BATCH_TO_MAX_RANK[CLIENT_BATCH_SIZES[ci]],
                  x_rounds[0], x_rounds[-1],
                  colors=CLIENT_COLORS[ci], lw=1, linestyles="dotted", alpha=0.5)
    ax.hlines(FIXED_RANK, x_rounds[0], x_rounds[-1],
              colors=HOMO_COL, lw=1.5, linestyles="dashed",
              alpha=0.8, label=f"Homo r={FIXED_RANK}")
    ax.set_title(name, fontsize=9, fontweight="bold")
    ax.set_xlabel("Round", fontsize=8); ax.set_ylabel("LoRA Rank", fontsize=8)
    ax.set_xticks(x_rounds); ax.set_yticks(ALL_CANDIDATE_RANKS)
    ax.set_ylim(0, max(ALL_CANDIDATE_RANKS) + 2)
    ax.grid(alpha=0.2, linestyle="--"); ax.legend(fontsize=6)

plt.tight_layout()
plt.savefig("fig2_adaptive_rank_per_client.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig2_adaptive_rank_per_client.png")

# FIGURE 3 — Final accuracy bar: homo vs adaptive

print("Generating Figure 3: Final accuracy comparison...")
fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle(f"Final Accuracy (Round {NUM_ROUNDS}): Adaptive vs Homogeneous\n"
             "Δ = Adaptive − Homogeneous", fontsize=12, fontweight="bold")

x_exp = np.arange(len(EXP_NAMES)); w = 0.35
h_finals = [results[n]["homo"][1]     for n in EXP_NAMES]
a_finals = [results[n]["adaptive"][1] for n in EXP_NAMES]

ax.bar(x_exp - w/2, h_finals, w, color=HOMO_COL,  alpha=0.85,
       label=f"Homogeneous r={FIXED_RANK}", edgecolor="white")
ax.bar(x_exp + w/2, a_finals, w, color=ADAPT_COL, alpha=0.85,
       label="Adaptive (capability-aware)", edgecolor="white")

for xi, (h, a) in enumerate(zip(h_finals, a_finals)):
    delta = a - h
    ax.text(xi, max(h, a) + 0.8, f"{delta:+.1f}%",
            ha="center", fontsize=9,
            color=ADAPT_COL if delta >= 0 else HOMO_COL, fontweight="bold")

ax.set_xticks(x_exp); ax.set_xticklabels(EXP_NAMES, fontsize=9)
ax.set_ylabel("Final Accuracy (%)", fontsize=10)
ax.grid(axis="y", alpha=0.2, linestyle="--")
ax.legend(fontsize=9, framealpha=0.8)
plt.tight_layout()
plt.savefig("fig3_final_accuracy_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig3_final_accuracy_bar.png")

# FIGURE 4 — Total FLOPs per round: homo vs adaptive (all clients combined)

print("Generating Figure 4: Total FLOPs per round...")
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
fig.suptitle(
    "Total FLOPs per Round — Homogeneous vs Adaptive\n"
    "Lower FLOPs = less compute burden on the federation",
    fontsize=12, fontweight="bold")

for ax, name in zip(axes, EXP_NAMES):
    h_flops_total = [sum(results[name]["homo"][2][r])     for r in range(NUM_ROUNDS)]
    a_flops_total = [sum(results[name]["adaptive"][2][r]) for r in range(NUM_ROUNDS)]

    ax.plot(x_rounds, h_flops_total, color=HOMO_COL,  lw=2.5, marker="s",
            markersize=6, label=f"Homo r={FIXED_RANK}")
    ax.plot(x_rounds, a_flops_total, color=ADAPT_COL, lw=2.5, marker="o",
            markersize=6, label="Adaptive")
    ax.fill_between(x_rounds, a_flops_total, h_flops_total,
                    where=[h >= a for h, a in zip(h_flops_total, a_flops_total)],
                    alpha=0.15, color=ADAPT_COL,
                    label="FLOPs saved")
    ax.set_title(name, fontsize=9, fontweight="bold")
    ax.set_xlabel("Round", fontsize=8)
    ax.set_ylabel("Total FLOPs", fontsize=8)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_xticks(x_rounds)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("fig4_total_flops_per_round.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig4_total_flops_per_round.png")

# FIGURE 5 — FLOPs per client per round: homo vs adaptive (stacked bars)

print("Generating Figure 5: FLOPs per client per round...")
fig, axes = plt.subplots(2, 5, figsize=(22, 8))
fig.suptitle(
    "FLOPs per Client per Round\n"
    "Top = Homogeneous (r=32, all clients equal cost)  |  Bottom = Adaptive (weak clients cheaper)",
    fontsize=12, fontweight="bold")

for ei, name in enumerate(EXP_NAMES):
    for row, mode in enumerate(["homo", "adaptive"]):
        ax       = axes[row][ei]
        fh       = results[name][mode][2]  
        bottoms  = np.zeros(NUM_ROUNDS)
        for ci in range(NUM_CLIENTS):
            vals = np.array([fh[r][ci] for r in range(NUM_ROUNDS)], dtype=float)
            ax.bar(x_rounds, vals, bottom=bottoms,
                   color=CLIENT_COLORS[ci], alpha=0.88,
                   label=f"C{ci}(bs={CLIENT_BATCH_SIZES[ci]})",
                   width=0.6, edgecolor="white", linewidth=0.5)
            bottoms += vals
        ax.set_title(f"{name} — {'Homo' if row == 0 else 'Adaptive'}",
                     fontsize=8, fontweight="bold")
        ax.set_xlabel("Round", fontsize=7)
        ax.set_ylabel("FLOPs", fontsize=7)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_xticks(x_rounds)
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.legend(fontsize=6, framealpha=0.7)

plt.tight_layout()
plt.savefig("fig5_flops_per_client.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig5_flops_per_client.png")

# FIGURE 6 — Accuracy vs Total FLOPs (Pareto scatter)

print("Generating Figure 6: Accuracy vs FLOPs Pareto scatter...")
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle(
    "Accuracy vs Total FLOPs — Pareto Efficiency\n"
    "Top-left = better (higher accuracy, lower compute)",
    fontsize=12, fontweight="bold")

for ei, name in enumerate(EXP_NAMES):
    h_flops_sum = sum(sum(results[name]["homo"][2][r])     for r in range(NUM_ROUNDS))
    a_flops_sum = sum(sum(results[name]["adaptive"][2][r]) for r in range(NUM_ROUNDS))
    h_acc_final = results[name]["homo"][1]
    a_acc_final = results[name]["adaptive"][1]

    ax.scatter(h_flops_sum, h_acc_final, color=HOMO_COL,  s=120,
               marker="s", zorder=3)
    ax.scatter(a_flops_sum, a_acc_final, color=ADAPT_COL, s=120,
               marker="o", zorder=3)
    # Arrow from homo → adaptive
    ax.annotate("", xy=(a_flops_sum, a_acc_final),
                xytext=(h_flops_sum, h_acc_final),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
    # Label experiment name at midpoint
    mx = (h_flops_sum + a_flops_sum) / 2
    my = (h_acc_final + a_acc_final) / 2
    ax.text(mx, my + 0.3, name, fontsize=7, ha="center", color="dimgray")

from matplotlib.lines import Line2D
legend_els = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=HOMO_COL,
           markersize=10, label=f"Homo r={FIXED_RANK}"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=ADAPT_COL,
           markersize=10, label="Adaptive"),
]
ax.legend(handles=legend_els, fontsize=9)
ax.set_xlabel("Total FLOPs (all rounds, all clients)", fontsize=10)
ax.set_ylabel("Final Accuracy (%)", fontsize=10)
ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
ax.grid(alpha=0.2, linestyle="--")
plt.tight_layout()
plt.savefig("fig6_pareto_accuracy_flops.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig6_pareto_accuracy_flops.png")


# ============================================================
# SUMMARY CSV
# ============================================================

print("\nGenerating summary CSV...")
rows = []
for name in EXP_NAMES:
    h_acc       = results[name]["homo"][1]
    a_acc       = results[name]["adaptive"][1]
    h_flops_all = results[name]["homo"][2]      # [round][client]
    a_flops_all = results[name]["adaptive"][2]
    rank_hist   = results[name]["adaptive"][3]

    h_total_flops = sum(sum(h_flops_all[r]) for r in range(NUM_ROUNDS))
    a_total_flops = sum(sum(a_flops_all[r]) for r in range(NUM_ROUNDS))
    flops_saved   = h_total_flops - a_total_flops
    flops_saved_pct = 100.0 * flops_saved / h_total_flops if h_total_flops > 0 else 0.0

    all_r = [r for row in rank_hist for r in row]

    row = {
        "Experiment":              name,
        "Homo Rank (fixed)":       FIXED_RANK,
        "Homo Final Acc (%)":      round(h_acc, 2),
        "Adap Final Acc (%)":      round(a_acc, 2),
        "Acc Delta (%)":           round(a_acc - h_acc, 2),
        "Homo Total FLOPs":        int(h_total_flops),
        "Adap Total FLOPs":        int(a_total_flops),
        "FLOPs Saved":             int(flops_saved),
        "FLOPs Saved (%)":         round(flops_saved_pct, 2),
        "Avg Adap Rank (all)":     round(float(np.mean(all_r)), 2),
    }
    for ci in range(NUM_CLIENTS):
        cr              = [rank_hist[rnd][ci] for rnd in range(NUM_ROUNDS)]
        h_client_flops  = sum(h_flops_all[r][ci] for r in range(NUM_ROUNDS))
        a_client_flops  = sum(a_flops_all[r][ci] for r in range(NUM_ROUNDS))
        row[f"C{ci}_bs={CLIENT_BATCH_SIZES[ci]}_avg_rank"]       = round(float(np.mean(cr)), 2)
        row[f"C{ci}_homo_total_flops"]  = int(h_client_flops)
        row[f"C{ci}_adap_total_flops"]  = int(a_client_flops)
        row[f"C{ci}_flops_saved"]       = int(h_client_flops - a_client_flops)
    rows.append(row)

df = pd.DataFrame(rows)
print("\n" + "="*75)
print("CAPABILITY-AWARE ADAPTIVE vs HOMOGENEOUS — SUMMARY")
print("="*75)
print(df.to_string(index=False))
df.to_csv("federated_lora_summary.csv", index=False)
print("\nSaved: federated_lora_summary.csv")

print("\n" + "="*65)
print("ALL OUTPUTS SAVED")
print("="*65)
print("  fig1_accuracy_curves.png        — homo vs adaptive accuracy")
print("  fig2_adaptive_rank_per_client.png — rank chosen per client")
print("  fig3_final_accuracy_bar.png     — final acc comparison + Δ")
print("  fig4_total_flops_per_round.png  — total FLOPs per round")
print("  fig5_flops_per_client.png       — FLOPs breakdown per client")
print("  fig6_pareto_accuracy_flops.png  — accuracy vs FLOPs Pareto")
print("  federated_lora_summary.csv      — full results table")
print("="*65)
print(f"\nClient capabilities:")
for i, bs in enumerate(CLIENT_BATCH_SIZES):
    print(f"  C{i}: batch_size={bs} → max_rank={BATCH_TO_MAX_RANK[bs]}")
