"""
========================
Federated LoRA Experiment Summary
========================

1. Environment Setup:
   - Uses PyTorch, Torchvision, TorchText, Pandas, Matplotlib, NumPy.
   - Device: GPU if available, otherwise CPU.
   - Matplotlib backend: 'Agg' (headless plotting).

2. Visualization Outputs:
   - Accuracy curves per model & modality: 'federated_lora_results.png'
   - Final accuracy bar chart: 'model_comparison_bar.png'
   - Summary CSV: 'federated_lora_summary.csv'
   - Per-client contribution (stacked bars): 'client_contribution_pct.png'
   - FedAvg client weights (doughnuts): 'client_aggregation_weights.png'
   - Cumulative samples per client: 'client_cumulative_samples.png'
   - Contribution % over rounds (line view): 'client_contribution_line.png'
   - FedAvg weight heatmap: 'client_weight_heatmap.png'
   - Combined dashboard (accuracy + contribution): 'collaboration_dashboard.png'

3. Estimated Runtime:
   - CIFAR10/FashionMNIST CNN/MLP: ~10–30s per experiment
   - AGNews LSTM: ~20–40s per experiment
   - AGNews BERT/GPT: ~1–2 min per experiment
   - Tabular-MLP: ~5–10s
   - AudioCNN: ~30–60s
   - Total runtime (all experiments) on GPU: ~5–10 minutes
   - Total runtime on CPU: ~30–60 minutes
   - Plotting and dashboard generation: <10s

Notes:
- Large models (BERT/GPT-style) dominate runtime.
- LoRA allows efficient adaptation with fewer parameters and faster training.
- Synthetic datasets are used if original data is unavailable.
- FedAvg ensures collaborative learning across multiple clients.
"""

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
import time
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# LoRA Layer (shared)
class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r=4):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_f, in_f) * 0.01)
        self.b = nn.Parameter(torch.zeros(out_f))
        self.A = nn.Parameter(torch.randn(r, in_f) * 0.01)
        self.B = nn.Parameter(torch.randn(out_f, r) * 0.01)

    def forward(self, x):
        return x @ self.W.t() + (x @ self.A.t()) @ self.B.t() + self.b


class LoRAEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, r=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.A = nn.Parameter(torch.randn(r, vocab_size) * 0.01)
        self.B = nn.Parameter(torch.randn(embed_dim, r) * 0.01)

    def forward(self, x):
        base = self.embed(x)
        return base
    
class LoRAMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, r=4, batch_first=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=batch_first)
        self.lora_q_A = nn.Parameter(torch.randn(r, d_model) * 0.01)
        self.lora_q_B = nn.Parameter(torch.randn(d_model, r) * 0.01)
        self.lora_k_A = nn.Parameter(torch.randn(r, d_model) * 0.01)
        self.lora_k_B = nn.Parameter(torch.randn(d_model, r) * 0.01)
        self.lora_v_A = nn.Parameter(torch.randn(r, d_model) * 0.01)
        self.lora_v_B = nn.Parameter(torch.randn(d_model, r) * 0.01)

    def forward(self, query, key, value, key_padding_mask=None):
        q_delta = (query @ self.lora_q_A.t()) @ self.lora_q_B.t()
        k_delta = (key   @ self.lora_k_A.t()) @ self.lora_k_B.t()
        v_delta = (value @ self.lora_v_A.t()) @ self.lora_v_B.t()
        query = query + q_delta
        key   = key   + k_delta
        value = value + v_delta
        out, _ = self.attn(query, key, value, key_padding_mask=key_padding_mask)
        return out

# IMAGE MODELS
class CNN(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, r=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),    nn.ReLU(), nn.MaxPool2d(2)
        )
        feat = 64 * 8 * 8 if in_ch == 3 else 64 * 7 * 7
        self.fc1 = LoRALinear(feat, 128, r)
        self.fc2 = LoRALinear(128, num_classes, r)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class MLP(nn.Module):
    def __init__(self, in_dim, num_classes=10, r=4):
        super().__init__()
        self.fc1 = LoRALinear(in_dim, 128, r)
        self.fc2 = LoRALinear(128, num_classes, r)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# TEXT MODELS
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed=64, hidden=128, num_layers=2, num_classes=4, r=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed, padding_idx=0)
        self.lstm = nn.LSTM(embed, hidden, num_layers=num_layers,
                            batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.dropout = nn.Dropout(0.3)
        self.fc = LoRALinear(hidden, num_classes, r)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        h = self.dropout(h[-1])
        return self.fc(h)


class BERTStyleModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=2,
                 max_len=128, num_classes=4, r=4):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed   = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": LoRAMultiheadAttention(d_model, nhead, r=r, batch_first=True),
                "norm1": nn.LayerNorm(d_model),
                "ff1":  LoRALinear(d_model, d_model * 4, r),
                "ff2":  LoRALinear(d_model * 4, d_model, r),
                "norm2": nn.LayerNorm(d_model),
            })
            for _ in range(num_layers)
        ])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head = LoRALinear(d_model, num_classes, r)
        self.dropout = nn.Dropout(0.1)
        self.max_len = max_len

    def forward(self, x):
        B, T = x.shape
        T = min(T, self.max_len - 1)
        x = x[:, :T]
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        tok = self.token_embed(x)
        pos = self.pos_embed(positions)
        h = self.dropout(tok + pos)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        for layer in self.layers:
            attn_out = layer["attn"](h, h, h)
            h = layer["norm1"](h + attn_out)
            ff = torch.relu(layer["ff1"](h))
            ff = layer["ff2"](ff)
            h = layer["norm2"](h + ff)
        cls_rep = h[:, 0, :]
        return self.head(cls_rep)


class GPTStyleModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=2,
                 max_len=128, num_classes=4, r=4):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed   = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn":  LoRAMultiheadAttention(d_model, nhead, r=r, batch_first=True),
                "norm1": nn.LayerNorm(d_model),
                "ff1":   LoRALinear(d_model, d_model * 4, r),
                "ff2":   LoRALinear(d_model * 4, d_model, r),
                "norm2": nn.LayerNorm(d_model),
            })
            for _ in range(num_layers)
        ])
        self.head = LoRALinear(d_model, num_classes, r)
        self.dropout = nn.Dropout(0.1)
        self.max_len = max_len

    def forward(self, x):
        B, T = x.shape
        T = min(T, self.max_len)
        x = x[:, :T]
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.dropout(self.token_embed(x) + self.pos_embed(positions))
        for layer in self.layers:
            attn_out = layer["attn"](h, h, h)
            h = layer["norm1"](h + attn_out)
            ff = torch.relu(layer["ff1"](h))
            ff = layer["ff2"](ff)
            h = layer["norm2"](h + ff)
        last = h[:, -1, :]
        return self.head(last)

# TABULAR MODEL
class TabularMLP(nn.Module):
    def __init__(self, in_dim, num_classes=2, r=4):
        super().__init__()
        self.net = nn.Sequential(
            LoRALinear(in_dim, 256, r), nn.ReLU(), nn.Dropout(0.3),
            LoRALinear(256, 128, r),   nn.ReLU(), nn.Dropout(0.2),
            LoRALinear(128, 64, r),    nn.ReLU(),
        )
        self.head = LoRALinear(64, num_classes, r)

    def forward(self, x):
        return self.head(self.net(x))

# AUDIO MODEL
class AudioCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=35, r=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=80, stride=16), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),           nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.fc = LoRALinear(128 * 8, 256, r)
        self.head = LoRALinear(256, num_classes, r)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        x = self.dropout(x)
        return self.head(x)


# DATASETS

class AGNewsDataset(Dataset):
    VOCAB_SIZE = 10000
    MAX_LEN = 64

    def __init__(self, split="train"):
        super().__init__()
        self._use_synthetic = False
        try:
            from torchtext.datasets import AG_NEWS
            from torchtext.data.utils import get_tokenizer
            from torchtext.vocab import build_vocab_from_iterator

            tokenizer = get_tokenizer("basic_english")
            raw = list(AG_NEWS(split=split))

            def yield_tokens(data):
                for label, text in data:
                    yield tokenizer(text)

            vocab = build_vocab_from_iterator(
                yield_tokens(raw), specials=["<pad>", "<unk>"],
                max_tokens=self.VOCAB_SIZE
            )
            vocab.set_default_index(vocab["<unk>"])

            self.data = []
            for label, text in raw:
                tokens = tokenizer(text)[:self.MAX_LEN]
                ids = vocab(tokens)
                ids += [0] * (self.MAX_LEN - len(ids))
                self.data.append((torch.tensor(ids, dtype=torch.long), int(label) - 1))

        except Exception as e:
            print(f"[AGNews] torchtext not available ({e}). Using synthetic data.")
            self._use_synthetic = True
            rng = np.random.RandomState(42 if split == "train" else 7)
            n = 5000 if split == "train" else 1000
            self.data = [
                (torch.randint(1, self.VOCAB_SIZE, (self.MAX_LEN,)), rng.randint(0, 4))
                for _ in range(n)
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, int(y)


class TabularDataset(Dataset):
    def __init__(self, split="train"):
        super().__init__()
        try:
            import urllib.request, os
            url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                   "heart-disease/processed.cleveland.data")
            path = "./data/heart.csv"
            os.makedirs("./data", exist_ok=True)
            if not os.path.exists(path):
                urllib.request.urlretrieve(url, path)
            df = pd.read_csv(path, header=None, na_values="?").dropna()
            X = df.iloc[:, :-1].values.astype(np.float32)
            y = (df.iloc[:, -1].values > 0).astype(np.int64)
            num_classes = 2
        except Exception as e:
            print(f"[Tabular] UCI Heart Disease unavailable ({e}). Using synthetic data.")
            rng = np.random.RandomState(0)
            n = 4000 if split == "train" else 800
            centers = rng.randn(4, 20) * 2
            X_list, y_list = [], []
            for i in range(4):
                ni = n // 4
                X_list.append(rng.randn(ni, 20) + centers[i])
                y_list.append(np.full(ni, i, dtype=np.int64))
            X = np.vstack(X_list).astype(np.float32)
            y = np.concatenate(y_list)
            num_classes = 4

        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        rng = np.random.RandomState(1)
        idx = rng.permutation(len(X))
        split_pt = int(0.8 * len(X))
        if split == "train":
            idx = idx[:split_pt]
        else:
            idx = idx[split_pt:]

        self.X = torch.from_numpy(X[idx])
        self.y = torch.from_numpy(y[idx])
        self.in_dim = X.shape[1]
        self.num_classes = int(y.max()) + 1

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AudioDataset(Dataset):
    SAMPLE_RATE = 16000
    NUM_CLASSES = 35

    def __init__(self, split="train"):
        super().__init__()
        self._loaded = False
        try:
            import torchaudio
            subset = "training" if split == "train" else "validation"
            ds = torchaudio.datasets.SPEECHCOMMANDS("./data", download=True, subset=subset)
            all_labels = sorted({ds[i][2] for i in range(min(500, len(ds)))})
            self.label2idx = {l: i for i, l in enumerate(all_labels)}
            self.NUM_CLASSES = len(self.label2idx)
            self.data = ds
            self._loaded = True
        except Exception as e:
            print(f"[Audio] torchaudio SpeechCommands unavailable ({e}). Using synthetic data.")
            rng = np.random.RandomState(3 if split == "train" else 9)
            n = 3000 if split == "train" else 600
            self.synth = [
                (torch.from_numpy(rng.randn(1, self.SAMPLE_RATE).astype(np.float32)),
                 rng.randint(0, self.NUM_CLASSES))
                for _ in range(n)
            ]

    def __len__(self):
        if self._loaded:
            return len(self.data)
        return len(self.synth)

    def __getitem__(self, idx):
        if self._loaded:
            waveform, sr, label, *_ = self.data[idx]
            target = self.SAMPLE_RATE
            if waveform.shape[-1] < target:
                waveform = torch.nn.functional.pad(waveform, (0, target - waveform.shape[-1]))
            else:
                waveform = waveform[:, :target]
            return waveform, self.label2idx.get(label, 0)
        return self.synth[idx]


def train_client(model, loader, epochs, delay=0):
    model.train()
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    total = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += y.size(0)
            if delay > 0:
                time.sleep(delay)
    return model.state_dict(), total


def fedavg(weights, samples):
    total = sum(samples)
    new = {}
    for k in weights[0].keys():
        new[k] = sum(w[k] * (s / total) for w, s in zip(weights, samples))
    return new


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def split_dataset(dataset, num_clients=3):
    n = len(dataset)
    size = n // num_clients
    return [Subset(dataset, list(range(i * size, (i + 1) * size)))
            for i in range(num_clients)]


# Federated Experiment Runner
def run_experiment(name, model_fn, trainset, testloader,
                   num_clients=3, num_rounds=5,
                   client_epochs=(1, 2, 3), delays=(0.0, 0.0, 0.0)):
    clients  = split_dataset(trainset, num_clients)
    loaders  = [DataLoader(c, batch_size=64, shuffle=True) for c in clients]

    global_model = model_fn().to(device)
    acc_curve    = []
    collab_history = [] 

    for rnd in range(num_rounds):
        weights, samples = [], []
        for i in range(num_clients):
            local = model_fn().to(device)
            local.load_state_dict(global_model.state_dict())
            w, s = train_client(local, loaders[i],
                                 client_epochs[i % len(client_epochs)],
                                 delays[i % len(delays)])
            weights.append(w)
            samples.append(s)

        collab_history.append(samples[:])  # untuk save copy

        total = sum(samples)
        print(f"\n[{name}] Round {rnd+1}/{num_rounds}")
        for i, s in enumerate(samples):
            print(f"  Client {i+1}: {100*s/total:.1f}% of round samples ({s})")

        global_model.load_state_dict(fedavg(weights, samples))
        acc = evaluate(global_model, testloader)
        acc_curve.append(acc)
        print(f"  Global accuracy: {acc:.2f}%")

    return acc_curve, acc_curve[-1], collab_history

# DATASETS
print("\n=== Loading Datasets ===")

transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])
cifar_train = torchvision.datasets.CIFAR10("./data", train=True,  download=True, transform=transform_cifar)
cifar_test  = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform_cifar)
cifar_testloader = DataLoader(cifar_test, batch_size=64)

transform_fashion = transforms.Compose([transforms.ToTensor()])
fashion_train = torchvision.datasets.FashionMNIST("./data", train=True,  download=True, transform=transform_fashion)
fashion_test  = torchvision.datasets.FashionMNIST("./data", train=False, download=True, transform=transform_fashion)
fashion_testloader = DataLoader(fashion_test, batch_size=64)

print("Loading AG News...")
agnews_train = AGNewsDataset("train")
agnews_test  = AGNewsDataset("test")
agnews_testloader = DataLoader(agnews_test, batch_size=64)

print("Loading Tabular dataset...")
tabular_train = TabularDataset("train")
tabular_test  = TabularDataset("test")
tabular_testloader = DataLoader(tabular_test, batch_size=64)
TAB_IN_DIM    = tabular_train.in_dim
TAB_N_CLASSES = tabular_train.num_classes

print("Loading Audio dataset...")
audio_train = AudioDataset("train")
audio_test  = AudioDataset("test")
audio_testloader = DataLoader(audio_test, batch_size=32)
AUDIO_N_CLASSES = audio_train.NUM_CLASSES

print(f"\nDataset sizes:")
print(f"  CIFAR-10  train={len(cifar_train)}, test={len(cifar_test)}")
print(f"  Fashion   train={len(fashion_train)}, test={len(fashion_test)}")
print(f"  AG News   train={len(agnews_train)}, test={len(agnews_test)}")
print(f"  Tabular   train={len(tabular_train)}, test={len(tabular_test)}, features={TAB_IN_DIM}, classes={TAB_N_CLASSES}")
print(f"  Audio     train={len(audio_train)}, test={len(audio_test)}, classes={AUDIO_N_CLASSES}")

VOCAB = AGNewsDataset.VOCAB_SIZE


# =========================
# Di sini mulai semua
# =========================
print("\n=== Running Federated LoRA Experiments ===")

results = {}

results["CIFAR-CNN"] = run_experiment(
    "CIFAR-CNN", lambda: CNN(3, 10), cifar_train, cifar_testloader)

results["CIFAR-MLP"] = run_experiment(
    "CIFAR-MLP", lambda: MLP(32*32*3), cifar_train, cifar_testloader)

results["Fashion-CNN"] = run_experiment(
    "Fashion-CNN", lambda: CNN(1, 10), fashion_train, fashion_testloader)

results["Fashion-MLP"] = run_experiment(
    "Fashion-MLP", lambda: MLP(28*28), fashion_train, fashion_testloader)

results["AGNews-LSTM"] = run_experiment(
    "AGNews-LSTM",
    lambda: LSTMModel(VOCAB, embed=64, hidden=128, num_layers=2, num_classes=4, r=4),
    agnews_train, agnews_testloader)

results["AGNews-BERT"] = run_experiment(
    "AGNews-BERT",
    lambda: BERTStyleModel(VOCAB, d_model=64, nhead=2, num_layers=2,
                           max_len=AGNewsDataset.MAX_LEN, num_classes=4, r=4),
    agnews_train, agnews_testloader)

results["AGNews-GPT"] = run_experiment(
    "AGNews-GPT",
    lambda: GPTStyleModel(VOCAB, d_model=64, nhead=2, num_layers=2,
                          max_len=AGNewsDataset.MAX_LEN, num_classes=4, r=4),
    agnews_train, agnews_testloader)

results["Tabular-MLP"] = run_experiment(
    "Tabular-MLP",
    lambda: TabularMLP(TAB_IN_DIM, TAB_N_CLASSES, r=4),
    tabular_train, tabular_testloader)

results["Audio-1DCNN"] = run_experiment(
    "Audio-1DCNN",
    lambda: AudioCNN(in_channels=1, num_classes=AUDIO_N_CLASSES, r=4),
    audio_train, audio_testloader)

# ORIGINAL ACCURACY VISUALISATIONS

groups = {
    "Image":   ["CIFAR-CNN",   "CIFAR-MLP",   "Fashion-CNN", "Fashion-MLP"],
    "Text":    ["AGNews-LSTM", "AGNews-BERT",  "AGNews-GPT"],
    "Tabular": ["Tabular-MLP"],
    "Audio":   ["Audio-1DCNN"],
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Federated LoRA: Accuracy over Rounds", fontsize=16, fontweight="bold")

colors = plt.cm.tab10.colors

for ax, (group_name, keys) in zip(axes.flat, groups.items()):
    for i, k in enumerate(keys):
        if k in results:
            acc_curve = results[k][0]
            ax.plot(range(1, len(acc_curve) + 1), acc_curve,
                    marker="o", label=k, color=colors[i % len(colors)])
    ax.set_title(group_name)
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("federated_lora_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: federated_lora_results.png")

# Final accuracy bar chart
plt.figure(figsize=(12, 6))
models     = list(results.keys())
final_accs = [results[m][1] for m in models]
plt.bar(models, final_accs)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Final Accuracy (%)")
plt.title("Final Model Comparison (All Models)")
for i, v in enumerate(final_accs):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center')
plt.tight_layout()
plt.savefig("model_comparison_bar.png", dpi=150)
plt.show()
print("Saved: model_comparison_bar.png")

# SUMMARY TABLE

table_data = {
    "Experiment":    list(results.keys()),
    "Modality":      [],
    "Architecture":  [],
    "Final Acc (%)": [round(results[k][1], 2) for k in results],
}

modality_map = {
    "CIFAR":   "Image",   "Fashion": "Image",
    "AGNews":  "Text",    "Tabular": "Tabular", "Audio": "Audio",
}
arch_map = {
    "CNN": "CNN", "MLP": "MLP", "LSTM": "LSTM",
    "BERT": "BERT-style Transformer", "GPT": "GPT-style Transformer",
    "1DCNN": "1D CNN",
}

for k in results.keys():
    for prefix, mod in modality_map.items():
        if prefix in k:
            table_data["Modality"].append(mod)
            break
    else:
        table_data["Modality"].append("Unknown")

    for suffix, arch in arch_map.items():
        if suffix in k:
            table_data["Architecture"].append(arch)
            break
    else:
        table_data["Architecture"].append("Unknown")

df = pd.DataFrame(table_data)
df = df.sort_values(["Modality", "Final Acc (%)"], ascending=[True, False])

print("\n" + "="*65)
print("FEDERATED LORA — FINAL ACCURACY SUMMARY")
print("="*65)
print(df.to_string(index=False))
print("="*65)
df.to_csv("federated_lora_summary.csv", index=False)
print("Saved: federated_lora_summary.csv")

# SECTION 3: PER-CLIENT COLLABORATION CHARTS
CLIENT_COLORS  = ["#378ADD", "#1D9E75", "#EF9F27"]
CLIENT_EPOCHS  = (1, 2, 3)          # epochs assigned per client in run_experiment
NUM_CLIENTS    = 3
NUM_ROUNDS     = 5
EXP_NAMES      = list(results.keys())


# Helper: extract actual per-round collaboration data 
def get_collab(name):
    """
    Returns collab_history: list of NUM_ROUNDS lists,
    each inner list has NUM_CLIENTS sample counts.
    """
    return results[name][2]   


# CHART A: Stacked contribution % per round 
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle(
    "Per-Client Contribution % per Federation Round\n"
    "(share of round samples used in FedAvg aggregation)",
    fontsize=15, fontweight="bold", y=1.01
)

for ax, name in zip(axes.flat, EXP_NAMES):
    collab = get_collab(name)        
    x      = np.arange(1, NUM_ROUNDS + 1)
    bottoms = np.zeros(NUM_ROUNDS)

    for ci in range(NUM_CLIENTS):
        round_samples = np.array([collab[r][ci] for r in range(NUM_ROUNDS)], dtype=float)
        round_totals  = np.array([sum(collab[r]) for r in range(NUM_ROUNDS)], dtype=float)
        pcts          = 100.0 * round_samples / round_totals

        label = f"Client {ci+1} ({CLIENT_EPOCHS[ci]} ep)"
        bars  = ax.bar(x, pcts, bottom=bottoms,
                       color=CLIENT_COLORS[ci], alpha=0.88,
                       label=label, width=0.6, edgecolor="white", linewidth=0.5)

        for r_idx, (pct, bot) in enumerate(zip(pcts, bottoms)):
            if pct > 4:   # only label if segment is tall enough to read
                ax.text(r_idx + 1, bot + pct / 2,
                        f"{pct:.1f}%",
                        ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        bottoms = bottoms + pcts

    ax.set_title(name, fontsize=10, fontweight="bold")
    ax.set_xlabel("Round", fontsize=8)
    ax.set_ylabel("Contribution (%)", fontsize=8)
    ax.set_xticks(x)
    ax.set_ylim(0, 108)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.7)

# Hide any unused subplot (9 experiments fit exactly in 3×3 so none should be hidden,
# but guard just in case)
for ax in axes.flat[len(EXP_NAMES):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig("client_contribution_pct.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: client_contribution_pct.png")

# CHART B: FedAvg aggregation
fig, axes = plt.subplots(3, 3, figsize=(14, 13))
fig.suptitle(
    "FedAvg Aggregation Weight per Client\n"
    "(average across all rounds — determines influence on global model)",
    fontsize=15, fontweight="bold", y=1.01
)

for ax, name in zip(axes.flat, EXP_NAMES):
    collab = get_collab(name)

    avg_weights = []
    for ci in range(NUM_CLIENTS):
        client_total = sum(collab[r][ci] for r in range(NUM_ROUNDS))
        grand_total  = sum(sum(collab[r]) for r in range(NUM_ROUNDS))
        avg_weights.append(client_total / grand_total)

    pie_labels = [
        f"Client {ci+1}\n({CLIENT_EPOCHS[ci]} ep)\n{avg_weights[ci]*100:.1f}%"
        for ci in range(NUM_CLIENTS)
    ]
    wedges, _ = ax.pie(
        avg_weights,
        colors=CLIENT_COLORS,
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.5)
    )
    ax.set_title(name, fontsize=9, fontweight="bold", pad=6)
    ax.legend(
        wedges, pie_labels,
        fontsize=7, loc="lower center",
        bbox_to_anchor=(0.5, -0.30), ncol=1,
        framealpha=0.6
    )

for ax in axes.flat[len(EXP_NAMES):]:
    ax.set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("client_aggregation_weights.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: client_aggregation_weights.png")

# CHART C: Cumulative samples per client per round
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle(
    "Cumulative Samples per Client across Federation Rounds\n"
    "(running total of training examples seen by each client)",
    fontsize=15, fontweight="bold", y=1.01
)

for ax, name in zip(axes.flat, EXP_NAMES):
    collab = get_collab(name)
    x      = np.arange(1, NUM_ROUNDS + 1)

    for ci in range(NUM_CLIENTS):
        round_samples = [collab[r][ci] for r in range(NUM_ROUNDS)]
        cumulative    = np.cumsum(round_samples)
        label         = f"Client {ci+1} ({CLIENT_EPOCHS[ci]} ep)"
        ax.plot(x, cumulative,
                marker="o", color=CLIENT_COLORS[ci],
                label=label, linewidth=2, markersize=5)
        ax.fill_between(x, cumulative, alpha=0.10, color=CLIENT_COLORS[ci])

        ax.annotate(
            f"{int(cumulative[-1]):,}",
            xy=(NUM_ROUNDS, cumulative[-1]),
            xytext=(5, 0), textcoords="offset points",
            fontsize=7, color=CLIENT_COLORS[ci], fontweight="bold"
        )

    ax.set_title(name, fontsize=10, fontweight="bold")
    ax.set_xlabel("Round", fontsize=8)
    ax.set_ylabel("Cumulative samples", fontsize=8)
    ax.set_xticks(x)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v):,}")
    )
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=7)

for ax in axes.flat[len(EXP_NAMES):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig("client_cumulative_samples.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: client_cumulative_samples.png")

# CHART D: Per-client contribution % over rounds 
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle(
    "Per-Client Contribution % over Rounds (line view)\n"
    "(how each client's share of samples evolves each round)",
    fontsize=15, fontweight="bold", y=1.01
)

for ax, name in zip(axes.flat, EXP_NAMES):
    collab = get_collab(name)
    x      = np.arange(1, NUM_ROUNDS + 1)

    for ci in range(NUM_CLIENTS):
        round_samples = np.array([collab[r][ci] for r in range(NUM_ROUNDS)], dtype=float)
        round_totals  = np.array([sum(collab[r]) for r in range(NUM_ROUNDS)], dtype=float)
        pcts          = 100.0 * round_samples / round_totals
        label         = f"Client {ci+1} ({CLIENT_EPOCHS[ci]} ep)"

        ax.plot(x, pcts,
                marker="o", color=CLIENT_COLORS[ci],
                label=label, linewidth=2, markersize=5)
        ax.fill_between(x, pcts, alpha=0.08, color=CLIENT_COLORS[ci])

    ax.set_title(name, fontsize=10, fontweight="bold")
    ax.set_xlabel("Round", fontsize=8)
    ax.set_ylabel("Contribution (%)", fontsize=8)
    ax.set_xticks(x)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=7)

    ax.axhline(100 / NUM_CLIENTS, color="gray",
               linestyle=":", linewidth=1, alpha=0.6,
               label=f"Equal split ({100/NUM_CLIENTS:.1f}%)")

for ax in axes.flat[len(EXP_NAMES):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig("client_contribution_line.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: client_contribution_line.png")

# CHART E: clients × rounds per experiment
fig, axes = plt.subplots(3, 3, figsize=(18, 13))
fig.suptitle(
    "Aggregation Weight Heatmap — Client × Round\n"
    "(colour = FedAvg weight; darker = more influence on global model)",
    fontsize=15, fontweight="bold", y=1.01
)

cmap = plt.cm.YlOrBr   

for ax, name in zip(axes.flat, EXP_NAMES):
    collab = get_collab(name)

    matrix = np.zeros((NUM_CLIENTS, NUM_ROUNDS))
    for r in range(NUM_ROUNDS):
        total = sum(collab[r])
        for ci in range(NUM_CLIENTS):
            matrix[ci, r] = collab[r][ci] / total

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(name, fontsize=9, fontweight="bold")
    ax.set_xlabel("Round", fontsize=8)
    ax.set_ylabel("Client", fontsize=8)
    ax.set_xticks(range(NUM_ROUNDS))
    ax.set_xticklabels([f"R{r+1}" for r in range(NUM_ROUNDS)], fontsize=7)
    ax.set_yticks(range(NUM_CLIENTS))
    ax.set_yticklabels([f"C{ci+1}\n({CLIENT_EPOCHS[ci]}ep)" for ci in range(NUM_CLIENTS)], fontsize=7)

    for ci in range(NUM_CLIENTS):
        for r in range(NUM_ROUNDS):
            val = matrix[ci, r]
            text_color = "white" if val > 0.5 else "black"
            ax.text(r, ci, f"{val*100:.1f}%",
                    ha="center", va="center",
                    fontsize=7.5, color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 format=lambda x, _: f"{x*100:.0f}%")

for ax in axes.flat[len(EXP_NAMES):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig("client_weight_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: client_weight_heatmap.png")

# CHART F: accuracy curve + contribution bars
fig = plt.figure(figsize=(22, 20))
fig.suptitle(
    "Federated LoRA — Accuracy & Client Collaboration Dashboard",
    fontsize=16, fontweight="bold", y=1.01
)

outer = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

for exp_idx, name in enumerate(EXP_NAMES):
    row, col = divmod(exp_idx, 3)
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[row, col], hspace=0.45, height_ratios=[1, 1]
    )

    ax_acc = fig.add_subplot(inner[0])
    acc_curve = results[name][0]
    x = range(1, NUM_ROUNDS + 1)
    ax_acc.plot(x, acc_curve, marker="o", color="#534AB7",
                linewidth=2, markersize=5)
    ax_acc.fill_between(x, acc_curve, alpha=0.10, color="#534AB7")
    ax_acc.set_title(name, fontsize=9, fontweight="bold")
    ax_acc.set_xlabel("Round", fontsize=7)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=7)
    ax_acc.set_xticks(list(x))
    ax_acc.tick_params(labelsize=7)
    ax_acc.grid(alpha=0.25, linestyle="--")
    ax_acc.annotate(f"{acc_curve[-1]:.1f}%",
                    xy=(NUM_ROUNDS, acc_curve[-1]),
                    xytext=(-18, 6), textcoords="offset points",
                    fontsize=7, color="#534AB7", fontweight="bold")

    ax_bar = fig.add_subplot(inner[1])
    collab  = get_collab(name)
    x_arr   = np.arange(1, NUM_ROUNDS + 1)
    bottoms = np.zeros(NUM_ROUNDS)

    for ci in range(NUM_CLIENTS):
        round_samples = np.array([collab[r][ci] for r in range(NUM_ROUNDS)], dtype=float)
        round_totals  = np.array([sum(collab[r]) for r in range(NUM_ROUNDS)], dtype=float)
        pcts          = 100.0 * round_samples / round_totals
        ax_bar.bar(x_arr, pcts, bottom=bottoms,
                   color=CLIENT_COLORS[ci], alpha=0.88,
                   label=f"C{ci+1}", width=0.6,
                   edgecolor="white", linewidth=0.4)
        for r_idx, (pct, bot) in enumerate(zip(pcts, bottoms)):
            if pct > 6:
                ax_bar.text(r_idx + 1, bot + pct / 2,
                            f"{pct:.0f}%",
                            ha="center", va="center",
                            fontsize=6.5, color="white", fontweight="bold")
        bottoms = bottoms + pcts

    ax_bar.set_xlabel("Round", fontsize=7)
    ax_bar.set_ylabel("Contribution (%)", fontsize=7)
    ax_bar.set_xticks(x_arr)
    ax_bar.set_ylim(0, 108)
    ax_bar.tick_params(labelsize=7)
    ax_bar.grid(axis="y", alpha=0.2, linestyle="--")
    ax_bar.legend(fontsize=6, loc="upper right", framealpha=0.6, ncol=3)

plt.savefig("collaboration_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: collaboration_dashboard.png")

# FINAL SUMMARY
print("\n" + "="*65)
print("ALL OUTPUTS SAVED")
print("="*65)
print("  federated_lora_results.png        — accuracy curves by modality")
print("  model_comparison_bar.png          — final accuracy bar chart")
print("  federated_lora_summary.csv        — summary table")
print("  client_contribution_pct.png       — stacked contribution % bars")
print("  client_aggregation_weights.png    — FedAvg weight doughnuts")
print("  client_cumulative_samples.png     — cumulative sample lines")
print("  client_contribution_line.png      — contribution % line view")
print("  client_weight_heatmap.png         — weight heatmap (client × round)")
print("  collaboration_dashboard.png       — accuracy + contribution combined")
print("="*65)
