"""
Multi-Modality PyTorch Model Suite
----------------------------------
This script provides five standard PyTorch model architectures designed for 
distinct data modalities. These models serve as clean baselines, removing 
all LoRA (Low-Rank Adaptation) and Federated Learning overhead.

Included Models:
1. CNN: A 2D Convolutional Neural Network for image processing.
2. ImageMLP: A Multi-Layer Perceptron for flattened image data.
3. TextTransformer: A BERT-style Encoder-only Transformer for sequence classification.
4. TabularMLP: A deep MLP optimized for structured/tabular feature sets.
5. AudioCNN: A 1D Convolutional Neural Network for raw audio waveform analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. IMAGE MODEL: CNN
# ==========================================
class CNNModel(nn.Module):
    def __init__(self, in_ch=3, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),    nn.ReLU(), nn.MaxPool2d(2)
        )
        # Dynamic flattening based on input channel (3 for CIFAR, 1 for FashionMNIST)
        feat_dim = 64 * 8 * 8 if in_ch == 3 else 64 * 7 * 7
        self.fc1 = nn.Linear(feat_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ==========================================
# 2. IMAGE MODEL: MLP
# ==========================================
class ImageMLPModel(nn.Module):
    def __init__(self, in_dim, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ==========================================
# 3. TEXT MODEL: TRANSFORMER (BERT-style)
# ==========================================
class TextTransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=2, max_len=128, num_classes=4):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed   = nn.Embedding(max_len, d_model)
        
        # Standard PyTorch Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        
        h = self.token_embed(x) + self.pos_embed(positions)
        
        # Prepend CLS token for classification
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        
        h = self.transformer(h)
        cls_rep = h[:, 0, :] # Use the representation of the CLS token
        return self.head(cls_rep)

# ==========================================
# 4. TABULAR MODEL: DEEP MLP
# ==========================================
class TabularMLPModel(nn.Module):
    def __init__(self, in_dim, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),   nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),    nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 5. AUDIO MODEL: 1D-CNN
# ==========================================
class AudioCNNModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=35):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=80, stride=16), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),           nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),          nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.fc = nn.Linear(128 * 8, 256)
        self.head = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        return self.head(x)
