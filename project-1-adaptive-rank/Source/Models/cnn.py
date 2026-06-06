# CNN and AudioCNN model definitions

import torch
import torch.nn as nn
from .lora import LoRALinear


class CNN(nn.Module):
    """
    Convolutional network for image classification (CIFAR-10, etc.).
    LoRA applied to fully-connected layers.
    """
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


class AudioCNN(nn.Module):
    """
    1-D convolutional network for audio / speech command classification.
    LoRA applied to fully-connected layers.
    """
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
