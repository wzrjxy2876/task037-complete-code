"""Tiny deterministic adapter used only for end-to-end smoke validation."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset


class Mlp(torch.nn.Module):
    def __init__(self, hidden_features: int = 12, output_features: int = 8):
        super().__init__()
        self.original_hidden_features = hidden_features
        self.fc2 = torch.nn.Linear(hidden_features, output_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(x))


class SmokeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = Mlp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SmokeProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Linear(3, 12)
        self.block = SmokeBlock()
        self.head = torch.nn.Linear(8, 3)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        features = video.permute(0, 2, 3, 4, 1)
        features = self.stem(features)
        features = self.block(features)
        return self.head(features.mean(dim=(1, 2, 3)))


def build_model_for_probe(
    device: torch.device | str,
    seed: int = 3407,
    **_: Any,
) -> torch.nn.Module:
    torch.manual_seed(int(seed))
    return SmokeProbeModel().to(device).eval()


def build_probe_loader(
    videos: int = 4,
    batch_size: int = 2,
    seed: int = 3407,
    **_: Any,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    tensors = torch.randn(int(videos), 3, 4, 8, 8, generator=generator)
    labels = torch.arange(int(videos), dtype=torch.long) % 3
    indices = torch.arange(1000, 1000 + int(videos), dtype=torch.long)
    return DataLoader(
        TensorDataset(tensors, labels, indices),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
    )
