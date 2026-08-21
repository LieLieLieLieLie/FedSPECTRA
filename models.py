from __future__ import annotations

from runtime_compat import bootstrap

bootstrap()

import torch
from torch import nn
import torch.nn.functional as F


class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, stride=(1, 1)):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )


class SpectralEncoder(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 96, bands: int = 8):
        super().__init__()
        self.bands = bands
        self.stem = ConvNormAct(1, 24, (1, 1))
        self.block1 = nn.Sequential(ConvNormAct(24, 40, (2, 2)), ConvNormAct(40, 40))
        self.block2 = nn.Sequential(ConvNormAct(40, 64, (2, 2)), ConvNormAct(64, 64))
        self.block3 = nn.Sequential(ConvNormAct(64, 96, (2, 2)), ConvNormAct(96, 96))
        self.projection = nn.Sequential(
            nn.Linear(96, embedding_dim), nn.LayerNorm(embedding_dim), nn.SiLU(), nn.Dropout(0.1)
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.band_attention = nn.Sequential(
            nn.Conv2d(40, 16, 1, bias=False), nn.SiLU(), nn.Conv2d(16, 1, 1)
        )

    def forward(self, x: torch.Tensor):
        h = self.stem(x)
        h1 = self.block1(h)
        band_map = self.band_attention(h1)
        band_logits = F.adaptive_avg_pool2d(band_map, (self.bands, 1)).flatten(1)
        spectral = F.softmax(band_logits, dim=1)
        h = self.block2(h1)
        h = self.block3(h)
        pooled = F.adaptive_avg_pool2d(h, 1).flatten(1)
        embedding = self.projection(pooled)
        logits = self.classifier(embedding)
        return logits, embedding, spectral


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def flatten_parameters(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


@torch.no_grad()
def assign_flat_parameters(model: nn.Module, vector: torch.Tensor) -> None:
    offset = 0
    for param in model.parameters():
        count = param.numel()
        param.copy_(vector[offset:offset + count].view_as(param))
        offset += count
    if offset != vector.numel():
        raise ValueError("Flat vector length does not match model parameters")

