"""Stage 1 model: timm backbone + linear head for binary occlusion classification."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class FrameClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.2,
        num_classes: int = 2,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Expose pooled features (used by later temporal stage)."""
        return self.backbone(x)
