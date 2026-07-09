"""
EfficientPIE variant that combines image features with normalized bbox context.
"""
import torch
import torch.nn as nn
from torch import Tensor

from models.EfficientPIE_baseline import EfficientPIE


class EfficientPIEBBox(EfficientPIE):
    def __init__(self, num_classes: int = 2, bbox_feature_dim: int = 4):
        super(EfficientPIEBBox, self).__init__(num_classes=num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(1280 + bbox_feature_dim, 256),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Tensor, bbox_features: Tensor) -> Tensor:
        x = self.commonConv(x)
        x = self.fm1(x)
        x = self.fm2(x)
        x = self.mb1(x)
        x = self.mb2(x)
        x = self.commonConv1(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = torch.cat([x, bbox_features], dim=1)
        return self.classifier(x)
