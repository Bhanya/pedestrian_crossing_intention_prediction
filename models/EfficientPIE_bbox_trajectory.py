"""
EfficientPIE variant that fuses image features with a 15-frame bbox trajectory.
"""
import torch
import torch.nn as nn
from torch import Tensor

from models.EfficientPIE_baseline import EfficientPIE


class EfficientPIEBBoxTrajectory(EfficientPIE):
    def __init__(
            self,
            num_classes: int = 2,
            bbox_feature_dim: int = 4,
            bbox_hidden_dim: int = 64):
        super(EfficientPIEBBoxTrajectory, self).__init__(num_classes=num_classes)
        self.bbox_gru = nn.GRU(     # GRU (Gated Recurrent Unit) reads the 15-frame bbox sequence.
            input_size=bbox_feature_dim,
            hidden_size=bbox_hidden_dim,
            batch_first=True,
        )   
        self.bbox_projection = nn.Sequential(       # Convert GRU output into bbox feature
            nn.Linear(bbox_hidden_dim, 128),
            nn.SiLU(),
            nn.Dropout(p=0.2),
        )
        self.classifier = nn.Sequential(        # classifier receives image feature + bbox motion feature.
            nn.Linear(1280 + 128, 256),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Tensor, bbox_trajectory: Tensor) -> Tensor:        # Foward pass
        x = self.commonConv(x)
        x = self.fm1(x)
        x = self.fm2(x)
        x = self.mb1(x)
        x = self.mb2(x)
        x = self.commonConv1(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)

        _, hidden = self.bbox_gru(bbox_trajectory)
        bbox_feature = self.bbox_projection(hidden[-1])
        x = torch.cat([x, bbox_feature], dim=1)
        return self.classifier(x)
