import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ----------------------------
# Pre-head (keep as-is)
# ----------------------------
class CNNPatchDownscaleHead(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=5, stride=2, padding=2),
        )

    def forward(self, x):
        x = self.features(x)
        x = F.interpolate(x, size=(224, 224), mode="bilinear")
        return x


# ----------------------------
# ShuffleNetV2 with custom head + classifier
# ----------------------------
class ShuffleNetV2WithCustomHead(nn.Module):
    def __init__(self, num_classes=5, dropout_p=0.5):
        super().__init__()
        self.head = CNNPatchDownscaleHead(in_channels=3, out_channels=3)

        # Load pretrained ShuffleNetV2
        backbone = models.shufflenet_v2_x1_0(
            weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
        )

        # Proper encoder: keep all conv/stage layers and add AdaptiveAvgPool2d
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.maxpool,
            backbone.stage2,
            backbone.stage3,
            backbone.stage4,
            backbone.conv5,
            nn.AdaptiveAvgPool2d(1),  # now output will be [B, 1024, 1, 1]
        )

        # Classifier (flatten and project to num_classes)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.head(x)  # [B, 3, 224, 224]
        x = self.encoder(x)  # [B, 1024, 1, 1]
        x = self.classifier(x)  # [B, num_classes]
        return x
