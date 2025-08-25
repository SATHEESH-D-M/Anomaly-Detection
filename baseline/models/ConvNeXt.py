import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging


# ----------------------------
# Pre-head
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
        x = F.interpolate(
            x, size=(224, 224), mode="bilinear"
        )  # ConvNeXt input size
        return x


# ----------------------------
# ConvNeXt with custom head + classifier
# ----------------------------
class ConvNeXtWithCustomHead(nn.Module):
    def __init__(self, num_classes=5, dropout_p=0.5):
        super().__init__()
        # Patch pre-head
        self.head = CNNPatchDownscaleHead(in_channels=3, out_channels=3)

        # Load pretrained ConvNeXt-Tiny
        backbone = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )

        # Keep only encoder (no classifier)
        self.encoder = nn.Sequential(
            backbone.features,  # Feature extractor
            backbone.avgpool,  # Global avg pool -> [B, 768, 1, 1]
        )

        # Custom classifier head (same as your ResNet18 setup, but input dim = 768 for ConvNeXt-Tiny)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, 128),
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
        x = self.encoder(x)  # [B, 768, 1, 1]
        x = self.classifier(x)  # [B, num_classes]
        return x
