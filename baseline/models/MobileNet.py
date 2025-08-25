import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging


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
        )  # Resize to match backbone
        return x


class MobileNetV3SmallWithCustomHead(nn.Module):
    def __init__(self, num_classes=5, dropout_p=0.5):
        super().__init__()
        # Patch pre-head
        self.head = CNNPatchDownscaleHead(in_channels=3, out_channels=3)

        # Load pretrained MobileNetV3 Small
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        self.encoder = backbone.features  # convolutional feature extractor

        # Output of MobileNetV3 Small features is 576 channels
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 576, 1, 1]

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 128),  # <-- 576 instead of 512
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
        x = self.encoder(x)  # [B, 576, H, W]
        x = self.pool(x)  # [B, 576, 1, 1]
        x = self.classifier(x)  # [B, num_classes]
        return x
