import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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


class CosineClassifier(nn.Module):
    def __init__(self, in_features, num_classes, scale=30.0):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # Normalize features and weights
        x_norm = F.normalize(x, p=2, dim=1)  # [B, in_features]
        w_norm = F.normalize(
            self.weight, p=2, dim=1
        )  # [num_classes, in_features]
        # Cosine similarity [B, num_classes]
        cos_theta = torch.matmul(x_norm, w_norm.t())
        return self.scale * cos_theta


class MobileNetV3SmallWithCosineHead(nn.Module):
    def __init__(self, num_classes=5, dropout_p=0.5, scale=30.0):
        super().__init__()
        # Patch pre-head
        self.head = CNNPatchDownscaleHead(in_channels=3, out_channels=3)

        # Pretrained MobileNetV3 Small
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        self.encoder = backbone.features  # conv features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 576, 1, 1]

        # Projector + Cosine Classifier in one Sequential
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            CosineClassifier(
                in_features=128, num_classes=num_classes, scale=scale
            ),
        )

    def forward(self, x):
        x = self.head(x)  # [B, 3, 224, 224]
        x = self.encoder(x)  # [B, 576, H, W]
        x = self.pool(x)  # [B, 576, 1, 1]
        logits = self.classifier(x)  # projector + cosine in one block
        return logits
