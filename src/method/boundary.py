import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class BoundaryArtifactDetector(nn.Module):
    def __init__(self, embed_dim: int = 384, evidence_dim: int = 128):
        super().__init__()

        # Pre-register Sobel kernels as buffers (avoids re-creating every forward pass)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        self.edge_encoder = nn.Sequential(
            ConvBlock(1, 32, 7, stride=2, padding=3),
            ConvBlock(32, 64, 3, stride=2, padding=1),
            ResidualBlock(64),
            SpatialAttention(64),
            ConvBlock(64, 128, 3, stride=2, padding=1),
            ResidualBlock(128),
            ConvBlock(128, 256, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )

        self.patch_processor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Linear(256 + embed_dim // 2, evidence_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(evidence_dim * 2, evidence_dim),
        )

        self.classifier = nn.Linear(evidence_dim, 1)

        self.apply(_init_weights)

    def forward(self, image: torch.Tensor, patch_features: torch.Tensor,
                sobel_cached: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        if sobel_cached is not None:
            if sobel_cached.ndim != 4 or sobel_cached.size(1) != 1:
                sobel_cached = None

        if sobel_cached is None:
            if image.ndim != 4:
                raise ValueError(f"Expected 4D image tensor, got shape {image.shape}")
            if image.size(1) not in [1, 3]:
                raise ValueError(f"Expected image with 1 or 3 channels, got {image.size(1)} channels. Full shape: {image.shape}")

            if image.size(1) == 3:
                gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
                gray = gray.unsqueeze(1)
            else:
                gray = image

            grad_x = F.conv2d(gray, self.sobel_x, padding=1)
            grad_y = F.conv2d(gray, self.sobel_y, padding=1)
            edges = torch.sqrt(grad_x**2 + grad_y**2)

            max_vals = edges.amax(dim=(1, 2, 3), keepdim=True)  # (B, 1, 1, 1)
            edges = edges / max_vals.clamp_min(1e-8)
        else:
            if sobel_cached.ndim == 3:
                edges = sobel_cached.unsqueeze(1)
            elif sobel_cached.ndim == 4 and sobel_cached.size(1) == 1:
                edges = sobel_cached
            else:
                edges = None

        if edges is None or edges.ndim != 4 or edges.size(1) != 1:
            raise ValueError(f"Failed to create valid edges tensor. Shape: {edges.shape if edges is not None else 'None'}")

        edge_feat = self.edge_encoder(edges).flatten(1)

        patch_mean = patch_features.mean(dim=1)
        patch_max = patch_features.max(dim=1)[0]
        patch_aggregated = torch.cat([patch_mean, patch_max], dim=1)
        patch_feat = self.patch_processor(patch_aggregated)

        fused = torch.cat([edge_feat, patch_feat], dim=1)
        evidence = self.fusion(fused)
        logit = self.classifier(evidence)

        return {"evidence": evidence, "logit": logit, "score": torch.sigmoid(logit)}
