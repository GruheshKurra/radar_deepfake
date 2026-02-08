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
        if sobel_cached is None:
            if image.size(1) != 3:
                raise ValueError(f"Expected image with 3 channels, got {image.size(1)} channels. Shape: {image.shape}")

            gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
            gray = gray.unsqueeze(1)

            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                  dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                  dtype=image.dtype, device=image.device).view(1, 1, 3, 3)

            grad_x = F.conv2d(gray, sobel_x, padding=1)
            grad_y = F.conv2d(gray, sobel_y, padding=1)
            sobel_cached = torch.sqrt(grad_x**2 + grad_y**2)

            max_vals = sobel_cached.flatten(1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
            sobel_cached = sobel_cached / max_vals.clamp_min(1e-8)

        edges = sobel_cached.unsqueeze(1) if sobel_cached.dim() == 3 else sobel_cached

        if edges.size(1) != 1:
            raise ValueError(f"Expected edges with 1 channel, got {edges.size(1)} channels. "
                           f"sobel_cached shape: {sobel_cached.shape}, edges shape: {edges.shape}")

        edge_feat = self.edge_encoder(edges).flatten(1)

        patch_mean = patch_features.mean(dim=1)
        patch_max = patch_features.max(dim=1)[0]
        patch_aggregated = torch.cat([patch_mean, patch_max], dim=1)
        patch_feat = self.patch_processor(patch_aggregated)

        fused = torch.cat([edge_feat, patch_feat], dim=1)
        evidence = self.fusion(fused)
        logit = self.classifier(evidence)

        return {"evidence": evidence, "logit": logit, "score": torch.sigmoid(logit)}
