import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


RGB_TO_GRAY_WEIGHTS = (0.299, 0.587, 0.114)


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def compute_frequency_spectrum(image: torch.Tensor, target_size: int = 112) -> torch.Tensor:
    gray = RGB_TO_GRAY_WEIGHTS[0] * image[:, 0] + RGB_TO_GRAY_WEIGHTS[1] * image[:, 1] + RGB_TO_GRAY_WEIGHTS[2] * image[:, 2]

    if gray.shape[-1] != target_size:
        gray = F.interpolate(gray.unsqueeze(1), size=(target_size, target_size),
                            mode='bilinear', align_corners=False).squeeze(1)

    fft = torch.fft.fft2(gray)
    fft_shifted = torch.fft.fftshift(fft)
    magnitude = torch.abs(fft_shifted)

    H, W = magnitude.shape[-2:]
    center_y, center_x = H // 2, W // 2
    y, x = torch.meshgrid(torch.arange(H, device=gray.device),
                         torch.arange(W, device=gray.device), indexing='ij')
    dist = torch.sqrt(((x - center_x).float())**2 + ((y - center_y).float())**2)

    cutoffs = [min(H, W) / 16, min(H, W) / 8, min(H, W) / 4]
    bands = []
    for cutoff in cutoffs:
        high_pass = torch.sigmoid((dist - cutoff) / 10)
        band = magnitude * high_pass.unsqueeze(0)
        band = torch.log1p(band)
        min_vals = band.amin(dim=(1, 2), keepdim=True)
        max_vals = band.amax(dim=(1, 2), keepdim=True)
        denom = (max_vals - min_vals).clamp_min(1e-8)
        band = (band - min_vals) / denom
        bands.append(band)

    return torch.stack(bands, dim=1)


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


class FrequencyArtifactDetector(nn.Module):
    def __init__(self, embed_dim: int = 384, evidence_dim: int = 128, fft_size: int = 112):
        super().__init__()
        self.fft_size = fft_size

        self.freq_encoder = nn.Sequential(
            ConvBlock(3, 48, 7, stride=2, padding=3),
            ConvBlock(48, 96, 3, stride=2, padding=1),
            ResidualBlock(96),
            ConvBlock(96, 192, 3, stride=2, padding=1),
            ResidualBlock(192),
            ConvBlock(192, 256, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )

        self.cls_processor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.fusion = nn.Sequential(
            nn.Linear(256 + embed_dim // 2, evidence_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(evidence_dim * 2, evidence_dim),
        )

        self.classifier = nn.Linear(evidence_dim, 1)

        self.apply(_init_weights)

    def forward(self, raw_image: torch.Tensor, cls_token: torch.Tensor,
                freq_cached: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        if freq_cached is not None and (freq_cached.dim() not in [3, 4] or freq_cached.size(1) > 3):
            freq_cached = None

        if freq_cached is None:
            freq_cached = compute_frequency_spectrum(raw_image, target_size=self.fft_size)

        magnitude = freq_cached
        if magnitude.dim() == 3:
            magnitude = magnitude.unsqueeze(1)

        if magnitude.shape[1] == 1:
            magnitude = magnitude.expand(-1, 3, -1, -1)

        if magnitude.shape[-1] != self.fft_size:
            magnitude = F.interpolate(magnitude, size=(self.fft_size, self.fft_size),
                                    mode="bilinear", align_corners=False)

        freq_feat = self.freq_encoder(magnitude).flatten(1)
        cls_feat = self.cls_processor(cls_token)

        fused = torch.cat([freq_feat, cls_feat], dim=1)
        evidence = self.fusion(fused)
        logit = self.classifier(evidence)

        return {"evidence": evidence, "logit": logit, "score": torch.sigmoid(logit)}
