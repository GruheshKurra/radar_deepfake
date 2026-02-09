import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from dataclasses import dataclass


@dataclass
class EnhancedLossConfig:
    lambda_main: float = 1.0
    lambda_branch: float = 0.3
    lambda_orthogonal: float = 0.5
    lambda_deep_supervision: float = 0.05
    lambda_edge_alignment: float = 0.2
    lambda_freq_alignment: float = 0.2
    label_smoothing: float = 0.05
    orthogonality_margin: float = 0.05


class EnhancedRADARLoss(nn.Module):
    def __init__(self, config: EnhancedLossConfig):
        super().__init__()
        self.config = config
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def compute_edge_map(self, images):
        gray = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
        gray = gray.unsqueeze(1)

        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        edges = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

        edges = edges / (edges.amax(dim=(1,2,3), keepdim=True) + 1e-8)
        return edges

    def compute_freq_map(self, images):
        fft = torch.fft.fft2(images)
        magnitude = torch.abs(fft)

        h, w = magnitude.shape[-2:]
        magnitude[:, :, :h//8, :w//8] = 0

        magnitude = magnitude / (magnitude.amax(dim=(1,2,3), keepdim=True) + 1e-8)
        return magnitude.mean(dim=1, keepdim=True)

    def forward(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor,
                images: torch.Tensor = None,
                use_badm: bool = True, use_aadm: bool = True) -> Dict[str, torch.Tensor]:

        labels_float = labels.float().unsqueeze(1)
        labels_smoothed = labels_float
        if self.config.label_smoothing > 0:
            labels_smoothed = labels_float * (1 - self.config.label_smoothing) + 0.5 * self.config.label_smoothing

        loss_main = self.bce(outputs["logit"], labels_smoothed)

        loss_branch = torch.tensor(0.0, device=labels.device)
        if use_badm and use_aadm:
            loss_branch = 0.5 * (
                self.bce(outputs["badm_logit"], labels_smoothed) +
                self.bce(outputs["aadm_logit"], labels_smoothed)
            )
        elif use_badm:
            loss_branch = self.bce(outputs["badm_logit"], labels_smoothed)
        elif use_aadm:
            loss_branch = self.bce(outputs["aadm_logit"], labels_smoothed)

        loss_orthogonal = torch.tensor(0.0, device=labels.device)
        if use_badm and use_aadm:
            badm_norm = torch.norm(outputs["badm_evidence"], dim=1, keepdim=True)
            aadm_norm = torch.norm(outputs["aadm_evidence"], dim=1, keepdim=True)

            if (badm_norm > 1e-6).all() and (aadm_norm > 1e-6).all():
                badm_ev = outputs["badm_evidence"] / (badm_norm + 1e-8)
                aadm_ev = outputs["aadm_evidence"] / (aadm_norm + 1e-8)
                cosine_sim = (badm_ev * aadm_ev).sum(dim=1)
                loss_orthogonal = F.relu(torch.abs(cosine_sim) - self.config.orthogonality_margin).mean()

        loss_edge_alignment = torch.tensor(0.0, device=labels.device)
        loss_freq_alignment = torch.tensor(0.0, device=labels.device)

        if images is not None and use_badm:
            edges = self.compute_edge_map(images)

            badm_ev_spatial = outputs["badm_evidence"].view(outputs["badm_evidence"].size(0), -1)
            edges_flat = edges.view(edges.size(0), -1)

            badm_ev_norm = F.normalize(badm_ev_spatial, dim=1)
            edges_norm = F.normalize(edges_flat, dim=1)

            alignment = -(badm_ev_norm * edges_norm).sum(dim=1).mean()
            loss_edge_alignment = alignment

        if images is not None and use_aadm:
            freq_map = self.compute_freq_map(images)

            aadm_ev_spatial = outputs["aadm_evidence"].view(outputs["aadm_evidence"].size(0), -1)
            freq_flat = freq_map.view(freq_map.size(0), -1)

            aadm_ev_norm = F.normalize(aadm_ev_spatial, dim=1)
            freq_norm = F.normalize(freq_flat, dim=1)

            alignment = -(aadm_ev_norm * freq_norm).sum(dim=1).mean()
            loss_freq_alignment = alignment

        loss_deep_supervision = torch.tensor(0.0, device=labels.device)
        if outputs["iteration_logits"].shape[1] >= 1:
            num_iters = outputs["iteration_logits"].shape[1]
            weights = torch.linspace(1.0 / num_iters, 1.0, num_iters, device=labels.device)
            weights = weights / weights.sum()

            labels_expanded = labels_smoothed.expand(-1, num_iters).to(device=labels.device)
            iter_losses = F.binary_cross_entropy_with_logits(outputs["iteration_logits"], labels_expanded, reduction='none')
            loss_deep_supervision = (iter_losses.mean(dim=0) * weights).sum()

        total_loss = (
            self.config.lambda_main * loss_main +
            self.config.lambda_branch * loss_branch +
            self.config.lambda_orthogonal * loss_orthogonal +
            self.config.lambda_deep_supervision * loss_deep_supervision +
            self.config.lambda_edge_alignment * loss_edge_alignment +
            self.config.lambda_freq_alignment * loss_freq_alignment
        )

        return {
            "total": total_loss,
            "main": loss_main,
            "branch": loss_branch,
            "orthogonal": loss_orthogonal,
            "deep_supervision": loss_deep_supervision,
            "edge_alignment": loss_edge_alignment,
            "freq_alignment": loss_freq_alignment,
        }
