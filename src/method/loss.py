import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from dataclasses import dataclass


@dataclass
class LossConfig:
    lambda_main: float = 1.0
    lambda_branch: float = 0.3
    lambda_orthogonal: float = 0.01
    lambda_deep_supervision: float = 0.05
    label_smoothing: float = 0.05
    orthogonality_margin: float = 0.1


class RADARLoss(nn.Module):
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor,
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
            self.config.lambda_deep_supervision * loss_deep_supervision
        )

        return {
            "total": total_loss,
            "main": loss_main,
            "branch": loss_branch,
            "orthogonal": loss_orthogonal,
            "deep_supervision": loss_deep_supervision,
        }
