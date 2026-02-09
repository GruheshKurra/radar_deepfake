import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional
import timm
import os
from pathlib import Path

from .frequency import FrequencyArtifactDetector
from .boundary import BoundaryArtifactDetector
from .reasoning import EvidenceRefinementModule

os.environ['TORCH_HOME'] = str(Path(__file__).parent.parent.parent / 'data' / 'torch_cache')
os.environ['TIMM_CACHE_DIR'] = str(Path(__file__).parent.parent.parent / 'data' / 'torch_cache' / 'hub')


@dataclass
class RADARConfig:
    backbone: str = "vit_small_patch16_224"
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 384
    evidence_dim: int = 128
    reasoning_iterations: int = 3
    reasoning_heads: int = 4
    fft_size: int = 112
    dropout: float = 0.1
    gating_init: float = 0.5
    freeze_backbone_epochs: int = 3


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class RADAR(nn.Module):
    def __init__(self, config: RADARConfig):
        super().__init__()
        self.config = config

        self.encoder = timm.create_model(
            config.backbone,
            pretrained=True,
            num_classes=0,
        )

        embed_dim = self.encoder.num_features

        self.badm = BoundaryArtifactDetector(embed_dim, config.evidence_dim)
        self.aadm = FrequencyArtifactDetector(
            embed_dim, config.evidence_dim, config.fft_size
        )

        self.reasoning = EvidenceRefinementModule(
            config.evidence_dim, config.reasoning_heads,
            config.reasoning_iterations, config.dropout,
            max_evidence_sources=2
        )

        self.external_classifier_dual = nn.Sequential(
            nn.Linear(config.evidence_dim * 2, config.evidence_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.evidence_dim, 1)
        )
        self.external_classifier_single = nn.Sequential(
            nn.Linear(config.evidence_dim, config.evidence_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.evidence_dim // 2, 1)
        )

        self._gating_logit = nn.Parameter(torch.tensor(0.0))
        if config.gating_init != 0.5:
            init_val = torch.tensor(config.gating_init).clamp(0.01, 0.99)
            self._gating_logit = nn.Parameter(torch.log(init_val / (1 - init_val)))

        self.external_classifier_dual.apply(_init_weights)
        self.external_classifier_single.apply(_init_weights)

    @property
    def gating_alpha(self) -> torch.Tensor:
        return torch.sigmoid(self._gating_logit)

    def freeze_backbone(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def get_backbone_params(self):
        return list(self.encoder.parameters())

    def get_head_params(self):
        head_modules = [self.badm, self.aadm, self.reasoning,
                        self.external_classifier_dual, self.external_classifier_single]
        params = [self._gating_logit]
        for module in head_modules:
            params.extend(module.parameters())
        return params

    def _extract_features(self, x: torch.Tensor):
        features = self.encoder.forward_features(x)
        if features.ndim == 3:
            cls_token = features[:, 0]
            patch_features = features[:, 1:]
        elif features.ndim == 4:
            cls_token = features.mean(dim=[2, 3])
            B, C, H, W = features.shape
            patch_features = features.flatten(2).transpose(1, 2)
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")
        return cls_token, patch_features

    def forward(self, x: torch.Tensor, freq_cached: Optional[torch.Tensor] = None,
                sobel_cached: Optional[torch.Tensor] = None, use_badm: bool = True,
                use_aadm: bool = True) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        device = x.device

        cls_token, patch_features = self._extract_features(x)

        badm_out = self.badm(x, patch_features, sobel_cached) if use_badm else None
        aadm_out = self.aadm(x, cls_token, freq_cached) if use_aadm else None

        evidence_list = []
        if badm_out is not None:
            evidence_list.append(badm_out["evidence"])
        if aadm_out is not None:
            evidence_list.append(aadm_out["evidence"])

        if len(evidence_list) > 0:
            reasoning_out = self.reasoning(*evidence_list)
            reasoning_logit = reasoning_out["final_logit"]

            if len(evidence_list) == 2:
                external_input = torch.cat(evidence_list, dim=1)
                external_logit = self.external_classifier_dual(external_input)
            else:
                external_logit = self.external_classifier_single(evidence_list[0])

            alpha = self.gating_alpha
            final_logit = alpha * reasoning_logit + (1 - alpha) * external_logit
        else:
            reasoning_out = {
                "final_logit": torch.zeros(B, 1, device=device),
                "iteration_logits": torch.zeros(B, self.config.reasoning_iterations, device=device),
                "iteration_probs": torch.zeros(B, self.config.reasoning_iterations, device=device),
                "attention_history": torch.zeros(B, self.config.reasoning_iterations, 2, device=device),
                "convergence_delta": torch.tensor(0.0, device=device),
            }
            reasoning_logit = torch.zeros(B, 1, device=device)
            external_logit = torch.zeros(B, 1, device=device)
            final_logit = torch.zeros(B, 1, device=device)

        return {
            "logit": final_logit,
            "prob": torch.sigmoid(final_logit),
            "reasoning_logit": reasoning_logit if len(evidence_list) > 0 else torch.zeros(B, 1, device=device),
            "reasoning_prob": torch.sigmoid(reasoning_logit) if len(evidence_list) > 0 else torch.zeros(B, 1, device=device),
            "external_logit": external_logit,
            "gating_alpha": self.gating_alpha.expand(B, 1),
            "badm_logit": badm_out["logit"] if badm_out is not None else torch.zeros(B, 1, device=device),
            "badm_score": badm_out["score"] if badm_out is not None else torch.zeros(B, 1, device=device),
            "badm_evidence": badm_out["evidence"] if badm_out is not None else torch.zeros(B, self.config.evidence_dim, device=device),
            "aadm_logit": aadm_out["logit"] if aadm_out is not None else torch.zeros(B, 1, device=device),
            "aadm_score": aadm_out["score"] if aadm_out is not None else torch.zeros(B, 1, device=device),
            "aadm_evidence": aadm_out["evidence"] if aadm_out is not None else torch.zeros(B, self.config.evidence_dim, device=device),
            "attention_history": reasoning_out["attention_history"],
            "iteration_logits": reasoning_out["iteration_logits"],
            "iteration_probs": reasoning_out["iteration_probs"],
            "convergence_delta": reasoning_out["convergence_delta"],
        }
