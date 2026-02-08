import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class EvidenceCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.apply(_init_weights)

    def forward(self, query: torch.Tensor, evidence: torch.Tensor):
        B = query.shape[0]
        num_evidence = evidence.shape[1]

        Q = self.q_proj(query).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(evidence).view(B, num_evidence, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(evidence).view(B, num_evidence, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, 1, -1)
        out = self.out_proj(out.squeeze(1))

        attn_weights = attn.mean(dim=1).squeeze(1)
        return out, attn_weights


class EvidenceRefinementModule(nn.Module):
    def __init__(self, evidence_dim: int = 128, num_heads: int = 4,
                 num_iterations: int = 3, dropout: float = 0.1,
                 max_evidence_sources: int = 2):
        super().__init__()
        self.num_iterations = num_iterations
        self.evidence_dim = evidence_dim
        self.max_evidence_sources = max_evidence_sources

        self.init_proj_single = nn.Sequential(
            nn.Linear(evidence_dim, evidence_dim),
            nn.ReLU(inplace=True),
        )
        self.init_proj_dual = nn.Sequential(
            nn.Linear(evidence_dim * 2, evidence_dim),
            nn.ReLU(inplace=True),
        )

        self.attention = EvidenceCrossAttention(evidence_dim, num_heads, dropout)
        self.gru = nn.GRUCell(evidence_dim, evidence_dim)
        self.layer_norm = nn.LayerNorm(evidence_dim)
        self.ffn = nn.Sequential(
            nn.Linear(evidence_dim, evidence_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(evidence_dim * 2, evidence_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(evidence_dim)
        self.predictor = nn.Linear(evidence_dim, 1)

        self.apply(_init_weights)

    def forward(self, *evidence_list: torch.Tensor) -> Dict[str, torch.Tensor]:
        num_sources = len(evidence_list)
        if num_sources < 1 or num_sources > self.max_evidence_sources:
            raise ValueError(f"ERM expects 1-{self.max_evidence_sources} evidence sources, got {num_sources}")

        B = evidence_list[0].shape[0]
        device = evidence_list[0].device
        evidence_stack = torch.stack(evidence_list, dim=1)

        if num_sources == 1:
            h = self.init_proj_single(evidence_list[0])
        else:
            h = self.init_proj_dual(torch.cat(evidence_list, dim=1))

        iteration_logits = torch.zeros(B, self.num_iterations, 1, device=device)
        iteration_probs = torch.zeros(B, self.num_iterations, 1, device=device)
        attention_history = torch.zeros(B, self.num_iterations, self.max_evidence_sources, device=device)

        for t in range(self.num_iterations):
            context, attn_weights = self.attention(h, evidence_stack)
            attention_history[:, t, :num_sources] = attn_weights

            h = self.layer_norm(self.gru(context, h))
            h = self.ffn_norm(h + self.ffn(h))

            logit = self.predictor(h)

            iteration_logits[:, t] = logit
            iteration_probs[:, t] = torch.sigmoid(logit)

        final_logit = iteration_logits[:, -1]

        convergence_delta = torch.tensor(0.0, device=device)
        if self.num_iterations > 1:
            prob_diff = torch.abs(iteration_probs[:, -1] - iteration_probs[:, -2])
            convergence_delta = prob_diff.mean()

        return {
            "final_logit": final_logit,
            "iteration_logits": iteration_logits.squeeze(-1),
            "iteration_probs": iteration_probs.squeeze(-1),
            "attention_history": attention_history,
            "convergence_delta": convergence_delta,
        }
