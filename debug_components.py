#!/usr/bin/env python3
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from method import RADAR, RADARConfig
from torch.utils.data import DataLoader
from data.dataset import DeepfakeDataset, get_val_transforms
from data.splits import load_presplit_data
import numpy as np

checkpoint_path = "./outputs/radar_combined/best.pth"
data_dir = Path("./data")

config = RADARConfig(
    img_size=224, patch_size=16, embed_dim=384,
    evidence_dim=128, reasoning_iterations=3,
    reasoning_heads=4, fft_size=112, dropout=0.1
)

model = RADAR(config)
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model_state_dict'])
model.eval()

val_images, val_labels = load_presplit_data(data_dir, "val")
val_dataset = DeepfakeDataset(
    val_images[:100], val_labels[:100],
    get_val_transforms(224),
    preprocess_dir=None, validate_cache=False
)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("\n" + "="*70)
print("COMPONENT CONTRIBUTION ANALYSIS")
print("="*70)

gating_alphas = []
badm_norms = []
aadm_norms = []
convergence_deltas = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images, freq_cached=None, sobel_cached=None,
                       use_badm=True, use_aadm=True)

        gating_alphas.append(outputs['gating_alpha'].mean().item())

        if 'badm_evidence' in outputs:
            badm_norms.append(outputs['badm_evidence'].norm(dim=1).mean().item())
        if 'aadm_evidence' in outputs:
            aadm_norms.append(outputs['aadm_evidence'].norm(dim=1).mean().item())
        if 'convergence_delta' in outputs:
            convergence_deltas.append(outputs['convergence_delta'].mean().item())

print(f"\n{'Metric':<30} {'Mean':<15} {'Interpretation'}")
print("-" * 70)

alpha_mean = np.mean(gating_alphas)
print(f"{'Gating Alpha (α)':<30} {alpha_mean:.4f}      ", end="")
if alpha_mean < 0.2:
    print("⚠️  External path dominates! Reasoning ignored!")
elif alpha_mean > 0.8:
    print("⚠️  Reasoning dominates! External ignored!")
else:
    print("✅ Both paths used")

badm_mean = np.mean(badm_norms)
aadm_mean = np.mean(aadm_norms)
print(f"{'BADM Evidence Norm':<30} {badm_mean:.4f}")
print(f"{'AADM Evidence Norm':<30} {aadm_mean:.4f}")

ratio = badm_mean / (aadm_mean + 1e-8)
print(f"{'BADM/AADM Ratio':<30} {ratio:.4f}      ", end="")
if 0.8 < ratio < 1.2:
    print("⚠️  Similar magnitudes! Might be redundant")
else:
    print("✅ Different contributions")

conv_mean = np.mean(convergence_deltas)
print(f"{'Convergence Delta':<30} {conv_mean:.6f}    ", end="")
if conv_mean < 0.001:
    print("⚠️  Converges in 1 iteration! Extra iterations useless!")
else:
    print("✅ Iterative refinement working")

print("\n" + "="*70)
print("RECOMMENDATIONS:")
print("="*70)

issues = []
if alpha_mean < 0.2 or alpha_mean > 0.8:
    issues.append("• Gating mechanism not balanced - one path ignored")
if 0.8 < ratio < 1.2:
    issues.append("• BADM and AADM producing similar features - redundant")
if conv_mean < 0.001:
    issues.append("• Reasoning converges immediately - extra iterations useless")

if issues:
    print("\n⚠️  PROBLEMS FOUND:")
    for issue in issues:
        print(issue)
    print("\nThis explains why ablations don't show big differences!")
else:
    print("\n✅ All components appear to be contributing!")
    print("If ablations still show no difference, check ViT baseline.")
