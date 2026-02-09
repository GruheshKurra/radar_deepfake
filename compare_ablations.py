#!/usr/bin/env python3
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from method import RADAR, RADARConfig
from baselines.vit import ViTBaseline
from torch.utils.data import DataLoader
from data.dataset import DeepfakeDataset, get_val_transforms
from data.splits import load_presplit_data
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

data_dir = Path("./data")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = RADARConfig(
    img_size=224, patch_size=16, embed_dim=384,
    evidence_dim=128, reasoning_iterations=3,
    reasoning_heads=4, fft_size=112, dropout=0.1
)

print("\n" + "="*70)
print("ABLATION COMPARISON ANALYSIS")
print("="*70)

val_images, val_labels = load_presplit_data(data_dir, "val")
val_dataset = DeepfakeDataset(
    val_images[:1000], val_labels[:1000],
    get_val_transforms(224),
    preprocess_dir=None, validate_cache=False
)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

labels_all = []
preds_full = []
preds_badm_only = []
preds_aadm_only = []
preds_no_reasoning = []

try:
    model_full = RADAR(config).to(device)
    checkpoint = torch.load("./outputs/radar_combined/best.pth", map_location=device)
    model_full.load_state_dict(checkpoint['model_state_dict'])
    model_full.eval()

    print("\nEvaluating configurations...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels_all.extend(labels.numpy())

            # Full model
            out_full = model_full(images, use_badm=True, use_aadm=True)
            preds_full.extend(out_full['prob'].cpu().numpy())

            # BADM only
            out_badm = model_full(images, use_badm=True, use_aadm=False)
            preds_badm_only.extend(out_badm['prob'].cpu().numpy())

            # AADM only
            out_aadm = model_full(images, use_badm=False, use_aadm=True)
            preds_aadm_only.extend(out_aadm['prob'].cpu().numpy())

            # No reasoning (1 iteration)
            model_full.erm.num_iterations = 1
            out_no_reasoning = model_full(images, use_badm=True, use_aadm=True)
            preds_no_reasoning.extend(out_no_reasoning['prob'].cpu().numpy())
            model_full.erm.num_iterations = 3

    labels_all = np.array(labels_all)
    preds_full = np.array(preds_full).flatten()
    preds_badm_only = np.array(preds_badm_only).flatten()
    preds_aadm_only = np.array(preds_aadm_only).flatten()
    preds_no_reasoning = np.array(preds_no_reasoning).flatten()

    print("\n" + "="*70)
    print(f"{'Configuration':<30} {'AUC':<10} {'Accuracy':<10} {'Δ AUC':<10}")
    print("-" * 70)

    auc_full = roc_auc_score(labels_all, preds_full)
    acc_full = accuracy_score(labels_all, (preds_full > 0.5).astype(int))
    print(f"{'Full RADAR':<30} {auc_full:.4f}    {acc_full:.4f}     -")

    auc_badm = roc_auc_score(labels_all, preds_badm_only)
    acc_badm = accuracy_score(labels_all, (preds_badm_only > 0.5).astype(int))
    delta_badm = (auc_full - auc_badm) * 100
    print(f"{'BADM only (no AADM)':<30} {auc_badm:.4f}    {acc_badm:.4f}     {delta_badm:+.2f}%")

    auc_aadm = roc_auc_score(labels_all, preds_aadm_only)
    acc_aadm = accuracy_score(labels_all, (preds_aadm_only > 0.5).astype(int))
    delta_aadm = (auc_full - auc_aadm) * 100
    print(f"{'AADM only (no BADM)':<30} {auc_aadm:.4f}    {acc_aadm:.4f}     {delta_aadm:+.2f}%")

    auc_no_reas = roc_auc_score(labels_all, preds_no_reasoning)
    acc_no_reas = accuracy_score(labels_all, (preds_no_reasoning > 0.5).astype(int))
    delta_reas = (auc_full - auc_no_reas) * 100
    print(f"{'No reasoning (1 iter)':<30} {auc_no_reas:.4f}    {acc_no_reas:.4f}     {delta_reas:+.2f}%")

    print("\n" + "="*70)
    print("INTERPRETATION:")
    print("="*70)

    if delta_badm < 1.0 and delta_aadm < 1.0:
        print("\n⚠️  BIG PROBLEM: Removing BADM OR AADM drops AUC < 1%")
        print("    → Both components contribute very little!")
        print("    → ViT backbone is likely doing all the work")
        print("    → Need to check ViT-only baseline")

    elif delta_badm < 1.0:
        print("\n⚠️  BADM contributes < 1% - might be redundant")

    elif delta_aadm < 1.0:
        print("\n⚠️  AADM contributes < 1% - might be redundant")

    if delta_reas < 0.5:
        print("\n⚠️  Reasoning (ERM) contributes < 0.5% - extra iterations useless!")

    if max(delta_badm, delta_aadm, delta_reas) >= 2.0:
        print("\n✅ At least one component shows >= 2% contribution - publishable!")

except FileNotFoundError:
    print("\n⚠️  Model checkpoint not found at ./outputs/radar_combined/best.pth")
    print("    Train the model first!")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
