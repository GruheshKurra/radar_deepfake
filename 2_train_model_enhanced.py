#!/usr/bin/env python3
"""
Enhanced RADAR Training with Component Specialization

Changes from standard training:
1. Stronger orthogonality loss (0.5 instead of 0.01)
2. Edge alignment loss - forces BADM to focus on boundaries
3. Frequency alignment loss - forces AADM to focus on high-freq
4. Gradual ViT unfreeze - components learn first, then ViT adapts
"""

import torch
from torch.utils.data import DataLoader
import argparse
import sys
import os
from pathlib import Path
import random
import numpy as np
import json
import yaml
from tqdm import tqdm
import multiprocessing as mp

sys.path.append(str(Path(__file__).parent / 'src'))

os.environ['TORCH_HOME'] = str(Path(__file__).parent / 'data' / 'torch_cache')
os.environ['HF_HOME'] = str(Path(__file__).parent / 'data' / 'hf_cache')
os.environ['TIMM_CACHE_DIR'] = str(Path(__file__).parent / 'data' / 'torch_cache' / 'hub')

from method import RADAR, RADARConfig
from method.loss_enhanced import EnhancedRADARLoss, EnhancedLossConfig
from data.dataset import DeepfakeDataset, get_train_transforms, get_val_transforms
from data.splits import load_presplit_data, is_presplit_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, accuracy_score


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, optimizer, scheduler, loss_fn, scaler, device, config):
    model.train()
    total_loss = 0
    loss_components = {"main": 0, "branch": 0, "orthogonal": 0,
                      "deep_supervision": 0, "edge_alignment": 0, "freq_alignment": 0}

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
            outputs = model(images, freq_cached=None, sobel_cached=None,
                          use_badm=True, use_aadm=True)
            losses = loss_fn(outputs, labels, images=images,
                           use_badm=True, use_aadm=True)
            loss = losses["total"]

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        for k in loss_components.keys():
            if k in losses:
                loss_components[k] += losses[k].item()

    num_batches = len(loader)
    return {k: v / num_batches for k, v in loss_components.items()}, total_loss / num_batches


def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            with autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
                outputs = model(images, use_badm=True, use_aadm=True)

            all_probs.extend(outputs['prob'].cpu().numpy().flatten())
            all_labels.extend(labels.numpy())

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, np.array(all_probs) > 0.5)

    return {"auc": auc, "accuracy": acc}


def main():
    parser = argparse.ArgumentParser(description="Enhanced RADAR Training")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--experiment_name", type=str, default="radar_enhanced")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "="*70)
    print("ENHANCED RADAR TRAINING - Component Specialization")
    print("="*70)
    print("Enhancements:")
    print("  • Stronger orthogonality loss (λ=0.5 vs 0.01)")
    print("  • Edge alignment loss for BADM")
    print("  • Frequency alignment loss for AADM")
    print("  • Gradual ViT unfreeze")
    print("="*70)

    data_dir = Path(args.data_dir)
    train_images, train_labels = load_presplit_data(data_dir, "train")
    val_images, val_labels = load_presplit_data(data_dir, "val")

    train_dataset = DeepfakeDataset(train_images, train_labels,
                                   get_train_transforms(224))
    val_dataset = DeepfakeDataset(val_images, val_labels,
                                 get_val_transforms(224))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=6, pin_memory=True)

    model_config = RADARConfig(
        img_size=224, patch_size=16, embed_dim=384,
        evidence_dim=128, reasoning_iterations=3,
        reasoning_heads=4, fft_size=112, dropout=0.1
    )
    model = RADAR(model_config).to(device)

    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_config = EnhancedLossConfig(
        lambda_main=1.0,
        lambda_branch=0.3,
        lambda_orthogonal=0.5,
        lambda_deep_supervision=0.05,
        lambda_edge_alignment=0.2,
        lambda_freq_alignment=0.2,
        label_smoothing=0.05,
        orthogonality_margin=0.05
    )
    loss_fn = EnhancedRADARLoss(loss_config).to(device)

    model.freeze_backbone()
    print("Backbone frozen for epochs 1-3 (components learn first)")

    backbone_lr = args.learning_rate * 0.02
    param_groups = [
        {"params": model.get_head_params(), "lr": args.learning_rate},
        {"params": model.get_backbone_params(), "lr": backbone_lr},
    ]

    optimizer = AdamW(param_groups, lr=args.learning_rate, weight_decay=0.05)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[args.learning_rate, backbone_lr],
        total_steps=len(train_loader) * args.num_epochs,
        pct_start=0.1,
        anneal_strategy='cos'
    )

    scaler = GradScaler(device_type='cuda' if device == 'cuda' else 'cpu')

    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    best_auc = 0
    history = []
    patience = 0

    for epoch in range(args.num_epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"{'='*70}")

        if epoch == 3:
            model.unfreeze_backbone()
            print("✓ Backbone unfrozen - full model training")

        loss_components, train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, scaler, device, {}
        )

        metrics = evaluate(model, val_loader, device)

        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"    ├─ Main: {loss_components['main']:.4f}")
        print(f"    ├─ Branch: {loss_components['branch']:.4f}")
        print(f"    ├─ Orthogonal: {loss_components['orthogonal']:.4f}")
        print(f"    ├─ Deep Supervision: {loss_components['deep_supervision']:.4f}")
        print(f"    ├─ Edge Alignment: {loss_components['edge_alignment']:.4f}")
        print(f"    └─ Freq Alignment: {loss_components['freq_alignment']:.4f}")
        print(f"  Val AUC: {metrics['auc']:.4f}")
        print(f"  Val Accuracy: {metrics['accuracy']*100:.2f}%")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            **loss_components,
            **metrics
        })

        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            patience = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'auc': best_auc
            }, output_dir / 'best.pth')
            print(f"  ✓ New best model saved (AUC: {best_auc:.4f})")
        else:
            patience += 1
            print(f"  Patience: {patience}/10")

        if patience >= 10:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Best Val AUC: {best_auc:.4f}")
    print(f"Model saved to: {output_dir / 'best.pth'}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
