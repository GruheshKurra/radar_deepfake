#!/usr/bin/env python3
"""
Ablation Study: Baseline vs Baseline + RADAR Modules

For each backbone architecture:
  1. Train baseline alone (backbone + linear head) for N epochs
  2. Train backbone + RADAR modules (BADM + AADM + ERM) for N epochs
  3. Compare test AUC/accuracy to show RADAR's contribution

Usage:
  python 3_train_ablation.py --data_dir ./data
  python 3_train_ablation.py --data_dir ./data --backbone resnet50
  python 3_train_ablation.py --data_dir ./data/radar-deepfake-frames.zip --num_epochs 10
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import argparse
import sys
import os
from pathlib import Path
from collections import OrderedDict
import random
import numpy as np
import json
import time
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
import timm

sys.path.append(str(Path(__file__).parent / 'src'))

os.environ['TORCH_HOME'] = str(Path(__file__).parent / 'data' / 'torch_cache')
os.environ['HF_HOME'] = str(Path(__file__).parent / 'data' / 'hf_cache')
os.environ['TIMM_CACHE_DIR'] = str(Path(__file__).parent / 'data' / 'torch_cache' / 'hub')

from method import RADAR, RADARConfig, RADARLoss, LossConfig
from data.dataset import DeepfakeDataset, get_train_transforms, get_val_transforms
from data.splits import load_presplit_data, is_presplit_dataset
from sklearn.metrics import roc_auc_score, accuracy_score

try:
    from data.zip_dataset import ZipDeepfakeDataset
    from data.zip_loader import get_zip_splits
    HAS_ZIP_SUPPORT = True
except ImportError:
    HAS_ZIP_SUPPORT = False


BACKBONES = OrderedDict([
    ("vit_small_patch16_224", "ViT-Small"),
    ("resnet50", "ResNet-50"),
    ("xception", "Xception"),
    ("tf_efficientnet_b0", "EfficientNet-B0"),
])


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False   # Trade reproducibility for speed
    torch.backends.cudnn.benchmark = True         # Auto-tune conv algorithms


@torch.inference_mode()
def evaluate_model(model, loader, device, is_radar=False):
    model.eval()
    all_probs = []
    all_labels = []

    for batch_data in loader:
        images = batch_data[0].to(device, non_blocking=True)
        if device == "cuda":
            images = images.to(memory_format=torch.channels_last)
        labels = batch_data[1]

        with autocast("cuda", enabled=(device == "cuda")):
            if is_radar:
                outputs = model(images)
                probs = outputs["prob"].cpu()
            else:
                logits = model(images)
                probs = torch.sigmoid(logits).cpu()

        all_probs.append(probs)
        all_labels.append(labels)

    all_probs = torch.cat(all_probs).numpy().flatten()
    all_labels = torch.cat(all_labels).numpy().flatten()
    preds = (all_probs > 0.5).astype(int)

    try:
        auc = float(roc_auc_score(all_labels, all_probs))
    except ValueError:
        auc = 0.5

    return {
        "auc": auc,
        "accuracy": float(accuracy_score(all_labels, preds)),
    }


def train_baseline(backbone_name, display_name, train_loader, val_loader, config, device):
    print(f"\n  Creating {display_name} baseline...")
    model = timm.create_model(backbone_name, pretrained=True, num_classes=1).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # torch.compile for ~10-20% speedup on A40 (Ampere+)
    compiled_model = model
    if device == "cuda" and hasattr(torch, 'compile'):
        try:
            compiled_model = torch.compile(model)
            print(f"  torch.compile enabled")
        except Exception as e:
            print(f"  torch.compile skipped: {e}")

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.05)
    grad_accum = config.get("gradient_accumulation_steps", 2)
    effective_steps = (len(train_loader) // grad_accum) * config["num_epochs"]
    scheduler = OneCycleLR(
        optimizer, max_lr=config["learning_rate"],
        total_steps=effective_steps, pct_start=0.1, anneal_strategy='cos'
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler("cuda", enabled=(device == "cuda"))

    # A40 optimization: channels_last memory format
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)

    best_auc = 0.0
    best_state = None

    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{config['num_epochs']}", leave=False)
        for batch_idx, batch_data in enumerate(pbar):
            images = batch_data[0].to(device, non_blocking=True)
            if device == "cuda":
                images = images.to(memory_format=torch.channels_last)
            labels = batch_data[1].to(device, non_blocking=True).float().unsqueeze(1)

            with autocast("cuda", enabled=(device == "cuda")):
                logits = compiled_model(images)
                loss = criterion(logits, labels) / grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            epoch_loss += loss.item() * grad_accum
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")

        pbar.close()

        val_metrics = evaluate_model(model, val_loader, device, is_radar=False)
        avg_loss = epoch_loss / max(num_batches, 1)
        marker = ""
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " ★"
        print(f"  Epoch {epoch+1}/{config['num_epochs']}: "
              f"loss={avg_loss:.4f}, val_auc={val_metrics['auc']:.4f}, val_acc={val_metrics['accuracy']:.4f}{marker}")

    if best_state:
        model.load_state_dict(best_state)
    model.to(device)

    return model, best_auc


def train_radar(backbone_name, display_name, train_loader, val_loader, config, device):
    print(f"\n  Creating {display_name} + RADAR modules...")
    model_config = RADARConfig(
        backbone=backbone_name,
        img_size=config["img_size"],
        evidence_dim=config["evidence_dim"],
        reasoning_iterations=config["reasoning_iterations"],
        reasoning_heads=config["reasoning_heads"],
        fft_size=config["fft_size"],
        dropout=config["dropout"],
        freeze_backbone_epochs=config["freeze_backbone_epochs"],
    )
    model = RADAR(model_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # torch.compile for ~10-20% speedup on A40 (Ampere+)
    compiled_model = model
    if device == "cuda" and hasattr(torch, 'compile'):
        try:
            compiled_model = torch.compile(model)
            print(f"  torch.compile enabled")
        except Exception as e:
            print(f"  torch.compile skipped: {e}")

    freeze_epochs = config["freeze_backbone_epochs"]
    if freeze_epochs > 0:
        model.freeze_backbone()
        print(f"  Backbone frozen for first {freeze_epochs} epochs")

    backbone_lr = config["learning_rate"] * 0.1
    param_groups = [
        {"params": model.get_head_params(), "lr": config["learning_rate"]},
        {"params": model.get_backbone_params(), "lr": backbone_lr},
    ]

    optimizer = AdamW(param_groups, lr=config["learning_rate"], weight_decay=0.05)
    grad_accum = config.get("gradient_accumulation_steps", 2)
    effective_steps = (len(train_loader) // grad_accum) * config["num_epochs"]
    scheduler = OneCycleLR(
        optimizer, max_lr=[config["learning_rate"], backbone_lr],
        total_steps=effective_steps, pct_start=0.1, anneal_strategy='cos'
    )

    loss_fn = RADARLoss(LossConfig(
        lambda_main=1.0, lambda_branch=0.3, lambda_orthogonal=0.01,
        lambda_deep_supervision=0.05, label_smoothing=0.05, orthogonality_margin=0.1
    ))
    scaler = GradScaler("cuda", enabled=(device == "cuda"))

    # A40 optimization: channels_last memory format
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)

    best_auc = 0.0
    best_state = None

    for epoch in range(config["num_epochs"]):
        if epoch == freeze_epochs and freeze_epochs > 0:
            model.unfreeze_backbone()
            print(f"  Backbone unfrozen at epoch {epoch + 1}")

        model.train()
        epoch_loss = 0
        num_batches = 0
        skipped = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{config['num_epochs']}", leave=False)
        for batch_idx, batch_data in enumerate(pbar):
            images = batch_data[0].to(device, non_blocking=True)
            if device == "cuda":
                images = images.to(memory_format=torch.channels_last)
            labels = batch_data[1].to(device, non_blocking=True)

            with autocast("cuda", enabled=(device == "cuda")):
                outputs = compiled_model(images)
                losses = loss_fn(outputs, labels)
                loss = losses["total"] / grad_accum

            if not torch.isfinite(loss):
                skipped += 1
                continue

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            epoch_loss += loss.item() * grad_accum
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")

        pbar.close()

        val_metrics = evaluate_model(model, val_loader, device, is_radar=True)
        avg_loss = epoch_loss / max(num_batches, 1)
        marker = ""
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " ★"
        line = (f"  Epoch {epoch+1}/{config['num_epochs']}: "
                f"loss={avg_loss:.4f}, val_auc={val_metrics['auc']:.4f}, val_acc={val_metrics['accuracy']:.4f}{marker}")
        if skipped > 0:
            line += f", skipped={skipped}"
        print(line)

    if best_state:
        model.load_state_dict(best_state)
    model.to(device)

    return model, best_auc


def main():
    parser = argparse.ArgumentParser(description="Ablation: Baseline vs Baseline + RADAR for each backbone")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Path to dataset directory or ZIP file")
    parser.add_argument("--output_dir", type=str, default="./outputs/ablations",
                        help="Output directory for results")
    parser.add_argument("--backbone", type=str, default=None,
                        help="Run only one backbone (e.g., resnet50). Default: run all.")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Training epochs per experiment")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0005,
                        help="Learning rate")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=2,
                        help="Epochs to freeze backbone in RADAR mode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f}GB)")

        # ── A40 / high-VRAM optimizations ──
        torch.backends.cuda.matmul.allow_tf32 = True   # Use TF32 for matmuls (19x faster on A40)
        torch.backends.cudnn.allow_tf32 = True          # Use TF32 for convolutions
        torch.backends.cudnn.benchmark = True            # Auto-tune conv algorithms
        torch.set_float32_matmul_precision('high')       # Prefer speed over precision

        if gpu_mem_gb > 40:  # A40-class GPU
            batch_size = args.batch_size if args.batch_size != 64 else 192
            eval_batch_size = 256  # Larger batch for eval (no gradients = less memory)
        else:
            batch_size = args.batch_size
            eval_batch_size = batch_size * 2

        num_workers = min(16, max(2, mp.cpu_count() - 1))
        print(f"  TF32 enabled, cudnn.benchmark=True")
        print(f"  Train batch: {batch_size}, Eval batch: {eval_batch_size}, Workers: {num_workers}")
    else:
        batch_size = args.batch_size
        eval_batch_size = batch_size
        num_workers = 4
        print(f"Device: {device}")

    config = {
        "num_epochs": args.num_epochs,
        "batch_size": batch_size,
        "learning_rate": args.learning_rate,
        "img_size": 224,
        "evidence_dim": 128,
        "reasoning_iterations": 3,
        "reasoning_heads": 4,
        "fft_size": 112,
        "dropout": 0.1,
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "gradient_accumulation_steps": 2,
    }

    # ---- Load data ----
    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    data_path = Path(args.data_dir)
    is_zip = data_path.suffix.lower() == '.zip' and data_path.is_file()

    if is_zip and not HAS_ZIP_SUPPORT:
        print("ERROR: ZIP file detected but zip support not available.")
        print("Extract first: python 1_fast_extract.py --zip_path", data_path)
        sys.exit(1)

    if is_zip and HAS_ZIP_SUPPORT:
        print(f"Loading from ZIP: {data_path.name}")
        splits = get_zip_splits(data_path)
        train_images, train_labels = splits['train']
        val_images, val_labels = splits['val']
        test_images, test_labels = splits['test']

        train_dataset = ZipDeepfakeDataset(
            data_path, train_images, train_labels, get_train_transforms(224))
        val_dataset = ZipDeepfakeDataset(
            data_path, val_images, val_labels, get_val_transforms(224))
        test_dataset = ZipDeepfakeDataset(
            data_path, test_images, test_labels, get_val_transforms(224))

    elif is_presplit_dataset(data_path):
        print(f"Loading pre-split dataset: {data_path}")
        train_images, train_labels = load_presplit_data(data_path, "train")
        val_images, val_labels = load_presplit_data(data_path, "val")
        if (data_path / "test").exists():
            test_images, test_labels = load_presplit_data(data_path, "test")
        else:
            test_images, test_labels = val_images, val_labels

        train_dataset = DeepfakeDataset(
            train_images, train_labels, get_train_transforms(224),
            preprocess_dir=None, validate_cache=False)
        val_dataset = DeepfakeDataset(
            val_images, val_labels, get_val_transforms(224),
            preprocess_dir=None, validate_cache=False)
        test_dataset = DeepfakeDataset(
            test_images, test_labels, get_val_transforms(224),
            preprocess_dir=None, validate_cache=False)
    else:
        print("ERROR: Data must be pre-split (train/val/test dirs) or a ZIP file.")
        sys.exit(1)

    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None)
    val_loader = DataLoader(
        val_dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None)
    test_loader = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None)

    # ---- Which backbones to run ----
    if args.backbone:
        if args.backbone not in BACKBONES:
            print(f"Unknown backbone: {args.backbone}")
            print(f"Available: {list(BACKBONES.keys())}")
            sys.exit(1)
        backbones_to_run = OrderedDict([(args.backbone, BACKBONES[args.backbone])])
    else:
        backbones_to_run = BACKBONES

    # ---- Run experiments ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "ablation_results.json"
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    print(f"\nExperiments: {len(backbones_to_run)} backbones × 2 modes = {len(backbones_to_run) * 2} runs")
    print(f"Epochs per run: {config['num_epochs']}")
    print(f"Batch size: {batch_size} (train), {eval_batch_size} (eval)")

    for backbone_name, display_name in backbones_to_run.items():
        print("\n" + "=" * 70)
        print(f"BACKBONE: {display_name} ({backbone_name})")
        print("=" * 70)

        # --- Train baseline ---
        print(f"\n{'─' * 70}")
        print(f"[1/2] Baseline: {display_name}")
        print(f"{'─' * 70}")

        set_seed(args.seed)
        start_time = time.time()
        baseline_model, baseline_val_auc = train_baseline(
            backbone_name, display_name, train_loader, val_loader, config, device)
        baseline_test = evaluate_model(baseline_model, test_loader, device, is_radar=False)
        baseline_time = time.time() - start_time

        baseline_params = sum(p.numel() for p in baseline_model.parameters())
        print(f"\n  ► Baseline result: val_auc={baseline_val_auc:.4f}, "
              f"test_auc={baseline_test['auc']:.4f}, test_acc={baseline_test['accuracy']:.4f} "
              f"({baseline_time:.0f}s)")

        # Save baseline checkpoint
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        baseline_ckpt = ckpt_dir / f"{backbone_name}_baseline.pt"
        torch.save({
            "model_state_dict": baseline_model.state_dict(),
            "backbone": backbone_name,
            "mode": "baseline",
            "test_auc": baseline_test["auc"],
            "test_acc": baseline_test["accuracy"],
            "val_auc": baseline_val_auc,
            "params": baseline_params,
            "config": config,
        }, baseline_ckpt)
        print(f"  ✓ Saved: {baseline_ckpt}")

        del baseline_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Train RADAR ---
        print(f"\n{'─' * 70}")
        print(f"[2/2] {display_name} + RADAR (BADM + AADM + ERM)")
        print(f"{'─' * 70}")

        set_seed(args.seed)
        start_time = time.time()
        radar_model, radar_val_auc = train_radar(
            backbone_name, display_name, train_loader, val_loader, config, device)
        radar_test = evaluate_model(radar_model, test_loader, device, is_radar=True)
        radar_time = time.time() - start_time

        radar_params = sum(p.numel() for p in radar_model.parameters())
        print(f"\n  ► RADAR result:   val_auc={radar_val_auc:.4f}, "
              f"test_auc={radar_test['auc']:.4f}, test_acc={radar_test['accuracy']:.4f} "
              f"({radar_time:.0f}s)")

        delta_auc = (radar_test["auc"] - baseline_test["auc"]) * 100
        delta_acc = (radar_test["accuracy"] - baseline_test["accuracy"]) * 100
        print(f"\n  ✓ AUC improvement:      {delta_auc:+.2f}%")
        print(f"  ✓ Accuracy improvement: {delta_acc:+.2f}%")
        print(f"  ✓ Parameter overhead:   {radar_params - baseline_params:,} "
              f"({(radar_params / baseline_params - 1) * 100:.1f}% more)")

        # Save RADAR checkpoint
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        radar_ckpt = ckpt_dir / f"{backbone_name}_radar.pt"
        torch.save({
            "model_state_dict": radar_model.state_dict(),
            "backbone": backbone_name,
            "mode": "radar",
            "test_auc": radar_test["auc"],
            "test_acc": radar_test["accuracy"],
            "val_auc": radar_val_auc,
            "params": radar_params,
            "config": config,
        }, radar_ckpt)
        print(f"  ✓ Saved: {radar_ckpt}")

        del radar_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        all_results[backbone_name] = {
            "display_name": display_name,
            "baseline": {
                "best_val_auc": float(baseline_val_auc),
                "test_auc": baseline_test["auc"],
                "test_acc": baseline_test["accuracy"],
                "params": baseline_params,
                "train_time_sec": round(baseline_time, 1),
            },
            "radar": {
                "best_val_auc": float(radar_val_auc),
                "test_auc": radar_test["auc"],
                "test_acc": radar_test["accuracy"],
                "params": radar_params,
                "train_time_sec": round(radar_time, 1),
            },
            "timestamp": datetime.now().isoformat(),
        }

        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

    # ---- Print summary table ----
    print("\n\n" + "=" * 90)
    print("FINAL RESULTS: Baseline vs Baseline + RADAR")
    print("=" * 90)

    header = (f"{'Backbone':<20} {'Baseline AUC':>14} {'+ RADAR AUC':>14} "
              f"{'Δ AUC':>10} {'Baseline Acc':>14} {'+ RADAR Acc':>14}")
    print(header)
    print("-" * 90)

    for backbone_name, data in all_results.items():
        name = data["display_name"]
        b_auc = data["baseline"]["test_auc"] * 100
        r_auc = data["radar"]["test_auc"] * 100
        b_acc = data["baseline"]["test_acc"] * 100
        r_acc = data["radar"]["test_acc"] * 100
        delta = r_auc - b_auc
        print(f"{name:<20} {b_auc:>13.2f}% {r_auc:>13.2f}% "
              f"{delta:>+9.2f}% {b_acc:>13.2f}% {r_acc:>13.2f}%")

    print("=" * 90)
    print(f"\nResults saved to: {results_file}")
    print(f"For formatted tables: python 4_compare_ablations.py")


if __name__ == "__main__":
    main()
