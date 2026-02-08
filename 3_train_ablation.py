#!/usr/bin/env python3

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
import time
from datetime import datetime
import multiprocessing as mp

sys.path.append(str(Path(__file__).parent / 'src'))

os.environ['TORCH_HOME'] = str(Path(__file__).parent / 'data' / 'torch_cache')
os.environ['HF_HOME'] = str(Path(__file__).parent / 'data' / 'hf_cache')
os.environ['TIMM_CACHE_DIR'] = str(Path(__file__).parent / 'data' / 'torch_cache' / 'hub')

from method import RADAR, RADARConfig
from data.dataset import DeepfakeDataset, get_train_transforms, get_val_transforms
from data.splits import load_presplit_data, is_presplit_dataset
from experiments.train import train_model as _train_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="RADAR Ablation Study")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--use_badm", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--use_aadm", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--reasoning_iterations", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.use_badm and not args.use_aadm:
        raise ValueError("At least one module (BADM or AADM) must be enabled")

    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        optimal_batch = 128 if gpu_mem_gb > 40 else 64
        optimal_workers = min(8, max(1, mp.cpu_count() - 1))
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {gpu_mem_gb:.1f}GB")
    else:
        optimal_batch = args.batch_size
        optimal_workers = 4

    config = {
        "experiment_name": args.experiment_name,
        "seed": args.seed,
        "data_dir": args.data_dir,
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 384,
        "evidence_dim": 128,
        "reasoning_iterations": args.reasoning_iterations,
        "reasoning_heads": 4,
        "fft_size": 112,
        "dropout": 0.1,
        "batch_size": optimal_batch,
        "num_epochs": args.num_epochs,
        "early_stopping_patience": 10,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
        "gradient_accumulation_steps": 1,
        "num_workers": optimal_workers,
        "lambda_main": 1.0,
        "lambda_branch": 0.3 if (args.use_badm and args.use_aadm) else 0.0,
        "lambda_orthogonal": 0.01 if (args.use_badm and args.use_aadm) else 0.0,
        "lambda_deep_supervision": 0.05,
        "label_smoothing": 0.05,
        "orthogonality_margin": 0.1,
        "use_badm": args.use_badm,
        "use_aadm": args.use_aadm,
        "freeze_backbone_epochs": 3,
    }

    print("\n" + "="*70)
    print("RADAR ABLATION STUDY")
    print("="*70)
    print(f"Experiment: {config['experiment_name']}")
    print(f"BADM: {'Enabled' if args.use_badm else 'Disabled'}")
    print(f"AADM: {'Enabled' if args.use_aadm else 'Disabled'}")
    print(f"Reasoning Iterations: {args.reasoning_iterations}")
    print(f"Evidence Dim: {config['evidence_dim']}")
    print(f"Seed: {config['seed']}")
    print(f"Batch size: {config['batch_size']}")
    print("="*70)

    set_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir) / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    print("\nLOADING DATASET")
    print("="*70)

    data_dir = Path(config["data_dir"])

    if is_presplit_dataset(data_dir):
        print(f"Detected pre-split dataset structure")
        train_images, train_labels = load_presplit_data(data_dir, "train")
        val_images, val_labels = load_presplit_data(data_dir, "val")
        test_images, test_labels = load_presplit_data(data_dir, "test") if (data_dir / "test").exists() else (val_images, val_labels)
    else:
        raise ValueError("This ablation script requires pre-split dataset")

    print(f"Train samples: {len(train_images):,}")
    print(f"Val samples: {len(val_images):,}")
    print(f"Test samples: {len(test_images):,}")

    train_dataset = DeepfakeDataset(
        train_images, train_labels,
        get_train_transforms(config["img_size"]),
        preprocess_dir=None,
        validate_cache=False
    )

    val_dataset = DeepfakeDataset(
        val_images, val_labels,
        get_val_transforms(config["img_size"]),
        preprocess_dir=None,
        validate_cache=False
    )

    test_dataset = DeepfakeDataset(
        test_images, test_labels,
        get_val_transforms(config["img_size"]),
        preprocess_dir=None,
        validate_cache=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=True, num_workers=config["num_workers"],
        pin_memory=True, drop_last=True,
        persistent_workers=config["num_workers"] > 0,
        prefetch_factor=4 if config["num_workers"] > 0 else None
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=config["num_workers"] > 0,
        prefetch_factor=4 if config["num_workers"] > 0 else None
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=config["num_workers"] > 0,
        prefetch_factor=4 if config["num_workers"] > 0 else None
    )

    model_config = RADARConfig(
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        evidence_dim=config["evidence_dim"],
        reasoning_iterations=config["reasoning_iterations"],
        reasoning_heads=config["reasoning_heads"],
        fft_size=config["fft_size"],
        dropout=config["dropout"],
        freeze_backbone_epochs=config["freeze_backbone_epochs"]
    )
    model = RADAR(model_config).to(device)

    history, best_auc = _train_model(model, train_loader, val_loader, config, device, output_dir)

    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)

    from experiments.train import evaluate

    checkpoint = torch.load(output_dir / "best.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_metrics = evaluate(model, test_loader, device,
                           use_badm=config["use_badm"],
                           use_aadm=config["use_aadm"])

    print(f"\nTest Results:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")

    results = {
        "ablation_config": {
            "use_badm": args.use_badm,
            "use_aadm": args.use_aadm,
            "reasoning_iterations": args.reasoning_iterations,
        },
        "history": history,
        "best_val_auc": float(best_auc),
        "test_metrics": {
            "auc": float(test_metrics["auc"]),
            "accuracy": float(test_metrics["accuracy"])
        },
        "timestamp": datetime.now().isoformat()
    }

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("ABLATION COMPLETE")
    print("="*70)
    print(f"Configuration: BADM={args.use_badm}, AADM={args.use_aadm}, Iters={args.reasoning_iterations}")
    print(f"Best Val AUC: {best_auc:.4f}")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    print(f"Results saved to: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
