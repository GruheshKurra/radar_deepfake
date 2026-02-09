import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import time
import json
from typing import Dict, Tuple
import sys
import os
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

os.environ['TORCH_HOME'] = str(Path(__file__).parent.parent.parent / 'data' / 'torch_cache')
os.environ['HF_HOME'] = str(Path(__file__).parent.parent.parent / 'data' / 'hf_cache')
os.environ['TIMM_CACHE_DIR'] = str(Path(__file__).parent.parent.parent / 'data' / 'torch_cache' / 'hub')

from method import RADAR, RADARConfig, RADARLoss
from method.loss import LossConfig


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, scheduler,
                loss_fn, scaler, device: str, gradient_accumulation_steps: int, config: dict) -> Tuple[Dict[str, float], int, float]:
    model.train()
    total_losses = {"total": 0, "main": 0, "branch": 0, "orthogonal": 0,
                   "deep_supervision": 0}
    num_batches = 0
    skipped_batches = 0
    accumulated_batches = 0
    total_grad_norm = 0.0
    grad_norm_count = 0

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training", unit="batch")
    for batch_idx, batch_data in pbar:
        if len(batch_data) == 3:
            images, labels, extras = batch_data
            sobel_cached = extras.get("sobel_cached").to(device, non_blocking=True) if "sobel_cached" in extras else None
            if sobel_cached is not None and (sobel_cached.ndim != 4 or sobel_cached.size(1) != 1):
                sobel_cached = None
        else:
            images, labels = batch_data
            sobel_cached = None

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=(device=="cuda")):
            outputs = model(images, freq_cached=None, sobel_cached=sobel_cached,
                           use_badm=config.get("use_badm", True),
                           use_aadm=config.get("use_aadm", True))
            losses = loss_fn(outputs, labels,
                           use_badm=config.get("use_badm", True),
                           use_aadm=config.get("use_aadm", True))
            loss = losses["total"] / gradient_accumulation_steps

        if not torch.isfinite(loss):
            skipped_batches += 1
            continue

        scaler.scale(loss).backward()

        for k, v in losses.items():
            total_losses[k] += v.item()
        num_batches += 1
        accumulated_batches += 1

        is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps == 0
        is_last_batch = (batch_idx + 1) == len(loader)

        if is_accumulation_step or is_last_batch:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            total_grad_norm += grad_norm.item()
            grad_norm_count += 1
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accumulated_batches = 0

        if num_batches > 0:
            avg_loss = total_losses["total"] / num_batches
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "skipped": skipped_batches})

    pbar.close()
    avg_grad_norm = total_grad_norm / grad_norm_count if grad_norm_count > 0 else 0.0
    return {k: v / max(num_batches, 1) for k, v in total_losses.items()}, skipped_batches, avg_grad_norm


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: str,
            use_badm: bool = True, use_aadm: bool = True) -> Dict:
    model.eval()
    all_probs = []
    all_reasoning_probs = []
    all_labels = []
    all_convergence_deltas = []
    all_gating_alphas = []

    pbar = tqdm(loader, desc="Evaluating", unit="batch")
    for batch_data in pbar:
        if len(batch_data) == 3:
            images, labels, extras = batch_data
            sobel_cached = extras.get("sobel_cached").to(device) if "sobel_cached" in extras else None
            if sobel_cached is not None and (sobel_cached.ndim != 4 or sobel_cached.size(1) != 1):
                sobel_cached = None
        else:
            images, labels = batch_data
            sobel_cached = None

        images = images.to(device)

        with autocast(enabled=(device=="cuda")):
            outputs = model(images, freq_cached=None, sobel_cached=sobel_cached,
                           use_badm=use_badm, use_aadm=use_aadm)

        all_probs.append(outputs["prob"].cpu())
        all_reasoning_probs.append(outputs["reasoning_prob"].cpu())
        all_labels.append(labels)
        all_convergence_deltas.append(outputs["convergence_delta"].cpu())
        all_gating_alphas.append(outputs["gating_alpha"].cpu())

    pbar.close()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_reasoning_probs = torch.cat(all_reasoning_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_preds = (all_probs > 0.5).astype(int)
    all_reasoning_preds = (all_reasoning_probs > 0.5).astype(int)

    avg_convergence_delta = torch.stack([d if d.dim() == 0 else d.mean() for d in all_convergence_deltas]).mean().item()
    avg_gating_alpha = torch.cat(all_gating_alphas, dim=0).mean().item()

    from sklearn.metrics import accuracy_score, roc_auc_score

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "auc": roc_auc_score(all_labels, all_probs),
        "reasoning_accuracy": accuracy_score(all_labels, all_reasoning_preds),
        "reasoning_auc": roc_auc_score(all_labels, all_reasoning_probs),
        "gating_alpha": avg_gating_alpha,
        "convergence_delta": avg_convergence_delta,
        "probs": all_probs,
        "reasoning_probs": all_reasoning_probs,
        "labels": all_labels,
    }


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                config: dict, device: str, checkpoint_dir: Path, loss_config: LossConfig = None):
    if loss_config is None:
        loss_config = LossConfig(
            lambda_main=config.get("lambda_main", 1.0),
            lambda_branch=config.get("lambda_branch", 0.3),
            lambda_orthogonal=config.get("lambda_orthogonal", 0.01),
            lambda_deep_supervision=config.get("lambda_deep_supervision", 0.05),
            label_smoothing=config.get("label_smoothing", 0.05),
            orthogonality_margin=config.get("orthogonality_margin", 0.1)
        )
    loss_fn = RADARLoss(loss_config)

    freeze_epochs = config.get("freeze_backbone_epochs", 3)

    if freeze_epochs > 0 and hasattr(model, 'freeze_backbone'):
        model.freeze_backbone()
        print(f"Backbone frozen for first {freeze_epochs} epochs")

    backbone_lr = config["learning_rate"] * 0.1
    if hasattr(model, 'get_head_params'):
        param_groups = [
            {"params": model.get_head_params(), "lr": config["learning_rate"]},
            {"params": model.get_backbone_params(), "lr": backbone_lr},
        ]
    else:
        param_groups = model.parameters()

    optimizer = AdamW(param_groups, lr=config["learning_rate"],
                     weight_decay=config["weight_decay"])

    total_steps = len(train_loader) * config["num_epochs"] // config["gradient_accumulation_steps"]
    scheduler = OneCycleLR(optimizer, max_lr=[config["learning_rate"], backbone_lr],
                          total_steps=total_steps, pct_start=config["warmup_ratio"])

    scaler = GradScaler(enabled=(device=="cuda"))

    best_auc = 0
    patience_counter = 0
    early_stopping_patience = config.get("early_stopping_patience", 10)

    history = {"train_loss": [], "val_auc": [], "val_acc": [], "val_convergence_delta": [],
               "val_reasoning_auc": [], "val_gating_alpha": []}

    for epoch in range(config["num_epochs"]):
        if epoch == freeze_epochs and freeze_epochs > 0 and hasattr(model, 'unfreeze_backbone'):
            model.unfreeze_backbone()
            print(f"Backbone unfrozen at epoch {epoch + 1}")

        epoch_start = time.time()

        train_losses, skipped, grad_norm = train_epoch(model, train_loader, optimizer, scheduler,
                                                          loss_fn, scaler, device, config["gradient_accumulation_steps"], config)
        val_metrics = evaluate(model, val_loader, device,
                              use_badm=config.get("use_badm", True),
                              use_aadm=config.get("use_aadm", True))

        history["train_loss"].append(train_losses["total"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_convergence_delta"].append(val_metrics["convergence_delta"])
        history["val_reasoning_auc"].append(val_metrics["reasoning_auc"])
        history["val_gating_alpha"].append(val_metrics["gating_alpha"])

        print(f"Epoch {epoch+1}/{config['num_epochs']}: "
              f"Loss={train_losses['total']:.4f}, AUC={val_metrics['auc']:.4f}, "
              f"ReasoningAUC={val_metrics['reasoning_auc']:.4f}, "
              f"a={val_metrics['gating_alpha']:.3f}, "
              f"ConvDelta={val_metrics['convergence_delta']:.4f}, "
              f"Time={time.time()-epoch_start:.1f}s")

        if skipped > 0:
            print(f"  Skipped {skipped} batches with non-finite loss")

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            patience_counter = 0
            checkpoint_path = checkpoint_dir / "best.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "auc": best_auc,
            }, checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    return history, best_auc
