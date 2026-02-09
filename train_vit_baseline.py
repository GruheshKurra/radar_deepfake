#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from baselines.vit import ViTBaseline
from data.dataset import DeepfakeDataset, get_train_transforms, get_val_transforms
from data.splits import load_presplit_data
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import json

print("="*70)
print("ViT-Small Baseline Training")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = Path("./data")

train_images, train_labels = load_presplit_data(data_dir, "train")
val_images, val_labels = load_presplit_data(data_dir, "val")

train_dataset = DeepfakeDataset(train_images, train_labels,
                                get_train_transforms(224))
val_dataset = DeepfakeDataset(val_images, val_labels,
                              get_val_transforms(224))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                          num_workers=6, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False,
                        num_workers=6, pin_memory=True)

model = ViTBaseline(pretrained=True).to(device)
optimizer = AdamW(model.parameters(), lr=0.0005, weight_decay=0.05)
scheduler = OneCycleLR(optimizer, max_lr=0.0005,
                       total_steps=len(train_loader) * 15,
                       pct_start=0.1)
criterion = torch.nn.BCEWithLogitsLoss()
scaler = GradScaler()

best_auc = 0
history = []

for epoch in range(15):
    model.train()
    train_loss = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/15"):
        images, labels = images.to(device), labels.to(device).float()

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs['logit'].squeeze(), labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss += loss.item()

    model.eval()
    val_probs = []
    val_labels_list = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            val_probs.extend(outputs['prob'].cpu().numpy().flatten())
            val_labels_list.extend(labels.numpy())

    val_auc = roc_auc_score(val_labels_list, val_probs)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, "
          f"Val AUC = {val_auc:.4f}")

    history.append({'epoch': epoch+1, 'val_auc': val_auc})

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), './outputs/vit_baseline_best.pth')
        print(f"  → New best: {best_auc:.4f}")

print("\n" + "="*70)
print(f"CRITICAL RESULT: ViT-Small Baseline AUC = {best_auc:.4f}")
print("="*70)

if best_auc >= 0.93:
    print("\n⚠️  ViT alone >= 93% → ViT is doing most of the work!")
    print("    Your RADAR components add minimal value")
    print("    This explains why ablations don't show differences")
else:
    print("\n✅ ViT alone < 93% → RADAR components add real value!")
    print(f"   RADAR (94%) - ViT ({best_auc:.1%}) = {94 - best_auc*100:.1f}% improvement")

with open('./outputs/vit_baseline_history.json', 'w') as f:
    json.dump(history, f, indent=2)
