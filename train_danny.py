import os, csv, copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

from hf_chestxray_dataset import HFChestXrayDataset
from models.densenet121 import get_densenet121
from multilabel_metrics import compute_metrics

# ---------- CONFIG ----------
CONFIG = {
    "batch_size"   : 8,
    "epochs"       : 25,
    "image_size"   : 224,
    "learning_rate": 5e-5,
    "num_workers"  : 2,
    "device"       : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "out_dir"      : "checkpoints_clean",
    "patience"     : 5,   # early‑stopping patience (val loss)
}
Path(CONFIG["out_dir"]).mkdir(parents=True, exist_ok=True)

# ---------- DATA ----------
aug_train = transforms.Compose([
    transforms.RandomResizedCrop(CONFIG["image_size"]),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
aug_val = transforms.Compose([
    transforms.Resize(CONFIG["image_size"] + 32),
    transforms.CenterCrop(CONFIG["image_size"]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print("[+] Loading HuggingFace NIH Chest‑Xray14 split …")
ds_train = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", "image-classification", split="train[:80%]", trust_remote_code=True)
ds_val   = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", "image-classification", split="train[80%:]", trust_remote_code=True)

train_set = HFChestXrayDataset(ds_train, transform=aug_train)
val_set   = HFChestXrayDataset(ds_val,   transform=aug_val, label_map=train_set.label_map)

# Optionally drop No‑Finding rows (논문과 동일)
mask_train = train_set.labels.sum(axis=1) > 0
mask_val   = val_set.labels.sum(axis=1) > 0
train_set.labels = train_set.labels[mask_train]
train_set.dataset = train_set.dataset.select(np.where(mask_train)[0])
val_set.labels   = val_set.labels[mask_val]
val_set.dataset  = val_set.dataset.select(np.where(mask_val)[0])

print(f"  train | {len(train_set):6d} images")
print(f"  val   | {len(val_set):6d} images")

loader = {
    "train": DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=CONFIG["num_workers"], pin_memory=True),
    "val"  : DataLoader(val_set,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True),
}
num_classes = train_set.num_classes

# ---------- MODEL ----------
model = get_densenet121(num_classes, freeze_backbone=False).to(CONFIG["device"])

class FocalLoss(nn.Module):
    """Multi-label focal loss (BCE with logits)."""
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt       = torch.exp(-bce_loss)
        focal    = self.alpha * (1. - pt) ** self.gamma * bce_loss
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal

criterion = FocalLoss(alpha=1.0, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=1, factor=0.1)

# ---------- METRIC CSV ----------
csv_path = Path(CONFIG["out_dir"]) / "metrics.csv"
if not csv_path.exists():
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "val_loss", "thr_mode", "accuracy", "f1_macro", "auc_macro"])

# ---------- TRAIN ----------
scaler = torch.cuda.amp.GradScaler()  # can be kept even on CPU (no‑op)
best_acc, best_wts, patience_cnt = 0.0, None, 0

for epoch in range(CONFIG["epochs"]):
    print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
    for phase in ("train", "val"):
        model.train() if phase == "train" else model.eval()
        running_loss, preds_all, labels_all = 0.0, [], []

        for x, y in loader[phase]:
            x = x.to(CONFIG["device"])
            y = y.to(CONFIG["device"])
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss   = criterion(logits, y)

                if phase == "train":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            if phase == "val":
                preds_all.append(torch.sigmoid(logits).detach().cpu().numpy())
                labels_all.append(y.cpu().numpy())

            running_loss += loss.item() * x.size(0)

        epoch_loss = running_loss / len(loader[phase].dataset)
        if phase == "val":
            scheduler.step(epoch_loss)
        print(f"  {phase:5s} | loss: {epoch_loss:.4f}")

        # ---- validation metrics & checkpoint ----
        if phase == "val":
            y_pred = np.concatenate(preds_all)
            y_true = np.concatenate(labels_all)

            # per‑class optimal threshold (0.05–0.95 step 0.01)
            grid   = np.arange(0.05, 0.951, 0.01)
            best_t = np.zeros(num_classes)
            for k in range(num_classes):
                best_t[k] = max(grid, key=lambda t: ((y_pred[:, k] > t) == y_true[:, k]).mean())

            metrics = compute_metrics(y_true, y_pred, threshold=best_t)
            acc     = metrics["accuracy"]
            print(f"        accuracy: {acc:.4f}")

            # log CSV
            with open(csv_path, "a", newline="") as f:
                row = [epoch + 1, f"{epoch_loss:.4f}", "per‑class"] + [f"{metrics[k]:.4f}" for k in ("accuracy", "f1_macro", "auc_macro")]
                csv.writer(f).writerow(row)

            # early stop & best checkpoint
            if acc > best_acc:
                best_acc, best_wts = acc, copy.deepcopy(model.state_dict())
                torch.save(best_wts, Path(CONFIG["out_dir"]) / "best_model.pth")
                np.save(Path(CONFIG["out_dir"]) / "best_thresh.npy", best_t)
                print("New best saved!")
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= CONFIG["patience"]:
                    print("\nEarly-stopping triggered.")
                    model.load_state_dict(best_wts)
                    print(f"[Done] best accuracy = {best_acc:.4f}")
                    exit()

print(f"\n[Done] best accuracy = {best_acc:.4f}")
