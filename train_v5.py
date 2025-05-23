import os, copy, csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from hf_chestxray_dataset import HFChestXrayDataset
from models.resnet50 import get_resnet50
from multilabel_metrics import compute_metrics

# ---------- 설정 ----------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ckpt_dir = "checkpoints_final"
os.makedirs(ckpt_dir, exist_ok=True)

# ---------- 전처리 ----------
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2),
        transforms.RandomAffine(0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# ---------- 데이터 ----------
ds_train = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", "image-classification", split="train[:80%]", trust_remote_code=True)
ds_val = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", "image-classification", split="train[80%:]", trust_remote_code=True)

train_dataset = HFChestXrayDataset(ds_train, transform=data_transforms['train'])
val_dataset = HFChestXrayDataset(ds_val, transform=data_transforms['val'], label_map=train_dataset.label_map)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
}

dataset_sizes = { 'train': len(train_dataset), 'val': len(val_dataset) }

# ---------- 모델 ----------
model = get_resnet50(train_dataset.num_classes, freeze_backbone=False).to(device)

# ---------- pos_weight ----------
def calc_pos_weight(loader, n_cls):
    pos = torch.zeros(n_cls)
    neg = torch.zeros(n_cls)
    for _, y in loader:
        pos += y.sum(0)
        neg += (1 - y).sum(0)
    w = neg / (pos + 1e-6)
    return torch.clamp(w, max=20)

pos_weight = calc_pos_weight(dataloaders['train'], train_dataset.num_classes).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scaler = GradScaler()

# ---------- 학습 ----------
best_acc, best_wts = 0., copy.deepcopy(model.state_dict())
csv_path = os.path.join(ckpt_dir, "metrics.csv")

with open(csv_path, 'w', newline='') as f:
    csv.writer(f).writerow(['epoch', 'val_loss', 'thr', 'accuracy', 'f1_macro', 'auc_macro'])

for epoch in range(30):
    print(f"\nEpoch {epoch+1}/30")
    for phase in ['train', 'val']:
        model.train() if phase == 'train' else model.eval()
        run_loss, preds_all, labels_all = 0.0, [], []

        for x, y in dataloaders[phase]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                with autocast():
                    out = model(x)
                    loss = criterion(out, y)

                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            if phase == 'val':
                preds_all.append(torch.sigmoid(out).cpu().numpy())
                labels_all.append(y.cpu().numpy())

            run_loss += loss.item() * x.size(0)

        epoch_loss = run_loss / dataset_sizes[phase]
        print(f"{phase} Loss: {epoch_loss:.4f}")

        if phase == 'val':
            y_pred = np.concatenate(preds_all)
            y_true = np.concatenate(labels_all)

            best_t, best_acc_epoch, best_metrics = 0.5, 0., None
            for t in np.arange(0.05, 0.95, 0.05):
                m = compute_metrics(y_true, y_pred, threshold=t)
                if m['accuracy'] > best_acc_epoch:
                    best_acc_epoch, best_t, best_metrics = m['accuracy'], t, m

            print(f"📊 Validation (thr={best_t:.2f})")
            for k, v in best_metrics.items():
                print(f"{k}: {v:.4f}")

            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([epoch+1, f"{epoch_loss:.4f}", f"{best_t:.2f}"] + [f"{best_metrics[k]:.4f}" for k in ['accuracy','f1_macro','auc_macro']])

            if best_acc_epoch > best_acc:
                best_acc = best_acc_epoch
                best_wts = copy.deepcopy(model.state_dict())
                torch.save(best_wts, os.path.join(ckpt_dir, "best_model.pth"))
                print("★ New best saved!")

print(f"\n[Done] best accuracy = {best_acc:.4f}")
