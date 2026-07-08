import os, copy, csv
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from hf_chestxray_dataset import HFChestXrayDataset
from models.resnet50 import get_resnet50
from multilabel_metrics import compute_metrics          # v2 그대로 사용
from torch.cuda.amp import GradScaler, autocast

# 1. 변형 --------------------------------------------------
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2),
        transforms.RandomAffine(0, translate=(.05, .05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# 2. 데이터 -------------------------------------------------
ds_train = load_dataset("alkzar90/NIH-Chest-X-ray-dataset",
                        "image-classification",
                        split="train[:80%]", trust_remote_code=True)
ds_val   = load_dataset("alkzar90/NIH-Chest-X-ray-dataset",
                        "image-classification",
                        split="train[80%:]", trust_remote_code=True)

train_dataset = HFChestXrayDataset(ds_train,
                                   transform=data_transforms['train'])
val_dataset   = HFChestXrayDataset(ds_val,
                                   transform=data_transforms['val'],
                                   label_map=train_dataset.label_map)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=32,
                        shuffle=True,  num_workers=4),
    'val'  : DataLoader(val_dataset,   batch_size=32,
                        shuffle=False, num_workers=4)
}
dataset_sizes = {k: len(v) for k, v in {'train': train_dataset,
                                        'val': val_dataset}.items()}

# 3. 모델 --------------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model  = get_resnet50(num_classes=train_dataset.num_classes,
                      freeze_backbone=False).to(device)

# --- pos_weight 계산 ### NEW ###
def pos_weight_from(loader, n_cls):
    pos = torch.zeros(n_cls)
    neg = torch.zeros(n_cls)
    for _, y in loader:
        pos += y.sum(0)
        neg += (1 - y).sum(0)
    w = neg / (pos + 1e-6)
    return torch.clamp(w, max=20)        # 폭주 방지

pos_weight = pos_weight_from(dataloaders['train'],
                             train_dataset.num_classes).to(device)
criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer   = optim.Adam(model.parameters(), lr=1e-3)
scaler      = GradScaler()

# 4. 학습 루프 ---------------------------------------------
num_epochs = 30
best_acc, best_wts = 0., copy.deepcopy(model.state_dict())
ckpt_dir = 'checkpoints_v3'; os.makedirs(ckpt_dir, exist_ok=True)

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}\n' + '-'*10)
    for phase in ['train', 'val']:
        model.train() if phase == 'train' else model.eval()
        run_loss, preds_all, labels_all = 0.0, [], []

        for x, y in dataloaders[phase]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase=='train'):
                with autocast():
                    out  = model(x)
                    loss = criterion(out, y)

                    if phase == 'val':
                        preds_all.append(torch.sigmoid(out).cpu().numpy())
                        labels_all.append(y.cpu().numpy())

                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            run_loss += loss.item() * x.size(0)

        epoch_loss = run_loss / dataset_sizes[phase]
        print(f'{phase} Loss: {epoch_loss:.4f}')

        if phase == 'val':
            y_pred = np.concatenate(preds_all)
            y_true = np.concatenate(labels_all)

            # --- threshold sweep ### NEW ###
            best_t, best_epoch_acc, best_metrics = 0.5, 0., None
            for t in np.arange(0.05, 0.95, 0.05):
                m = compute_metrics(y_true, y_pred, threshold=t)
                if m['accuracy'] > best_epoch_acc:
                    best_epoch_acc, best_t, best_metrics = m['accuracy'], t, m

            print(f'📊 Validation (thr={best_t:.2f})')
            for k, v in best_metrics.items():
                print(f'{k}: {v:.4f}')

            # csv 로그 (v2 방식 유지)
            csv_path = os.path.join(ckpt_dir, 'metrics.csv')
            header   = not os.path.exists(csv_path)
            with open(csv_path, 'a', newline='') as f:
                w = csv.writer(f)
                if header:
                    w.writerow(['epoch', 'val_loss', 'thr'] + list(best_metrics.keys()))
                w.writerow([epoch+1, f'{epoch_loss:.4f}', f'{best_t:.2f}'] +
                           [f'{v:.4f}' for v in best_metrics.values()])

            # 매 에포크 체크포인트 (v2 동일)
            ckpt = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch+1:02d}.pth')
            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss,
                        'threshold': best_t,
                        'scaler': scaler.state_dict()}, ckpt)
            print(f'Checkpoint saved ➜ {ckpt}')

            # best accuracy 갱신
            if best_epoch_acc > best_acc:
                best_acc, best_wts = best_epoch_acc, copy.deepcopy(model.state_dict())

# 5. 최종 best 모델 저장 ------------------------------------
best_path = os.path.join(ckpt_dir, 'best_model.pth')
torch.save(best_wts, best_path)
print(f'Best ACC={best_acc:.4f}  (saved to {best_path})')
