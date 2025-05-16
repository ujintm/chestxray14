import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from hf_chestxray_dataset import HFChestXrayDataset
from models.resnet50 import get_resnet50
from multilabel_metrics import compute_metrics


# 데이터 전처리 정의
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Hugging Face 데이터셋 로드
ds_train = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", "image-classification", split="train[:80%]", trust_remote_code=True)
ds_val = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", "image-classification", split="train[80%:]", trust_remote_code=True)

# Dataset 래핑
train_dataset = HFChestXrayDataset(ds_train, transform=data_transforms['train'])
val_dataset = HFChestXrayDataset(ds_val, transform=data_transforms['val'], label_map=train_dataset.label_map)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
}

dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset)
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = get_resnet50(num_classes=train_dataset.num_classes, freeze_backbone=True)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler()

def train_model(model, criterion, optimizer, num_epochs=10, checkpoint_dir='checkpoints', start_epoch=0):
    os.makedirs(checkpoint_dir, exist_ok=True)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}\n' + '-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'val':
                all_preds = []
                all_labels = []

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            outputs = torch.sigmoid(outputs)

                    all_preds.append(outputs.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())

                y_pred = np.concatenate(all_preds)
                y_true = np.concatenate(all_labels)

                metrics = compute_metrics(y_true, y_pred, threshold=0.5)
                print("📊 Validation metrics:")
                for k, v in metrics.items():
                    print(f"{k}: {v:.4f}")
            
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1:02d}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'scaler': scaler.state_dict()
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_loss:.4f}')

    model.load_state_dict(best_model_wts)
    return model

# 이어서 학습을 위한 체크포인트 로드
resume_path = 'checkpoint_interrupted.pth'
start_epoch = 0

if os.path.exists(resume_path):
    print(f"🔁 Resuming from checkpoint: {resume_path}")
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler'])
    start_epoch = checkpoint['epoch']
    print(f"➡️ Resuming from epoch {start_epoch}")

# 학습 실행 (try-except 포함)
try:
    model = train_model(model, criterion, optimizer, num_epochs=10, checkpoint_dir='checkpoints', start_epoch=start_epoch)
except KeyboardInterrupt:
    print("⛔ Interrupted! Saving checkpoint...")
    torch.save({
        'epoch': start_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': None,
        'scaler': scaler.state_dict()
    }, 'checkpoint_interrupted.pth')
    print("✅ Saved to checkpoint_interrupted.pth")

# 최종 모델 저장
torch.save(model.state_dict(), 'best_resnet50.pth')
