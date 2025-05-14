# PyTorch 공식 튜토리얼(transfer learning)을 기반으로, 
# ResNet50 + pretrained + freeze + AMP 학습 구조로 구성

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from models.resnet50 import get_resnet50
import time
import copy

# ==== GPU 설정 ====
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==== 데이터 전처리 ====
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
}

# (예시용 폴더 구조)
data_dir = 'data/images'  # NIH 데이터셋 압축 푼 경로에 맞게 수정
image_datasets = {x: datasets.ImageFolder(root=f'{data_dir}/{x}',
                                          transform=data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                              shuffle=True, num_workers=4)
              for x in ['train', 'val']}

model = get_resnet50(num_classes=14, freeze_backbone=True)
model = model.to(device)

# ==== 손실함수, 옵티마이저, 스케줄러 ====
criterion = nn.BCEWithLogitsLoss()  # 멀티라벨 분류
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ==== AMP 세팅 ====
scaler = torch.cuda.amp.GradScaler()

# ==== 학습 루프 ====
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = float('inf')

epochs = 10
for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    print('-' * 10)

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

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if phase == 'train':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(image_datasets[phase])
        print(f'{phase} Loss: {epoch_loss:.4f}')

        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    scheduler.step()

# ==== 모델 저장 ====
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), 'best_resnet50.pth')
