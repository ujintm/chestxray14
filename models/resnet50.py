import torch.nn as nn
from torchvision import models

def get_resnet50(num_classes=14, freeze_backbone=True):
    # 1. pretrained ResNet50 로드
    model = models.resnet50(pretrained=True)

    # 2. feature extractor(freeze) 설정
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # 3. 마지막 classifier 수정 (NIH는 14개 질병 멀티라벨)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model
