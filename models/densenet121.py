from torchvision import models
import torch.nn as nn


def get_densenet121(num_classes: int,
                    freeze_backbone: bool = True,
                    pretrained: bool = True) -> nn.Module:
    """
    DenseNet-121 생성 함수.

    Args:
        num_classes (int): 최종 출력 클래스 수.
        freeze_backbone (bool): True면 features 파라미터를 모두 고정.
        pretrained (bool): ImageNet 사전학습 weight 로드 여부.

    Returns:
        nn.Module: 분류기(head)만 교체한 DenseNet-121 모델.
    """
    if pretrained:
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    else:
        model = models.densenet121(weights=None)

    # 백본 고정 옵션
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    # classifier(head) 교체
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model