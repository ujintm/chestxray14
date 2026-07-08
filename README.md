# ChestX-ray Multi-label Disease Classification

NIH ChestX-ray14 데이터셋을 활용한 흉부 X-ray 다중 질환 분류 프로젝트입니다.
단일 이미지에서 14개 질환 소견을 동시에 예측하는 multi-label classification 모델을 구축하고,
클래스 불균형 문제에 대응하기 위한 다양한 기법(Focal Loss, pos_weight, per-class threshold 등)을 실험했습니다.

## 프로젝트 목표

- **데이터셋**: 112,120장의 X-ray 이미지, 14개 질환 라벨, 30,805명의 고유 환자 (NIH ChestX-ray14)
- **목표**: 14개 질병 소견을 동시에 예측하는 다중 라벨 분류 모델 구축
- **도전 과제**: 대부분의 이미지가 정중앙 정렬되어 있고 뼈대 구조가 거의 동일하여, 작은 병변(lesion)의 위치·밝기 차이만으로 진단이 갈리는 세밀한 분류 문제

## 데이터셋 특징: 극심한 클래스 불균형

| Combination | Number of Cases | Percentage |
|---|---|---|
| No Finding | 60,361 | 53.84% |
| Infiltration | 9,547 | 8.51% |
| Atelectasis | 4,215 | 3.76% |
| Effusion | 3,955 | 3.53% |
| Nodule | 2,705 | 2.41% |
| Pneumothorax | 2,194 | 1.96% |
| Mass | 2,139 | 1.91% |
| ... | ... | ... |

- 이미지 1장에 최대 14개 질병이 동시에 나타날 수 있어 이론적으로 2¹⁴ = 16,384가지 조합 가능
- 실제 데이터셋에는 836개 조합만 존재하며, `No Finding`이 약 50%, `Infiltration`이 약 8%, 나머지 800여 개 조합은 각각 4% 미만으로 매우 희귀

## 환경 설정

- **GPU**: Vast.ai에서 대여한 RTX 4070Ti 서버 (1x RTX 4070S Ti, 42.0 TFLOPS)
- **RAM**: 16GB / **Storage**: 99GB SSD
- **데이터 관리**: Hugging Face Datasets를 통해 캐시된 데이터를 디스크에서 효율적으로 관리

## 모델 구조

### ResNet50
- 2개 이상의 Convolutional Layer와 skip-connection을 활용해 하나의 Residual Block으로 구성
- VGG-19 구조를 뼈대로 컨볼루션 층을 추가해 깊게 만든 후 shortcut 연결 추가

### DenseNet121
- 모든 이전 레이어의 output을 이후 레이어의 input으로 concatenation하여 받아오는 구조
- 이어붙인 connection들의 덩어리를 하나의 Dense Block으로 구성

## 평가 지표

- **AUC (macro)**: ROC curve(FPR-TPR) 아래 면적. Positive/negative class 구별 능력 측정
- **F1 (macro)**: Precision/Recall의 조화평균. 양성 클래스 각각을 얼마나 정확히 찾아냈는지 측정

## 실험 과정

| Version | Backbone | Loss | Threshold | AUC | F1 | 비고 |
|---|---|---|---|---|---|---|
| v1 | ResNet50 (freeze) | BCEWithLogitsLoss | global 0.5 | 0.73 | 0.07 | ImageNet 사전학습 feature만으로는 흉부 X-ray 도메인 표현에 한계 |
| v2~v4 | ResNet50 (unfreeze) | BCE + pos_weight | per-class sweep (0.05~0.95) | 0.75~0.80 | 0.13~0.27 | pos_weight로 클래스 불균형 대응, threshold sweep 도입 |
| v5 | ResNet+DenseNet 앙상블 (soft voting) | Focal Loss (γ=1.5, α=0.4) | per-class | 0.82 | 0.14 | 모델이 헷갈리는 양성 샘플에 집중, 단 두 모델 성향이 유사해 앙상블 효과 미미 |
| v6 (DannyNet) | DenseNet121 (unfreeze) | Focal Loss + AdamW | per-class | 0.85 | 0.39 | CheXNet 논문 재현 및 개선 (Reproducing and Improving CheXNet 참고) |
| v6 (논문 세팅 재현) | DenseNet121 (unfreeze) | Focal Loss (γ=2, α=1/freq) | per-class | 0.84 | 0.18 | 논문 설정 그대로 alpha=1 실험 시 collapse 현상 발생 |

### 주요 실험 인사이트
1. **Backbone freeze → unfreeze**: ImageNet은 일반 사물(강아지, 고양이, 컵 등) 분류 데이터셋이라 흉부 X-ray와 도메인이 달라, 사전학습 feature만으로는 세밀한 병변 구분이 어려움
2. **클래스 불균형 대응**: `pos_weight` 적용 및 0.05~0.95 범위의 threshold sweep으로 ResNet50에서 얻을 수 있는 성능을 최대한 끌어냄
3. **Focal Loss 도입**: 모델이 헷갈려하는 샘플에 집중하도록 유도, per-class threshold 방식으로 전환
4. **DannyNet (최고 성능)**: DenseNet121 전체 unfreeze + Focal Loss + AdamW + per-class threshold 조합으로 AUC 0.85, F1 0.39 달성 (baseline 대비 F1 약 5배 개선)
5. **논문 설정 그대로 재현 시 한계**: patient-level split이 아닌 이미지 단위 8:2 랜덤 분할, alpha=1 설정 시 collapse 현상 확인 → 하이퍼파라미터 선택의 민감도 확인

## GradCam 시각화

`gradcam.py`를 통해 모델이 특정 클래스를 예측할 때 어떤 이미지 영역을 중요하게 보는지 시각화했습니다 (`results/logs/v4_앙상블_결과.png` 참고).

## 회고 및 결론

### 한계
- 다양한 기법을 적용해도 F1 score는 0.18~0.39 수준에 그침
- 주요 원인으로 추정: 데이터의 클래스 불균형, 멀티라벨 특성, 라벨 노이즈

### 지표 간 Trade-off
- F1/Sensitivity가 오르면 Accuracy는 떨어짐 (양성에 민감하게 반응한 대가로 전체 정확도 감소)
- 반대로 Accuracy/Specificity가 높으면 Sensitivity가 매우 낮음 (음성은 잘 잡지만 양성을 놓침)

### 향후 방향
- 흉부 X-ray에 특화된 사전학습 가중치로 전이학습 시도 (CheXNet, TorchXRayVision 등)
- 클래스별 학습을 분리하는 방식 도입
- 더 정밀한 threshold 학습 방법 적용
- 더 복잡한 구조의 모델 활용

## 참고 문헌
- Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases" (NIH ChestX-ray14 데이터셋)
- Strick, Garcia, Huang, "Reproducing and Improving CheXNet: Deep Learning for Chest X-ray Disease Classification" (2025)
- He et al., "Deep Residual Learning for Image Recognition" (ResNet)
- Huang et al., "Densely Connected Convolutional Networks" (DenseNet)

---

[ChestX-ray 다중 질환 분류 모델.pdf](https://github.com/user-attachments/files/22056477/ChestX-ray.pdf)
