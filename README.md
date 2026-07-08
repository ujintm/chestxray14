# ChestX-ray Multi-label Disease Classification

NIH ChestX-ray14 데이터셋을 활용한 흉부 X-ray 다중 질환 분류 프로젝트입니다.  
ResNet50, DenseNet121 등 기존 backbone을 가져와 multi-label classification에 적용해보고,  
극심한 클래스 불균형 문제에 어떤 기법(Focal Loss, pos_weight, per-class threshold 등)이 효과가 있는지 실험하며 원인을 진단해본 프로젝트입니다.


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

| Version | Backbone | Loss | Threshold | AUC | F1 |
|---|---|---|---|---|---|
| v1 | ResNet50 (freeze) | BCEWithLogitsLoss | global 0.5 | 0.73 | 0.07 |
| v2~v4 | ResNet50 (unfreeze) | BCE + pos_weight | per-class sweep (0.05~0.95) | 0.75~0.80 | 0.13~0.27 |
| v5 | ResNet+DenseNet 앙상블 (soft voting) | Focal Loss (γ=1.5, α=0.4) | per-class | 0.82 | 0.14 |
| v6 | DenseNet121 (unfreeze) | Focal Loss (γ=2, α=1/freq) | per-class | 0.84 | 0.18 |

**버전별 변경 사항 및 분석**
- **v1**: ImageNet 사전학습 feature만으로는 흉부 X-ray 도메인 표현에 한계 → backbone unfreeze 필요성 확인
- **v2~v4**: `pos_weight`로 클래스 불균형 대응, threshold sweep(0.05~0.95) 도입 → ResNet50에서 얻을 수 있는 성능을 최대한 끌어냄
- **v5**: Focal Loss로 헷갈리는 양성 샘플에 집중, per-class threshold 도입. ResNet+DenseNet 앙상블 시도했으나 두 모델 성향이 유사해 앙상블 효과는 미미
- **v6**: 논문 세팅을 최대한 재현. 전처리에서 No Finding 제거, patient-level split 대신 이미지 단위 8:2 랜덤 분할 적용. alpha=1로 실험 시 collapse 현상 발생 확인. 참고 논문의 보고 AUC(0.85)에 근접한 0.84를 달성하여 재현이 유의미하다고 판단, 해당 지점에서 실험을 마무리

### 주요 실험 인사이트
1. **Backbone freeze → unfreeze**: ImageNet은 일반 사물(강아지, 고양이, 컵 등) 분류 데이터셋이라 흉부 X-ray와 도메인이 달라, 사전학습 feature만으로는 세밀한 병변 구분이 어려움
2. **클래스 불균형 대응**: `pos_weight` 적용 및 threshold sweep으로 ResNet50에서 얻을 수 있는 성능을 최대한 끌어냄
3. **Focal Loss 도입**: 모델이 헷갈려하는 샘플에 집중하도록 유도, per-class threshold 방식으로 전환
4. **하이퍼파라미터 민감도**: Focal Loss의 alpha 설정에 따라 collapse 현상이 발생할 수 있음을 확인, 데이터 split 방식(patient-level vs image-level)이 성능에 영향을 줄 수 있음을 인지
5. **논문 재현 완료 판단**: v6에서 AUC 0.84로 참고 논문의 보고치(0.85)에 근접한 결과를 확인, 재현이 유의미하다고 판단하여 이 시점에서 추가 실험을 마무리

## GradCam 시각화
<img width="758" height="412" alt="GradCam" src="https://github.com/user-attachments/assets/71a11146-badb-4990-90c7-c881cdadeb62" />

`gradcam.py`를 통해 모델이 특정 클래스를 예측할 때 어떤 이미지 영역을 중요하게 보는지 시각화했습니다.

시각화 결과, class 1과 class 6의 heatmap이 서로 다른 병변 위치가 아닌 **유사한 영역(좌측 폐 하단, 심장 경계 부근)에 공통적으로 집중**되는 경향이 관찰되었습니다.  
이는 모델이 클래스별로 구별되는 병변 특징을 학습했다기보다, 이미지 내에서 두드러지는 공통 영역에 의존하고 있을 가능성을 시사합니다.  
이 결과는 F1 score가 낮게 나온 원인 중 하나로, 모델이 병변의 위치를 세밀하게 구분하지 못했음을 뒷받침하는 근거로 해석했습니다.

## 회고 및 결론

이 프로젝트는 결과적으로 완성도 높은 모델을 만들었다기보다, **클래스 불균형이 심한 멀티라벨 문제에서 무엇이 어떻게 실패하는지를 직접 진단하고 원인을 좁혀나간 실험 과정**에 의의가 있습니다.

### 한계
- 다양한 기법(pos_weight, Focal Loss, per-class threshold, 앙상블 등)을 적용했음에도 F1 score는 0.07~0.39 수준에 그침
- GradCam 시각화 결과, 서로 다른 클래스임에도 유사한 영역에 attention이 집중되는 현상이 관찰되어, 모델이 병변별 특징을 세밀하게 구분하지 못했을 가능성을 확인
- 주요 원인으로 추정: 데이터의 클래스 불균형, 멀티라벨 특성, 라벨 노이즈, 병변 간 시각적 유사성(작은 영역 차이)

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
