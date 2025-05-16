
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix

def compute_metrics(y_true, y_pred_prob, threshold=0.5):
    # 확률 → 예측 이진 벡터
    y_pred = (y_pred_prob >= threshold).astype(int)

    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['auc_macro'] = roc_auc_score(y_true, y_pred_prob, average='macro')

    # sensitivity, specificity per class
    sensitivities = []
    specificities = []

    for i in range(y_true.shape[1]):
        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]
        tn, fp, fn, tp = confusion_matrix(y_true_i, y_pred_i, labels=[0, 1]).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

        sensitivities.append(sensitivity)
        specificities.append(specificity)

    metrics['sensitivity_macro'] = np.mean(sensitivities)
    metrics['specificity_macro'] = np.mean(specificities)

    return metrics
