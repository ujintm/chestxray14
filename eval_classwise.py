import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from hf_chestxray_dataset import HFChestXrayDataset
from models.resnet50 import get_resnet50
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score

# ---------- 설정 ----------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "checkpoints_final/best_model.pth"

# ---------- 데이터 로딩 ----------
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_ds = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", "image-classification", split="train[80%:]", trust_remote_code=True)
val_set = HFChestXrayDataset(val_ds, transform=val_tf)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)

num_classes = val_set.num_classes
model = get_resnet50(num_classes, freeze_backbone=False).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ---------- 추론 ----------
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_pred.append(probs)
        y_true.append(y.numpy())

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

# ---------- 클래스별 지표 ----------
grid = np.arange(0.05, 0.951, 0.01)
best_thresh = np.zeros(num_classes)
for k in range(num_classes):
    best_thresh[k] = max(grid, key=lambda t: ((y_pred[:, k] > t) == y_true[:, k]).mean())

print("\n📊 클래스별 성능 지표:")
for k in range(num_classes):
    y_t = y_true[:, k]
    y_p = (y_pred[:, k] > best_thresh[k]).astype(int)

    try:
        f1 = f1_score(y_t, y_p)
        sens = recall_score(y_t, y_p)
        spec = recall_score(1 - y_t, 1 - y_p)
        auc = roc_auc_score(y_t, y_pred[:, k])
    except:
        f1, sens, spec, auc = [np.nan] * 4

    print(f"Class {k:02d} | F1: {f1:.3f}  Sensitivity: {sens:.3f}  Specificity: {spec:.3f}  AUC: {auc:.3f}")
