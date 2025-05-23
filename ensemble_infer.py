import argparse, numpy as np, torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from hf_chestxray_dataset import HFChestXrayDataset
from models.resnet50    import get_resnet50
from models.densenet121 import get_densenet121
from multilabel_metrics import compute_metrics

p = argparse.ArgumentParser()
p.add_argument("--resnet_ckpt", required=True)
p.add_argument("--densenet_ckpt", required=True)
p.add_argument("--resnet_thresh", required=True)
p.add_argument("--densenet_thresh", required=True)
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- data (val split) -------
aug = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
ds_val = load_dataset("alkzar90/NIH-Chest-X-ray-dataset",
                      "image-classification",
                      split="train[80%:]", trust_remote_code=True)
val_set = HFChestXrayDataset(ds_val, transform=aug)
val_loader = DataLoader(val_set, batch_size=64,
                        shuffle=False,num_workers=4)
num_classes = val_set.num_classes

# -------- load models ----------
resnet = get_resnet50(num_classes, False).to(device)
densenet = get_densenet121(num_classes, False).to(device)
resnet.load_state_dict(torch.load(args.resnet_ckpt, map_location=device))
densenet.load_state_dict(torch.load(args.densenet_ckpt, map_location=device))
resnet.eval(); densenet.eval()

t_res = np.load(args.resnet_thresh); t_den = np.load(args.densenet_thresh)
t_ens = (t_res + t_den) / 2          # 간단히 평균 threshold

# -------- inference -------------
preds, labels = [], []
with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        p1 = torch.sigmoid(resnet(x))
        p2 = torch.sigmoid(densenet(x))
        p  = (p1 + p2) / 2           # 앙상블 확률
        preds.append(p.cpu().numpy())
        labels.append(y.numpy())

preds  = np.concatenate(preds)
labels = np.concatenate(labels)

# -------- global threshold sweep (accuracy 기준) ----------
grid = np.arange(0.05, 0.951, 0.01)
best_t = 0.5
best_acc = 0.0

for t in grid:
    preds_bin = (preds > t).astype(int)
    acc = (preds_bin == labels).mean()      # 전체 element-wise 정확도
    if acc > best_acc:
        best_acc = acc
        best_t = t

print(f"🔍 Best global threshold: {best_t:.2f} (accuracy={best_acc:.4f})")

# 최종 metric 계산
metrics = compute_metrics(labels, preds, threshold=best_t)
np.save("ensemble_best_thresh_global.npy", np.array([best_t]))
print("Ensemble metrics:")
for k,v in metrics.items():
    print(f"{k}: {v:.4f}")
