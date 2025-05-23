import argparse, numpy as np, torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from hf_chestxray_dataset import HFChestXrayDataset
from models.resnet50    import get_resnet50
from models.densenet121 import get_densenet121
from multilabel_metrics import compute_metrics

# -------- argparse --------
p = argparse.ArgumentParser()
p.add_argument("--resnet_ckpt", required=True)
p.add_argument("--densenet_ckpt", required=True)
p.add_argument("--out", type=str, default="ensemble_best_thresh.npy")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- validation data --------
val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
ds_val = load_dataset("alkzar90/NIH-Chest-X-ray-dataset",
                      "image-classification",
                      split="train[80%:]", trust_remote_code=True)
val_set = HFChestXrayDataset(ds_val, transform=val_tf)
val_loader = DataLoader(val_set, batch_size=64,
                        shuffle=False, num_workers=4)
num_classes = val_set.num_classes

# -------- load models --------
resnet   = get_resnet50(num_classes, False).to(device)
densenet = get_densenet121(num_classes, False).to(device)
resnet.load_state_dict(torch.load(args.resnet_ckpt, map_location=device))
densenet.load_state_dict(torch.load(args.densenet_ckpt, map_location=device))
resnet.eval(); densenet.eval()

# -------- inference --------
probs, labels = [], []
with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        p1 = torch.sigmoid(resnet(x))
        p2 = torch.sigmoid(densenet(x))
        p  = (p1 + p2) / 2     # 확률 앙상블
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
probs = np.concatenate(probs)
labels= np.concatenate(labels)

# -------- per-class threshold 튜닝 (accuracy 기준) --------
grid = np.arange(0.05, 0.951, 0.01)
best_t = np.zeros(num_classes)

for k in range(num_classes):
    best_t[k] = max(
        grid,
        key=lambda t: ((probs[:, k] > t) == labels[:, k]).mean()
    )

# -------- metric 계산 + 저장 --------
metrics = compute_metrics(labels, probs, threshold=best_t)
np.save(args.out, best_t)

print(f"✅ Saved best threshold → {args.out}")
print("\n📊 Ensemble Metrics (best threshold):")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")