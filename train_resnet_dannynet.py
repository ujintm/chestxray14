import os, csv, copy, math, argparse
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torchvision.ops import sigmoid_focal_loss
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.ops import sigmoid_focal_loss
from datasets import load_dataset
from hf_chestxray_dataset import HFChestXrayDataset
from models.resnet50    import get_resnet50
from models.densenet121 import get_densenet121
from multilabel_metrics import compute_metrics

# ---------- arg ----------
ap = argparse.ArgumentParser()
ap.add_argument("--arch", choices=["resnet50","densenet121"], default="resnet50")
ap.add_argument("--out", default="checkpoints_danny", help="체크포인트 폴더")
ap.add_argument("--bs",   type=int, default=64)
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--lr",     type=float, default=3e-4)
ap.add_argument("--resume", type=str, default=None)

args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------- data ----------
aug_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
aug_val = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

ds_train = load_dataset("alkzar90/NIH-Chest-X-ray-dataset",
                        "image-classification",
                        split="train[:80%]", trust_remote_code=True)
ds_val   = load_dataset("alkzar90/NIH-Chest-X-ray-dataset",
                        "image-classification",
                        split="train[80%:]", trust_remote_code=True)

train_set = HFChestXrayDataset(ds_train, transform=aug_train)
val_set   = HFChestXrayDataset(ds_val,   transform=aug_val,
                               label_map=train_set.label_map)

loader = {
    'train': DataLoader(train_set, batch_size=args.bs,
                        shuffle=True, num_workers=4, pin_memory=True),
    'val'  : DataLoader(val_set,   batch_size=args.bs*2,
                        shuffle=False, num_workers=4, pin_memory=True)
}
ds_len = {'train': len(train_set), 'val': len(val_set)}
num_classes = train_set.num_classes

# ---------- model ----------
if args.arch == "resnet50":
    model = get_resnet50(num_classes, freeze_backbone=False)
else:
    model = get_densenet121(num_classes, freeze_backbone=False)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
# 2-epoch warm-up + cosine
T_warm = len(loader['train'])*2
sched_warm = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,
                                         total_iters=T_warm)
sched_cos  = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                            T_0=5, T_mult=2)
scaler = GradScaler()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

start_epoch, best_acc, best_wts = 0, 0., copy.deepcopy(model.state_dict())

# ---------- resume ----------
if args.resume and os.path.exists(args.resume):
    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scaler.load_state_dict(ckpt['scaler'])
    start_epoch = ckpt['epoch']
    best_acc    = ckpt.get('best_acc', 0.)
    print(f"🔁 Resumed from {args.resume} @ epoch {start_epoch}")

# ---------- train ----------
csv_path = os.path.join(args.out, "metrics.csv")
if not os.path.exists(csv_path):
    with open(csv_path,'w',newline='') as f:
        csv.writer(f).writerow(['epoch','val_loss','thr_mode','accuracy','f1_macro','auc_macro'])

for epoch in range(start_epoch, args.epochs):
    print(f"\nEpoch {epoch+1}/{args.epochs}")
    for phase in ['train','val']:
        model.train() if phase=='train' else model.eval()
        running_loss, preds_all, labels_all = 0., [], []

        for step,(x,y) in enumerate(loader[phase]):
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase=='train'):
                with autocast():
                    logits = model(x)
                    loss = sigmoid_focal_loss(logits, y.float(),
                                              alpha=0.4, gamma=1.5, reduction='mean')
                if phase=='train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            if phase=='val':
                preds_all.append(torch.sigmoid(logits).cpu().numpy())
                labels_all.append(y.cpu().numpy())

            running_loss += loss.item()*x.size(0)
            val_loss = loss.item() if phase == 'val' else None

            # warm-up / cosine lr step
            if phase=='train':
                gstep = epoch*len(loader['train']) + step
                if gstep < T_warm: sched_warm.step()
                else: sched_cos.step(epoch + step/len(loader['train']))

        epoch_loss = running_loss / ds_len[phase]
        if phase == 'val': scheduler.step(epoch_loss)
        print(f"{phase} Loss: {epoch_loss:.4f}")

        # ---- validation metrics & ckpt ----
        if phase=='val':
            y_pred = np.concatenate(preds_all); y_true = np.concatenate(labels_all)

            # best per-class threshold (0.01 grid)
            grid = np.arange(0.05,0.951,0.01)
            best_t = np.zeros(num_classes)
            for k in range(num_classes):
                best_t[k] = max(grid, key=lambda t: ((y_pred[:,k]>t)==y_true[:,k]).mean())
            metrics = compute_metrics(y_true, y_pred, threshold=best_t)
            acc = metrics['accuracy']
            print(f" Accuracy={acc:.4f}")

            # CSV 로그
            with open(csv_path,'a',newline='') as f:
                row = [epoch+1, f"{epoch_loss:.4f}", 'per-class'] + \
                      [f"{metrics[k]:.4f}" for k in ['accuracy','f1_macro','auc_macro']]
                csv.writer(f).writerow(row)

            # 에포크 체크포인트
            if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
                ckpt_file = os.path.join(
                    args.out,
                    f"checkpoint_epoch_{epoch+1:02d}.pth"
                )
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_acc': best_acc
                }, ckpt_file)
                print(f"Checkpoint ➜ {ckpt_file}")
            
            # best ckpt
            if acc > best_acc:
                best_acc, best_wts = acc, copy.deepcopy(model.state_dict())
                torch.save(best_wts, os.path.join(args.out, "best_model.pth"))
                np.save(os.path.join(args.out, "best_thresh.npy"), best_t)
                print("★ New best saved!")

print(f"\n[Done] best accuracy = {best_acc:.4f}")