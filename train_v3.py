import argparse, os, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from datasets import load_dataset
from hf_chestxray_dataset import HFChestXrayDataset
from models.resnet50 import get_resnet50
from multilabel_metrics import compute_metrics

def seed_everything(seed=42):
    import random
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True

def compute_pos_weight(loader, num_classes):
    pos = torch.zeros(num_classes)
    neg = torch.zeros(num_classes)
    for _, y in loader:
        pos += y.sum(0)
        neg += (1 - y).sum(0)
    return (neg / (pos + 1e-6))

def get_sampler(labels):
    has_pos = labels.sum(1) > 0
    class_sample_count = torch.tensor([(~has_pos).sum(), has_pos.sum()]).float()
    weight = 1. / class_sample_count
    samples_weight = weight[has_pos.long()].to(torch.float32)
    return WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

def tune_threshold(probs, targets, metric="acc"):
    best_t, best_score = 0.5, -1
    for t in np.arange(0.05, 0.95, 0.05):
        preds = (probs > t).astype(np.int8)
        if metric == "acc":
            score = (preds == targets).mean()
        else:
            metrics = compute_metrics(targets, preds, threshold=None)
            score = metrics["f1_macro"]  # 또는 "auc_macro"
        if score > best_score:
            best_score, best_t = score, t
    return best_t, best_score

def main(cfg):
    seed_everything()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------- transforms ----------
    tr = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # --------- load dataset from Hugging Face ---------
    ds_train = load_dataset("alkzar90/NIH-Chest-X-ray-dataset",
                            "image-classification",
                            split="train[:80%]",
                            trust_remote_code=True)
    ds_val = load_dataset("alkzar90/NIH-Chest-X-ray-dataset",
                          "image-classification",
                          split="train[80%:]",
                          trust_remote_code=True)

    train_set = HFChestXrayDataset(ds_train, transform=tr)
    val_set = HFChestXrayDataset(ds_val, transform=val_t, label_map=train_set.label_map)
    num_classes = train_set.num_classes

    tmp_loader = DataLoader(train_set, batch_size=cfg.bs, num_workers=4, pin_memory=True)
    pos_weight = compute_pos_weight(tmp_loader, num_classes).to(device)
    print("[INFO] pos_weight =", pos_weight.cpu().numpy().round(2))

    if cfg.sampler:
        sampler = get_sampler(train_set.labels) 
        train_loader = DataLoader(train_set, batch_size=cfg.bs,
                                  sampler=sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_set, batch_size=cfg.bs, shuffle=True,
                                  num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.bs * 2,
                            shuffle=False, num_workers=4, pin_memory=True)

    # --------- model ----------
    if cfg.arch == "resnet50":
        model = get_resnet50(num_classes, freeze_backbone=False)
    else:
        raise NotImplementedError("Only resnet50 is supported.")
    model.to(device)

    # --------- optim / loss ----------
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    start_epoch = 0
    best_acc, best_thresh = 0, 0.5
    
    if cfg.resume and os.path.exists(cfg.resume):
        print(f"🔁 Resuming from {cfg.resume}")
        ckpt = torch.load(cfg.resume)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "threshold" in ckpt:
            best_thresh = ckpt["threshold"]
        start_epoch = cfg.start if cfg.start is not None else ckpt.get("epoch", 0)
        print(f"➡️ Restarting at epoch {start_epoch}")
    
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        tot_loss = 0
        for x, y in tqdm(train_loader, desc=f"[{epoch+1}/{cfg.epochs}]"):
            x, y = x.to(device), y.to(device).float()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tot_loss += loss.item() * x.size(0)
        scheduler.step()

        # validate
        model.eval()
        probs_list, labels_list = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                probs = torch.sigmoid(logits).cpu()
                probs_list.append(probs)
                labels_list.append(y)
        probs = torch.cat(probs_list).numpy()
        labels = torch.cat(labels_list).numpy()

        t, acc = tune_threshold(probs, labels, metric="acc")
        print(f"  ▸ val accuracy={acc:.4f} @ threshold={t:.2f}")

        if acc > best_acc:
            best_acc, best_thresh = acc, t
            torch.save({
                "state_dict": model.state_dict(),
                "threshold": best_thresh,
                "epoch": epoch + 1,
                "optimizer": optimizer.state_dict()
            },f"checkpoints_v3/best_{cfg.arch}.pth")
            print("  ★ checkpoint saved")

    print(f"\n[Done] best acc={best_acc:.4f} @ thresh={best_thresh:.2f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["resnet50"], default="resnet50")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--sampler", action="store_true")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    p.add_argument("--start", type=int, default=None, help="Start epoch override")
    cfg = p.parse_args()
    main(cfg)
