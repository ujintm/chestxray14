from hf_chestxray_dataset import HFChestXrayDataset
from datasets import load_dataset

ds_train = load_dataset("alkzar90/NIH-Chest-X-ray-dataset",
                        "image-classification",
                        split="train[:80%]", trust_remote_code=True)
train_set = HFChestXrayDataset(ds_train)

print("train labels shape:", train_set.labels.shape)
print("양성 비율 (전체):", train_set.labels.mean())
print("0인 샘플 수:", (train_set.labels.sum(axis=1) == 0).sum())