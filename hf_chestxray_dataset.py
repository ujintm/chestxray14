import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class HFChestXrayDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, label_map=None):
        self.dataset = hf_dataset
        self.transform = transform

        if label_map is None:
            unique_labels = sorted({label for example in self.dataset for label in example["labels"]})
            self.label_map = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label_map = label_map

        self.num_classes = len(self.label_map)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']
        labels = sample['labels']

        # ✅ 강제로 RGB 변환 추가
        image = image.convert("RGB")

         # 멀티라벨 바이너리 벡터로 변환
        target = np.zeros(self.num_classes, dtype=np.float32)
        for label in labels:
            if label in self.label_map:
                target[self.label_map[label]] = 1.0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(target)