# datasets.py
import os
from PIL import Image
from torch.utils.data import Dataset


class AugmentedDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files, self.labels = [], []
        self.classes = sorted(
            [
                d
                for d in os.listdir(image_dir)
                if os.path.isdir(os.path.join(image_dir, d))
            ]
        )
        for idx, class_name in enumerate(self.classes):
            class_folder = os.path.join(image_dir, class_name)
            for f in os.listdir(class_folder):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_files.append(os.path.join(class_folder, f))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
