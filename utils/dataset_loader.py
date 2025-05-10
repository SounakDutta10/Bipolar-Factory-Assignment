
import os
import cv2
import torch
from torch.utils.data import Dataset
from utils.transforms import get_transforms

class YOLODataset(Dataset):
    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.images = [img for img in os.listdir(image_dir)]

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".jpg", ".txt"))
        image = cv2.imread(img_path)[..., ::-1]
        h, w = image.shape[:2]

        boxes = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f.readlines():
                    cls, xc, yc, bw, bh = map(float, line.strip().split())
                    boxes.append([cls, xc, yc, bw, bh])

        boxes = torch.tensor(boxes) if boxes else torch.zeros((0, 5))
        if self.transforms:
            image = self.transforms(image)

        return image, boxes

    def __len__(self):
        return len(self.images)
