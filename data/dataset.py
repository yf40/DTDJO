import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FoggyDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, augment=False, is_train=True):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment
        self.is_train = is_train
        
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if is_train and augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label_path = os.path.join(self.label_dir, self.img_files[idx].replace('.jpg', '.txt').replace('.png', '.txt'))
        
        boxes = []
        class_labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    values = line.strip().split()
                    if len(values) >= 5:
                        class_id = int(values[0])
                        x_center, y_center, width, height = map(float, values[1:5])
                        boxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
        
        if len(boxes) == 0:
            boxes = [[0, 0, 0, 0]]
            class_labels = [0]
        
        transformed = self.transform(image=img, bboxes=boxes, class_labels=class_labels)
        img = transformed['image']
        boxes = transformed['bboxes']
        class_labels = transformed['class_labels']
        
        targets = torch.zeros((len(boxes), 6))
        for i, (box, cls) in enumerate(zip(boxes, class_labels)):
            targets[i] = torch.tensor([0, cls, box[0], box[1], box[2], box[3]])
        
        return img, targets, img_path


def collate_fn(batch):
    imgs, targets, paths = zip(*batch)
    imgs = torch.stack(imgs, 0)
    
    for i, target in enumerate(targets):
        target[:, 0] = i
    
    targets = torch.cat(targets, 0)
    return imgs, targets, paths
