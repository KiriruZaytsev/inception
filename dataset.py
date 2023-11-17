import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn
from pathlib import Path
from torchvision import transforms
import random
import pathlib
import numpy as np
import matplotlib.pyplot as plt

SEED = 42

BATCH_SIZE = 64

LR = 1e-5


random.seed(SEED)

image_path = Path("Sports data/")

train_path = image_path / "train"

transform = transforms.Compose([
    transforms.Resize(size = (224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def find_classes(directory):
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx

class SportsDataset(Dataset):

    def __init__(self, images_dir, transform = None):
        self.paths = list(pathlib.Path(images_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(images_dir)

    def load_image(self, idx):
        image_path = self.paths[idx]
        return Image.open(image_path).convert('RGB')

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        image = self.load_image(idx)
        class_name = self.paths[idx].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(image), class_idx 
        else:
            return image, class_idx
        
def train_test_val_splitting(train_path, test_path, val_path, transform):
    train_data = SportsDataset(train_path, transform)
    test_data = SportsDataset(test_path, transform)
    val_data = SportsDataset(val_path, transform)
    return train_data, test_data, val_data

def get_class_names(train_path = train_path, transform = transform):
    data = SportsDataset(train_path, transform)
    classes = data.classes
    return classes


