from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import torchvision.transforms.functional as TF

class JointTransform:
    def __init__(self, image_transform=None, mask_transform=None, flip=False):
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.flip = flip
        
    def __call__(self, image, mask):
        if self.flip:
        # Random horizontal flipping
            if random.random() > 0.25:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random vertical flipping
            if random.random() > 0.25:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        # Apply other transformations
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return image, mask

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.images = []
        self.masks = []
        self.labels = []
        
        categories = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

        for idx, category in enumerate(categories):
            image_dir = os.path.join(root_dir, category, 'images')
            mask_dir = os.path.join(root_dir, category, 'masks')

            images = glob.glob(os.path.join(image_dir, '*.png'))
            masks = glob.glob(os.path.join(mask_dir, '*.png'))

            self.images.extend(images)
            self.masks.extend(masks)
            self.labels.extend([idx] * len(images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask, label
