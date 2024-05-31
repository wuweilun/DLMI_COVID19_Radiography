import argparse
import os
import sys
import warnings
import glob
import pandas as pd
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import random_split, Dataset, DataLoader
from utils import CustomDataset, bce_dice_loss, Trainer, JointTransform
from transformers import AutoModel

model_name = sys.argv[1]
ratio = float(sys.argv[2])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size, criterion_s, criterion_c, model = 224, None, torch.nn.CrossEntropyLoss(), AutoModel.from_pretrained('facebook/dinov2-base')
num_classes = 4
model.classifier = nn.Linear(768, num_classes) 
# for param in model.parameters():
#     param.requires_grad = False
# for param in model.classifier.parameters():
#     param.requires_grad = True
for param in model.embeddings.parameters():
    param.requires_grad = False
freeze_up_to_layer = 8
for layer_index, layer in enumerate(model.encoder.layer):
    if layer_index < freeze_up_to_layer:
        for param in layer.parameters():
            param.requires_grad = False
# for name, param in model.named_parameters():
#     print(f"{name} is {'frozen' if not param.requires_grad else 'not frozen'}")
image_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

root_dir = "COVID-19_Radiography_Dataset"
train_joint_transform = JointTransform(image_transform=image_transform, mask_transform=mask_transform, flip=True)
val_joint_transform = JointTransform(image_transform=image_transform, mask_transform=mask_transform, flip=False)
dataset = CustomDataset(root_dir)

dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
val_size = dataset_size - train_size
batch_size = 64

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
if(ratio < 1):
    unused_size = int(train_size * (1 - ratio))
    train_dataset, unused_dataset = random_split(train_dataset, [train_size - unused_size, unused_size])
train_dataset.dataset.transform = train_joint_transform
val_dataset.dataset.transform = val_joint_transform

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

model = model.to(device)
epochs = 20
learning_rate = 0.0001
weight_decay = 1e-6

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

trainer = Trainer(model=model, model_name=model_name, num_epochs=epochs, optimizer=optimizer, device=device, project_name="DLMI_PROJECT", criterion_segmentation=criterion_s, criterion_classification=criterion_c)

trainer.train(train_loader, val_loader)