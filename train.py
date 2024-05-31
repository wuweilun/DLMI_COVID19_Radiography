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
import segmentation_models_pytorch as smp
from torchvision.models import convnext_base, densenet161, efficientnet_v2_l, resnext101_64x4d, swin_b, vit_b_16
torch.manual_seed(123)

def get_model(model_name):
    num_classes = 4 
    default_size = 224
    if model_name == "convnext_base":
        model = convnext_base(weights='ConvNeXt_Base_Weights.IMAGENET1K_V1')
        model.classifier = nn.Sequential(
            model.classifier,
            nn.ReLU(),
            nn.Linear(1000, num_classes)
        )
        return default_size, None, torch.nn.CrossEntropyLoss(), model
    elif model_name == "densenet161":
        model = densenet161(weights='DenseNet161_Weights.IMAGENET1K_V1')
        model.classifier = nn.Sequential(
            model.classifier,
            nn.ReLU(),
            nn.Linear(1000, num_classes)
        )
        return default_size, None, torch.nn.CrossEntropyLoss(), model
    elif model_name == "efficientnet_v2_l":
        model = efficientnet_v2_l(weights='EfficientNet_V2_L_Weights.IMAGENET1K_V1')
        model.classifier[1] = nn.Sequential(
            model.classifier[1],
            nn.ReLU(),
            nn.Linear(1000, num_classes)
        )
        return default_size, None, torch.nn.CrossEntropyLoss(), model
    elif model_name == "resnext101_64x4d":
        model = resnext101_64x4d(weights='ResNeXt101_64X4D_Weights.IMAGENET1K_V1')
        model.fc = nn.Sequential(
            model.fc,
            nn.ReLU(),
            nn.Linear(1000, num_classes)
        )
        return default_size, None, torch.nn.CrossEntropyLoss(), model
    elif model_name == "swin_b":
        model = swin_b(weights='Swin_B_Weights.IMAGENET1K_V1')
        model.head = nn.Sequential(
            model.head,
            nn.ReLU(),
            nn.Linear(1000, num_classes)
        )
        return default_size, None, torch.nn.CrossEntropyLoss(), model
    elif model_name == "vit_b_16":
        model = vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_V1')
        model.heads = nn.Sequential(
            model.heads,
            nn.ReLU(),
            nn.Linear(1000, num_classes)
        )
        return default_size, None, torch.nn.CrossEntropyLoss(), model
    elif model_name == "unet":
        return default_size, bce_dice_loss, None, smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1)
    elif model_name == "unet++":
        return default_size, bce_dice_loss, None, smp.UnetPlusPlus(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1)
    elif model_name == "manet":
        return default_size, bce_dice_loss, None, smp.MAnet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1)
    elif model_name == "linknet":
        return default_size, bce_dice_loss, None, smp.Linknet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1)
    elif model_name == "fpn":
        return default_size, bce_dice_loss, None, smp.FPN(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1)
    elif model_name == "pspnet":
        return default_size, bce_dice_loss, None, smp.PSPNet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1)
    elif model_name == "pan":
        return default_size, bce_dice_loss, None, smp.PAN(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1)
    elif model_name == "deeplabv3":
        return default_size, bce_dice_loss, None, smp.DeepLabV3(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1)
    elif model_name == "deeplabv3+":
        return default_size, bce_dice_loss, None, smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1)
    else:
        raise ValueError("Model name not supported!")
    
model_name = sys.argv[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size, criterion_s, criterion_c, model = get_model(model_name)

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
batch_size = 16

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataset.dataset.transform = train_joint_transform
val_dataset.dataset.transform = val_joint_transform

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = model.to(device)
epochs = 20
learning_rate = 0.0001
weight_decay = 1e-6

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

trainer = Trainer(model=model, model_name=model_name, num_epochs=epochs, optimizer=optimizer, device=device, project_name="DLMI_PROJECT", criterion_segmentation=criterion_s, criterion_classification=criterion_c)

trainer.train(train_loader, val_loader)