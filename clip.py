import argparse
import os
import sys
import warnings
import glob
import pandas as pd
warnings.filterwarnings("ignore")

from tqdm import tqdm
import numpy as np
import torch
from utils import CustomDataset, JointTransform
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = 224

root_dir = "COVID-19_Radiography_Dataset"
images_path = []
masks = []
all_labels = []

categories = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

for idx, category in enumerate(categories):
    image_dir = os.path.join(root_dir, category, 'images')
    images = glob.glob(os.path.join(image_dir, '*.png'))
    
    images_path.extend(images)
    # wprint(idx)
    all_labels.extend([idx] * len(images))

model = model.to(device)
# texts = [
#     "a photo of a lung X-ray with COVID-19.",
#     "a photo of a lung X-ray with opacity.",
#     "a photo of a normal lung X-ray.",
#     "a photo of a lung X-ray with viral pneumonia."
# ]
# texts = [
#     "a photo of a COVID lung.",
#     "a photo of a opacity lung.",
#     "a photo of a normal lung.",
#     "a photo of a viral pneumonia lung."
# ]
# texts = [
#     "a photo of a lung X-ray showing signs of COVID-19, characterized by bilateral peripheral ground glass opacities.",
#     "a photo of a lung X-ray showing signs of lung opacity, which might include fluid overload or infection not necessarily COVID-19.",
#     "a photo of a normal lung X-ray with clear lungs and no signs of infection or opacity.",
#     "a photo of a lung X-ray showing signs of viral pneumonia, which may include diffuse air space enlargements and interstitial patterns."
# ]
texts = [
    "A chest X-ray image showing features characteristic of COVID-19, such as bilateral ground-glass opacities.",
    "A chest X-ray image showing lung opacity which might indicate conditions like pneumonia or edema but not specific to COVID-19.",
    "A chest X-ray image of normal lungs without any signs of infection, opacity, or other abnormalities.",
    "A chest X-ray image showing signs of viral pneumonia, distinct from bacterial causes, possibly including patterns like patchy airspace opacities."
]
all_preds = []
with torch.no_grad():
    for image_name in tqdm(images_path, desc="Processing images"):
        image = Image.open(image_name)
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        predicted_class_indices = probs.argmax(dim=1)
        all_preds.extend(predicted_class_indices.cpu().numpy())
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds) 
    print(f"\nprecision: {precision:.2f}%")
    print(f"\nrecall: {recall:.2f}%")
    print(f"\nf1: {f1:.2f}%")
    print(f"\nAccuracy: {accuracy:.2f}%")
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - CLIP')
    plt.show()
    plt.savefig('confusion_matrix_clip.png', format='png', dpi=300)
    plt.close()