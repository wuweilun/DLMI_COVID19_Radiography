# DLMI_COVID19_Radiography
This repository is NTU Deep Learning for Medical Imaging course 2024 final project. It contains models and methodologies for classifying and segmenting chest X-ray images into categories such as COVID-19, Lung Opacity, Normal, and Viral Pneumonia. The models also perform multi-task learning to handle both classification and segmentation in a unified framework.
 
## Data

The data used in this project is the COVID-19 Chest X-Ray Database available on Kaggle. It includes 21,165 images with corresponding lung masks, categorized into:
- **COVID-19**: 3,616 images
- **Lung Opacity**: 6,012 images
- **Normal**: 10,192 images
- **Viral Pneumonia**: 1,345 images

Each image and mask is provided at a resolution of 300x300 pixels in PNG format.

## Project Objective

The project's goal is to leverage advanced machine learning techniques to enhance the accuracy and efficiency of diagnosing chest-related diseases from X-ray images. This involves:
- **Classification**: Using supervised, self-supervised, and zero-shot methods.
- **Segmentation**: Employing various supervised segmentation models.
- **Multi-Task Learning**: Integrating classification and segmentation tasks within a single model framework.

## Methodology

### Classification Techniques

1. **Supervised Learning**: Models like Swin Transformer, VIT Base, and others are fine-tuned using the complete training dataset.
2. **Self-Supervised Learning**: Implements models such as DINOv2 and BEITv2, which utilize partial dataset fine-tuning and frozen encoder layers to enhance training speed and reduce performance degradation.
3. **Zero-Shot Learning**: Utilizes the CLIP model with specifically designed prompts to classify images without direct training on the task.

### Segmentation Models

Models such as Unet, Unet++, and DeepLabV3+ are used to segment the chest X-ray images, focusing on achieving high Dice scores and accurate lung mask segmentation.

### Multi-Task Learning

The Unet architecture is modified to include a classification branch post-encoder to simultaneously perform classification and segmentation.

## Experiments and Analysis

- **Training Details**: The models are trained with an 80/20 split for training and validation sets using Adam optimizer, a learning rate of 5e-5, and a weight decay of 1e-6.
- **Performance Metrics**: Models are evaluated based on F1 score, precision, recall, accuracy, and training speed.

## References

- [Kaggle COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- [torchvision Models](https://pytorch.org/vision/stable/models.html)
