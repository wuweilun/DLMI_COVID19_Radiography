import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn

def dice_coef_loss(inputs, target):
    smooth = 1e-5
    intersection = 2.0 * (target*inputs).sum() + smooth
    union = target.sum() + inputs.sum() + smooth
    return 1 - (intersection/union)

def bce_dice_loss(inputs, target):
    dice_score = dice_coef_loss(inputs, target)
    bce_loss = nn.BCELoss()
    bce_score = bce_loss(inputs, target)
    
    return bce_score + dice_score

def to_numpy(tensor):
    # Move tensor to CPU and convert to NumPy array
    return tensor.cpu().detach().numpy()

def plot_metrics(metrics, image_path, model_name):
    num_epochs = len(metrics['train_losses'])
    epochs = np.arange(1, num_epochs + 1)

    # Convert tensors to NumPy arrays
    train_losses_np = metrics['train_losses']
    val_losses_np = metrics['val_losses']
    train_dices_np = [to_numpy(dice) for dice in metrics['train_dices']]
    val_dices_np = [to_numpy(dice) for dice in metrics['val_dices']]

    # Plot Losses
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses_np, label='Train Loss')
    plt.plot(epochs, val_losses_np, label='Val Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Dice Coefficients
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_dices_np, label='Train Dice')
    plt.plot(epochs, val_dices_np, label='Val Dice')
    plt.title('Training and Validation Dice Coefficients')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'./{image_path}/{model_name}_metric.png')  # Save the figure to a file
    plt.close()

def threshold_prediction(predicted, threshold=0.5):
    # Threshold predicted values
    predicted[predicted < threshold] = 0
    predicted[predicted >= threshold] = 1
    return predicted

def plot_subplots(image, mask, predicted, threshold, image_path, model_name, i):
    # Convert tensors to NumPy arrays
    image_np, mask_np, predicted_np = map(to_numpy, (image, mask, predicted))

    # Threshold the predicted values
    predicted_np_thresholded = threshold_prediction(predicted_np, threshold)

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))  # Adjust figsize as needed

    # Plot Image, Mask, Predicted, and Thresholded Predicted
    titles = ['Image', 'Mask', 'Predicted']
    for ax, data, title in zip(axes, [image_np, mask_np, predicted_np, predicted_np_thresholded], titles):
        ax.imshow(data.squeeze(), cmap='gray' if 'Mask' in title else 'gray')
        ax.set_title(title)
        ax.axis('off')

    plt.savefig(f'./{image_path}/{model_name}_{i}.png')  # Save the figure to a file
    plt.close()

def plot_predictions(image, mask, predictions, titles, threshold, image_path, idx):

    # plt.figure(figsize=(20, 10)) 
    num_images = len(predictions) + 2  
    plt.figure(figsize=(num_images * 3, 4))  

    plt.subplot(1, len(predictions) + 2, 1)
    plt.imshow(image.squeeze(), cmap='gray') 
    if idx == 2:
        plt.title('Original Image', fontsize=28)
    plt.axis('off')

    plt.subplot(1, len(predictions) + 2, 2)
    plt.imshow(mask.squeeze(), cmap='gray')
    if idx == 2:
        plt.title('Ground Truth', fontsize=28)
    plt.axis('off')

    for i, pred in enumerate(predictions):
        plt.subplot(1, len(predictions) + 2, i + 3)
        predicted_np_thresholded = threshold_prediction(pred, threshold)
        plt.imshow(predicted_np_thresholded.squeeze(), cmap='gray')
        if idx == 2:
            plt.title(titles[i], fontsize=28)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'./{image_path}/predict_{idx}.png')  # Save the figure to a file
    plt.close()
