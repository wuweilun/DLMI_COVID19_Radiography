import torch
import os
import wandb
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score


class Trainer:
    def __init__(self, model, model_name, num_epochs, optimizer, device, project_name, criterion_segmentation=None, criterion_classification=None):
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.criterion_segmentation = criterion_segmentation
        self.criterion_classification = criterion_classification
        self.model = model
        self.device = device

        self.is_classification = criterion_classification is not None
        self.is_segmentation = criterion_segmentation is not None
        
        # Initialize wandb
        wandb.init(project=project_name, entity="weilunwu", name=model_name) 

        self.best_model = None
        self.best_dice = 0.0
        self.best_acc = 0.0
        self.best_epoch = 0

    def dice_coeff(self, predicted, target, smooth=1e-5):
        predicted = F.sigmoid(predicted)  # Applying sigmoid to the output if not already applied
        intersection = torch.sum(predicted * target)
        total = torch.sum(predicted) + torch.sum(target)
        dice = (2. * intersection + smooth) / (total + smooth)
        return dice
    
    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    
    def save_best_model(self, epoch, loss, dice, acc):
        if self.is_segmentation and self.is_classification and dice > self.best_dice and dice > 0.65:
            self.best_dice = dice
            self.best_epoch = epoch
            self.best_model = self.model.state_dict()

            log_directory = 'log'
            os.makedirs(log_directory, exist_ok=True)
            filename = f'{log_directory}/{self.model_name}_epoch{epoch}_dice{dice:.4f}_acc{acc:.4f}.pth'
            torch.save(self.best_model, filename)           
        elif self.is_segmentation and dice > self.best_dice and dice > 0.65:
            self.best_dice = dice
            self.best_epoch = epoch
            self.best_model = self.model.state_dict()

            log_directory = 'log'
            os.makedirs(log_directory, exist_ok=True)
            filename = f'{log_directory}/{self.model_name}_epoch{epoch}_dice{dice:.4f}.pth'
            torch.save(self.best_model, filename)
        elif self.is_classification and acc > self.best_acc and acc > 0.7:
            self.best_acc = acc
            self.best_epoch = epoch
            self.best_model = self.model.state_dict()

            log_directory = 'log'
            os.makedirs(log_directory, exist_ok=True)
            filename = f'{log_directory}/{self.model_name}_epoch{epoch}_acc{acc:.4f}.pth'
            torch.save(self.best_model, filename)            

    def train(self, train_loader, val_loader):
        scaler = GradScaler()
        for epoch in range(self.num_epochs):
            train_loss, train_dice, train_accuracy = 0.0, 0.0, 0.0
            val_loss, val_dice, val_accuracy = 0.0, 0.0, 0.0
            
            num_batches = len(train_loader)
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            for images, masks, labels in progress_bar:
                images, masks, labels = images.to(self.device), masks.to(self.device), labels.to(self.device)

                self.model.train()
                loss = 0
                self.optimizer.zero_grad()
                with autocast():
                    if self.is_segmentation and self.is_classification:
                        seg_outputs, class_outputs = self.model(images)
                        loss_segmentation = self.criterion_segmentation(seg_outputs, masks)
                        loss_classification = self.criterion_classification(class_outputs, labels)
                        loss = loss_segmentation + loss_classification
                        train_dice += self.dice_coeff(seg_outputs, masks).item()
                        train_accuracy += self.accuracy(class_outputs, labels).item()
                    elif self.is_segmentation:
                        seg_outputs = self.model(images)
                        loss = self.criterion_segmentation(seg_outputs, masks)
                        train_dice += self.dice_coeff(seg_outputs, masks).item()
                    elif self.is_classification:
                        class_outputs = self.model(images)
                        loss = self.criterion_classification(class_outputs, labels)
                        train_accuracy += self.accuracy(class_outputs, labels).item()
        
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                train_loss += loss.item()

            self.model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                num_batches_val = len(val_loader)
                for images, masks, labels in val_loader:
                    images, masks, labels = images.to(self.device), masks.to(self.device), labels.to(self.device)
                    with autocast():
                        if self.is_segmentation and self.is_classification:
                            seg_outputs, class_outputs = self.model(images)
                            val_loss_segmentation = self.criterion_segmentation(seg_outputs, masks)
                            val_loss_classification = self.criterion_classification(class_outputs, labels)
                            val_loss += (val_loss_segmentation.item() + val_loss_classification.item())
                            val_dice += self.dice_coeff(seg_outputs, masks).item()
                            # val_accuracy += self.accuracy(class_outputs, labels).item()
                            _, preds = torch.max(class_outputs, dim=1)
                            all_preds.extend(preds.cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())
                        elif self.is_segmentation:
                            seg_outputs = self.model(images)
                            val_loss_segmentation = self.criterion_segmentation(seg_outputs, masks)
                            val_loss += val_loss_segmentation.item()
                            val_dice += self.dice_coeff(seg_outputs, masks).item()
                        elif self.is_classification:
                            class_outputs = self.model(images)
                            val_loss_classification = self.criterion_classification(class_outputs, labels)
                            val_loss += val_loss_classification.item()
                            # val_accuracy += self.accuracy(class_outputs, labels).item()
                            _, preds = torch.max(class_outputs, dim=1)
                            all_preds.extend(preds.cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())
            precision = precision_score(all_labels, all_preds, average='macro')
            recall = recall_score(all_labels, all_preds, average='macro')
            f1 = f1_score(all_labels, all_preds, average='macro')
            accuracy = accuracy_score(all_labels, all_preds) 
            # Log epoch-level metrics
            wandb.log({
                "Train Loss": train_loss / num_batches,
                "Train Dice": train_dice / num_batches,
                "Train Accuracy": train_accuracy / num_batches,
                "Val Loss": val_loss / num_batches_val,
                "Val Dice": val_dice / num_batches_val,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "Val Accuracy": accuracy
            })

            self.save_best_model(epoch + 1, val_loss / num_batches_val, val_dice / num_batches_val, val_accuracy / num_batches_val)
