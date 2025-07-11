"""
Training script for Alzheimer's detection model
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from datetime import datetime
import logging
import json

from .model import get_model
from .data_loader import AlzheimerDataLoader
from .utils import save_config


class Trainer:
    def __init__(self, model, device, train_loader, val_loader, 
                 optimizer, criterion, scheduler=None, logger=None):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.logger = logger or logging.getLogger('alzheimer_detection')
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Store predictions for confusion matrix
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / len(pbar),
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_predictions, all_labels
    
    def train(self, epochs):
        """Train the model for specified epochs"""
        best_val_acc = 0.0
        best_model_path = None
        
        for epoch in range(epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validation
            val_loss, val_acc, _, _ = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Logging
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f"Learning rate: {current_lr}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_name = f"alzheimer_model_{timestamp}_acc{val_acc:.2f}.pth"
                model_path = os.path.join('models/saved_models', model_name)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                }, model_path)
                
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                best_model_path = model_path
                
                self.logger.info(f"Saved best model with val_acc: {val_acc:.2f}%")
        
        return best_model_path


def train_model(model_name='resnet50', batch_size=32, epochs=50, 
                learning_rate=0.001, data_dir='data/raw'):
    """Main training function"""
    logger = logging.getLogger('alzheimer_detection')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    data_loader = AlzheimerDataLoader(data_dir=data_dir, batch_size=batch_size)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    
    # Get class weights for imbalanced dataset
    class_weights = data_loader.get_class_weights().to(device)
    logger.info(f"Class weights: {class_weights}")
    
    # Create model
    model = get_model(model_name=model_name, num_classes=data_loader.num_classes)
    model = model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                 patience=5, verbose=True)
    
    # Save training configuration
    config = {
        'model_name': model_name,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'num_classes': data_loader.num_classes,
        'class_names': data_loader.class_names,
        'optimizer': 'Adam',
        'criterion': 'CrossEntropyLoss',
        'device': str(device),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    save_config(config, 'training_config.json')
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        logger=logger
    )
    
    # Train model
    logger.info("Starting training...")
    best_model_path = trainer.train(epochs)
    
    # Save training history
    history = {
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'train_accs': trainer.train_accs,
        'val_accs': trainer.val_accs
    }
    save_config(history, 'training_history.json')
    
    logger.info(f"Training completed! Best model saved at: {best_model_path}")
    
    return best_model_path