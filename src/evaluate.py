"""
Evaluation script for Alzheimer's detection model
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import logging
from tqdm import tqdm

from .model import get_model
from .data_loader import AlzheimerDataLoader
from .utils import load_config


def plot_confusion_matrix(cm, class_names, save_path='results/figures/confusion_matrix.png'):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curves(y_true, y_score, class_names, save_path='results/figures/roc_curves.png'):
    """Plot ROC curves for multi-class classification"""
    n_classes = len(class_names)
    
    # Binarize the labels
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Multi-class Classification')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_history(history_path='results/training_history.json'):
    """Plot training history"""
    history = load_config('training_history.json')
    
    if not history:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot losses
    epochs = range(1, len(history['train_losses']) + 1)
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(epochs, history['train_accs'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_accs'], 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/figures/training_history.png', dpi=300)
    plt.close()


def evaluate_model(model_name='resnet50', model_path=None, data_dir='data/raw'):
    """Evaluate the trained model"""
    logger = logging.getLogger('alzheimer_detection')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load configuration
    config = load_config('training_config.json')
    if not config:
        logger.error("Training configuration not found!")
        return
    
    # Create data loader
    data_loader = AlzheimerDataLoader(data_dir=data_dir, batch_size=32)
    _, _, test_loader = data_loader.get_data_loaders()
    
    # Load model
    model = get_model(model_name=model_name, num_classes=data_loader.num_classes)
    
    # Find best model if not specified
    if model_path is None:
        model_files = [f for f in os.listdir('models/saved_models') 
                      if f.endswith('.pth') and 'alzheimer_model' in f]
        if not model_files:
            logger.error("No saved models found!")
            return
        
        # Get the most recent model
        model_files.sort()
        model_path = os.path.join('models/saved_models', model_files[-1])
    
    logger.info(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Evaluate on test set
    all_predictions = []
    all_labels = []
    all_probs = []
    
    logger.info("Evaluating model on test set...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = 100. * np.sum(all_predictions == all_labels) / len(all_labels)
    logger.info(f"Test Accuracy: {accuracy:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    logger.info("Confusion Matrix:")
    logger.info(cm)
    
    # Classification report
    report = classification_report(all_labels, all_predictions, 
                                 target_names=data_loader.class_names)
    logger.info("\nClassification Report:")
    logger.info(report)
    
    # Save classification report
    with open('results/classification_report.txt', 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, data_loader.class_names)
    
    # Plot ROC curves
    plot_roc_curves(all_labels, all_probs, data_loader.class_names)
    
    # Plot training history
    plot_training_history()
    
    logger.info("Evaluation completed! Results saved in results/")
    
    return accuracy, cm, report