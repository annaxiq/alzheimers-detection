"""
Data loader for Alzheimer's Detection dataset
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import logging


class AlzheimerDataset(Dataset):
    """Custom Dataset for Alzheimer's MRI images"""
    
    def __init__(self, data_df, transform=None):
        self.data_df = data_df
        self.transform = transform
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        img_path = self.data_df.iloc[idx]['path']
        label = self.data_df.iloc[idx]['label']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


class AlzheimerDataLoader:
    """Data loader manager for Alzheimer's dataset"""
    
    def __init__(self, data_dir='data/raw', batch_size=32, img_size=224):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.logger = logging.getLogger('alzheimer_detection')
        
        # Class mapping
        self.class_mapping = {
            'Non_Demented': 0,
            'Very_Mild_Dementia': 1,
            'Mild_Dementia': 2,
            'Moderate_Dementia': 3
        }
        
        self.class_names = list(self.class_mapping.keys())
        self.num_classes = len(self.class_names)
        
    def prepare_data(self):
        """Prepare data by creating a DataFrame with paths and labels"""
        data_list = []
        
        for class_name, label in self.class_mapping.items():
            class_dir = os.path.join(self.data_dir, class_name)
            
            if not os.path.exists(class_dir):
                self.logger.warning(f"Directory {class_dir} not found!")
                continue
                
            for img_file in os.listdir(class_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    data_list.append({
                        'path': os.path.join(class_dir, img_file),
                        'label': label,
                        'class_name': class_name
                    })
        
        df = pd.DataFrame(data_list)
        self.logger.info(f"Total images found: {len(df)}")
        self.logger.info(f"Class distribution:\n{df['class_name'].value_counts()}")
        
        return df
    
    def get_transforms(self, train=True):
        """Get image transforms for training or validation"""
        if train:
            transform = transforms.Compose([
                transforms.Resize((self.img_size + 20, self.img_size + 20)),
                transforms.RandomCrop(self.img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        return transform
    
    def get_data_loaders(self, val_split=0.15, test_split=0.15, random_state=42):
        """Get train, validation, and test data loaders"""
        # Prepare data
        df = self.prepare_data()
        
        # Split data
        # First split: train+val and test
        train_val_df, test_df = train_test_split(
            df, test_size=test_split, stratify=df['label'], random_state=random_state
        )
        
        # Second split: train and val
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_split/(1-test_split), 
            stratify=train_val_df['label'], random_state=random_state
        )
        
        self.logger.info(f"Train size: {len(train_df)}")
        self.logger.info(f"Validation size: {len(val_df)}")
        self.logger.info(f"Test size: {len(test_df)}")
        
        # Create datasets
        train_dataset = AlzheimerDataset(train_df, transform=self.get_transforms(train=True))
        val_dataset = AlzheimerDataset(val_df, transform=self.get_transforms(train=False))
        test_dataset = AlzheimerDataset(test_df, transform=self.get_transforms(train=False))
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=4, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_class_weights(self):
        """Calculate class weights for handling imbalanced dataset"""
        df = self.prepare_data()
        class_counts = df['label'].value_counts().sort_index()
        
        # Calculate weights inversely proportional to class frequencies
        total_samples = len(df)
        weights = []
        
        for i in range(self.num_classes):
            class_weight = total_samples / (self.num_classes * class_counts[i])
            weights.append(class_weight)
        
        return torch.FloatTensor(weights)