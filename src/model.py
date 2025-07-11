"""
Neural network models for Alzheimer's detection
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights


class AlzheimerClassifier(nn.Module):
    """Base classifier for Alzheimer's detection"""
    
    def __init__(self, model_name='resnet50', num_classes=4, pretrained=True, dropout_rate=0.5):
        super(AlzheimerClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        if model_name == 'resnet50':
            # Load pretrained ResNet50
            if pretrained:
                self.base_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            else:
                self.base_model = models.resnet50(weights=None)
            
            # Get the number of features in the last layer
            num_features = self.base_model.fc.in_features
            
            # Replace the final fully connected layer
            self.base_model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(dropout_rate/2),
                nn.Linear(512, num_classes)
            )
            
        elif model_name == 'efficientnet':
            # Load pretrained EfficientNet-B0
            if pretrained:
                self.base_model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                self.base_model = models.efficientnet_b0(weights=None)
            
            # Get the number of features in the last layer
            num_features = self.base_model.classifier[1].in_features
            
            # Replace the classifier
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(dropout_rate/2),
                nn.Linear(512, num_classes)
            )
            
        elif model_name == 'vit':
            # Vision Transformer (requires timm library)
            try:
                import timm
                self.base_model = timm.create_model('vit_base_patch16_224', 
                                                   pretrained=pretrained, 
                                                   num_classes=num_classes)
            except ImportError:
                raise ImportError("Please install timm library: pip install timm")
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def forward(self, x):
        return self.base_model(x)
    
    def get_feature_extractor(self):
        """Get the feature extraction part of the model"""
        if self.model_name == 'resnet50':
            # Return everything except the final fc layer
            modules = list(self.base_model.children())[:-1]
            return nn.Sequential(*modules)
        elif self.model_name == 'efficientnet':
            # Return everything except the classifier
            return nn.Sequential(
                self.base_model.features,
                self.base_model.avgpool
            )
        else:
            raise NotImplementedError(f"Feature extraction not implemented for {self.model_name}")


class MultiModalClassifier(nn.Module):
    """Multi-modal classifier combining image and clinical features"""
    
    def __init__(self, image_model_name='resnet50', num_classes=4, 
                 clinical_features_dim=10, pretrained=True):
        super(MultiModalClassifier, self).__init__()
        
        # Image model
        self.image_model = AlzheimerClassifier(
            model_name=image_model_name, 
            num_classes=num_classes, 
            pretrained=pretrained
        )
        
        # Get feature extractor
        self.image_feature_extractor = self.image_model.get_feature_extractor()
        
        # Image features dimension
        if image_model_name == 'resnet50':
            image_features_dim = 2048
        elif image_model_name == 'efficientnet':
            image_features_dim = 1280
        else:
            image_features_dim = 768  # Default for ViT
        
        # Clinical features processing
        self.clinical_processor = nn.Sequential(
            nn.Linear(clinical_features_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        
        # Combined classifier
        combined_dim = image_features_dim + 64
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, image, clinical_features=None):
        # Extract image features
        img_features = self.image_feature_extractor(image)
        img_features = img_features.view(img_features.size(0), -1)
        
        if clinical_features is not None:
            # Process clinical features
            clinical_features = self.clinical_processor(clinical_features)
            
            # Combine features
            combined_features = torch.cat([img_features, clinical_features], dim=1)
        else:
            combined_features = img_features
        
        # Final classification
        output = self.classifier(combined_features)
        
        return output


def get_model(model_name='resnet50', num_classes=4, pretrained=True, **kwargs):
    """Factory function to get model"""
    return AlzheimerClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )