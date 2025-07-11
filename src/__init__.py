"""
Alzheimer's Detection Source Package
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import main components for easier access
from .data_loader import AlzheimerDataLoader
from .model import AlzheimerClassifier
from .train import train_model
from .evaluate import evaluate_model

__all__ = [
    'AlzheimerDataLoader',
    'AlzheimerClassifier',
    'train_model',
    'evaluate_model'
]