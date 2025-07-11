#!/usr/bin/env python3
"""
Main entry point for Alzheimer's Detection using OASIS dataset
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import AlzheimerDataLoader
from src.preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import setup_directories, setup_logging, download_dataset


def main():
    parser = argparse.ArgumentParser(description='Alzheimer\'s Detection using Deep Learning')
    parser.add_argument('--download', action='store_true', help='Download the dataset from Kaggle')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the dataset')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model-name', type=str, default='resnet50', 
                       choices=['resnet50', 'efficientnet', 'vit'], help='Model architecture')
    
    args = parser.parse_args()
    
    # Setup directories and logging
    setup_directories()
    logger = setup_logging()
    
    logger.info("Starting Alzheimer's Detection Pipeline")
    
    # Download dataset if requested
    if args.download:
        logger.info("Downloading OASIS dataset from Kaggle...")
        download_dataset()
    
    # Preprocess data if requested
    if args.preprocess:
        logger.info("Preprocessing dataset...")
        preprocess_data()
    
    # Train model if requested
    if args.train:
        logger.info(f"Training {args.model_name} model...")
        train_model(
            model_name=args.model_name,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr
        )
    
    # Evaluate model if requested
    if args.evaluate:
        logger.info("Evaluating model...")
        evaluate_model(model_name=args.model_name)
    
    # If no arguments provided, run the full pipeline
    if not any([args.download, args.preprocess, args.train, args.evaluate]):
        logger.info("Running full pipeline...")
        
        # Check if data exists
        if not os.path.exists('data/raw/Non_Demented'):
            logger.info("Dataset not found. Downloading...")
            download_dataset()
        
        logger.info("Preprocessing data...")
        preprocess_data()
        
        logger.info("Training model...")
        train_model(
            model_name=args.model_name,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr
        )
        
        logger.info("Evaluating model...")
        evaluate_model(model_name=args.model_name)
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()