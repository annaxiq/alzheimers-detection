"""
Utility functions for the Alzheimer's Detection project
"""

import os
import sys
import logging
import json
import zipfile
from pathlib import Path
from datetime import datetime
import subprocess
from dotenv import load_dotenv


def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'data/raw',
        'data/processed',
        'models/saved_models',
        'results/figures',
        'results/logs',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def setup_logging():
    """Setup logging configuration"""
    log_dir = 'results/logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('alzheimer_detection')
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(f'{log_dir}/alzheimer_detection_{timestamp}.log')
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def download_dataset():
    """Download OASIS dataset from Kaggle"""
    logger = logging.getLogger('alzheimer_detection')
    
    # Load environment variables
    load_dotenv()
    
    # Check if Kaggle credentials are set
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    kaggle_key = os.getenv('KAGGLE_KEY')
    
    if not kaggle_username or not kaggle_key:
        # Try to read from kaggle.json if env vars not set
        kaggle_json_path = os.path.expanduser('~/.kaggle/kaggle.json')
        if os.path.exists(kaggle_json_path):
            logger.info("Using Kaggle credentials from ~/.kaggle/kaggle.json")
        else:
            logger.error(
                "Kaggle credentials not found. Please either:\n"
                "1. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables, or\n"
                "2. Place your kaggle.json file in ~/.kaggle/, or\n"
                "3. Create a .env file with KAGGLE_USERNAME and KAGGLE_KEY"
            )
            return False
    
    try:
        # Download dataset using Kaggle API
        logger.info("Downloading OASIS Alzheimer's Detection dataset...")
        
        # Using subprocess to run kaggle command
        cmd = ['kaggle', 'datasets', 'download', '-d', 'ninadaithal/imagesoasis', '-p', 'data/raw']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Failed to download dataset: {result.stderr}")
            return False
        
        # Extract the dataset
        logger.info("Extracting dataset...")
        zip_path = 'data/raw/imagesoasis.zip'
        
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('data/raw')
            
            # Remove the zip file
            os.remove(zip_path)
            
            logger.info("Dataset downloaded and extracted successfully!")
            
            # Rename folders to match expected structure
            rename_folders()
            
            return True
        else:
            logger.error("Downloaded zip file not found")
            return False
            
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        return False


def rename_folders():
    """Rename dataset folders to match expected structure"""
    logger = logging.getLogger('alzheimer_detection')
    
    folder_mappings = {
        'Non Demented': 'Non_Demented',
        'Very mild Dementia': 'Very_Mild_Dementia',
        'Mild Dementia': 'Mild_Dementia',
        'Moderate Dementia': 'Moderate_Dementia'
    }
    
    data_dir = Path('data/raw')
    
    for old_name, new_name in folder_mappings.items():
        old_path = data_dir / old_name
        new_path = data_dir / new_name
        
        if old_path.exists() and not new_path.exists():
            old_path.rename(new_path)
            logger.info(f"Renamed '{old_name}' to '{new_name}'")


def save_config(config_dict, filename='config.json'):
    """Save configuration to JSON file"""
    config_path = f'results/{filename}'
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    return config_path


def load_config(filename='config.json'):
    """Load configuration from JSON file"""
    config_path = f'results/{filename}'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def get_class_distribution(data_dir='data/raw'):
    """Get the distribution of classes in the dataset"""
    class_counts = {}
    
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.endswith('.jpg')])
            class_counts[class_dir] = count
    
    return class_counts


def create_kaggle_json(username, key):
    """Create kaggle.json file for authentication"""
    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    
    kaggle_config = {
        "username": username,
        "key": key
    }
    
    kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')
    with open(kaggle_json_path, 'w') as f:
        json.dump(kaggle_config, f)
    
    # Set file permissions to 600 (only owner can read/write)
    os.chmod(kaggle_json_path, 0o600)
    
    return kaggle_json_path