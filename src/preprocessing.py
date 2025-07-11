"""
Preprocessing functions for MRI images
"""

import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
from pathlib import Path
import multiprocessing as mp
from functools import partial


def preprocess_image(img_path, output_size=(224, 224)):
    """Preprocess a single MRI image"""
    try:
        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return None
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        # Denoise
        img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
        
        # Find brain region (simple thresholding)
        _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (brain region)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Crop to brain region with padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            
            img = img[y:y+h, x:x+w]
        
        # Resize to target size
        img = cv2.resize(img, output_size, interpolation=cv2.INTER_CUBIC)
        
        # Convert to 3-channel (RGB) for model compatibility
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        return img_rgb
        
    except Exception as e:
        logging.error(f"Error processing {img_path}: {str(e)}")
        return None


def process_class_images(class_info, output_base_dir='data/processed'):
    """Process all images in a class directory"""
    class_name, class_dir = class_info
    logger = logging.getLogger('alzheimer_detection')
    
    output_dir = os.path.join(output_base_dir, class_name)
    os.makedirs(output_dir, exist_ok=True)
    
    img_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    logger.info(f"Processing {len(img_files)} images in {class_name}")
    
    processed_count = 0
    for img_file in tqdm(img_files, desc=f"Processing {class_name}"):
        img_path = os.path.join(class_dir, img_file)
        output_path = os.path.join(output_dir, img_file)
        
        # Skip if already processed
        if os.path.exists(output_path):
            processed_count += 1
            continue
        
        # Process image
        processed_img = preprocess_image(img_path)
        
        if processed_img is not None:
            # Save processed image
            cv2.imwrite(output_path, processed_img)
            processed_count += 1
    
    logger.info(f"Processed {processed_count}/{len(img_files)} images in {class_name}")
    return class_name, processed_count


def preprocess_data(input_dir='data/raw', output_dir='data/processed', num_workers=None):
    """Preprocess all images in the dataset"""
    logger = logging.getLogger('alzheimer_detection')
    logger.info("Starting data preprocessing...")
    
    # Get class directories
    class_dirs = []
    for class_name in ['Non_Demented', 'Very_Mild_Dementia', 'Mild_Dementia', 'Moderate_Dementia']:
        class_dir = os.path.join(input_dir, class_name)
        if os.path.exists(class_dir):
            class_dirs.append((class_name, class_dir))
        else:
            logger.warning(f"Class directory not found: {class_dir}")
    
    if not class_dirs:
        logger.error("No class directories found! Please download the dataset first.")
        return
    
    # Process images
    if num_workers is None:
        num_workers = mp.cpu_count() - 1
    
    logger.info(f"Using {num_workers} workers for parallel processing")
    
    # Sequential processing for debugging
    # for class_info in class_dirs:
    #     process_class_images(class_info, output_dir)
    
    # Parallel processing
    with mp.Pool(num_workers) as pool:
        process_func = partial(process_class_images, output_base_dir=output_dir)
        results = pool.map(process_func, class_dirs)
    
    # Summary
    logger.info("Preprocessing completed!")
    for class_name, count in results:
        logger.info(f"{class_name}: {count} images processed")


def calculate_dataset_stats(data_dir='data/processed'):
    """Calculate mean and std of the dataset for normalization"""
    logger = logging.getLogger('alzheimer_detection')
    logger.info("Calculating dataset statistics...")
    
    means = []
    stds = []
    
    for class_name in ['Non_Demented', 'Very_Mild_Dementia', 'Mild_Dementia', 'Moderate_Dementia']:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        img_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in tqdm(img_files[:100], desc=f"Stats for {class_name}"):  # Sample 100 images per class
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            
            if img is not None:
                img = img.astype(np.float32) / 255.0
                means.append(np.mean(img, axis=(0, 1)))
                stds.append(np.std(img, axis=(0, 1)))
    
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)
    
    logger.info(f"Dataset mean: {mean}")
    logger.info(f"Dataset std: {std}")
    
    return mean, std