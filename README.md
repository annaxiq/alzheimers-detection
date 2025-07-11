# Alzheimer's Detection using Deep Learning

This repository contains a deep learning pipeline for detecting Alzheimer's disease stages using the OASIS MRI dataset.

## Overview

The project uses brain MRI images to classify patients into four categories:
- Non-Demented
- Very Mild Dementia
- Mild Dementia
- Moderate Dementia

## Dataset

The project uses the OASIS Alzheimer's Detection dataset from Kaggle, which contains:
- 86,400+ brain MRI images
- 4 classes based on Clinical Dementia Rating (CDR)
- Preprocessed from the original OASIS dataset

## Features

- Multiple model architectures (ResNet50, EfficientNet, Vision Transformer)
- Data augmentation and preprocessing pipeline
- Class imbalance handling
- Comprehensive evaluation metrics
- Visualization of results

## Project Structure

```
alzheimer-detection/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── setup_and_run.ipynb    # Jupyter notebook for setup and execution
├── src/                   # Source code
│   ├── data_loader.py     # Data loading utilities
│   ├── preprocessing.py   # Image preprocessing
│   ├── model.py          # Neural network models
│   ├── train.py          # Training logic
│   ├── evaluate.py       # Evaluation metrics
│   └── utils.py          # Utility functions
├── data/                  # Dataset directory
│   ├── raw/              # Original images
│   └── processed/        # Preprocessed images
├── models/               # Saved models
├── notebooks/            # Jupyter notebooks
└── results/              # Training results and figures
```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/alzheimer-detection.git
cd alzheimer-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Kaggle API credentials
Create a `.env` file with your Kaggle credentials:
```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

Or place your `kaggle.json` file in `~/.kaggle/`

### 4. Download the dataset
```bash
python main.py --download
```

## Usage

### Run the complete pipeline
```bash
python main.py
```

### Run specific steps
```bash
# Download dataset only
python main.py --download

# Preprocess data
python main.py --preprocess

# Train model
python main.py --train --epochs 50 --batch-size 32 --lr 0.001

# Evaluate model
python main.py --evaluate
```

### Use different models
```bash
# Train with EfficientNet
python main.py --train --model-name efficientnet

# Train with Vision Transformer (requires timm)
pip install timm
python main.py --train --model-name vit
```

## Jupyter Notebook

For an interactive experience, use the provided Jupyter notebook:
```bash
jupyter notebook setup_and_run.ipynb
```

## Results

After training and evaluation, you'll find:
- Trained models in `models/saved_models/`
- Training logs in `results/logs/`
- Confusion matrix and ROC curves in `results/figures/`
- Classification report in `results/classification_report.txt`

## Performance

The model achieves competitive performance on the OASIS dataset:
- Overall accuracy: ~90%+ (varies by model and training parameters)
- Strong performance on Non-Demented class
- Good discrimination between dementia stages

## Citation

If you use this code or the OASIS dataset, please cite:
```
OASIS-1: Cross-Sectional: https://doi.org/10.1162/jocn.2007.19.9.1498
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OASIS dataset providers
- PyTorch team for the deep learning framework
- Kaggle for hosting the dataset