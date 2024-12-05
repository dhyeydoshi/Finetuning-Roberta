# Finetuning RoBERTa for Sentiment Analysis

## Overview
This project focuses on fine-tuning a sentiment analysis model using data augmentation techniques. The primary goal is to enhance the model's performance by augmenting the training data with back-translation and synonym replacement methods.

## Notebooks
The project consists of three main Jupyter notebooks:

1.`Data Collection.ipynb`
2. `Data Pre-processing.ipynb`
3. `Finetuning for Sentiment Analysis.ipynb`

### 1. Data Pre-processing.ipynb
This notebook handles the data augmentation process, including back-translation and synonym replacement.

#### Key Sections:
- **Augmentation Ratio and Parallel Processing Parameters**: Defines the ratio of data to be augmented and the number of parallel processes.
- **Batch Augmentation Function**: Applies augmentation to a subset of the dataset.
- **Back-Translation with Multi-GPU Support**: Utilizes multiple GPUs for back-translation.
- **Data Augmentation - Synonym Replacement**: Applies synonym replacement to the dataset.
- **Saving Augmented Datasets**: Saves the augmented datasets to disk.

#### Important Functions:
- `apply_batch_augmentation_with_ratio(dataset, aug_func, batch_size=64)`: Applies augmentation to a specified ratio of the dataset.
- `back_translation_multi_gpus()`: Performs back-translation using multiple GPUs.

### 2. Finetuning for Sentiment Analysis.ipynb
This notebook focuses on fine-tuning a sentiment analysis model using the augmented datasets.

#### Key Sections:
- **Environment Setup**: Checks for CUDA availability and sets up the environment for multiprocessing.
- **Loading Datasets**: Loads the original and augmented datasets from disk.
- **Data Preprocessing**: Tokenizes the datasets and prepares them for training.
- **Model Training**: Fine-tunes the sentiment analysis model using the augmented datasets.
- **Evaluation**: Evaluates the model's performance on the validation dataset.

#### Important Functions:
- `preprocess(batch)`: Tokenizes the input text and prepares it for the model.
- `train_func()`: Trains the sentiment analysis model with early stopping and saves the best model.

## Installation
To run the notebooks, you need to install the following dependencies:
```bash
pip install torch torchvision transformers datasets accelerate
