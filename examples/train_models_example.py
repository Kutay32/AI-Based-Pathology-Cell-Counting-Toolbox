"""
Train Models Example Script

This script demonstrates how to train both segmentation and classification models
using the AI-Based Pathology Cell Counting Toolbox.

Usage:
    python train_models_example.py --dataset <path_to_dataset> --output <output_directory> --model-type <segmentation|classification|both>

Options:
    --dataset: Path to the dataset directory (should contain 'images' and 'labels' subdirectories)
    --output: Directory to save the trained models and training results
    --model-type: Type of model to train (segmentation, classification, or both)
    --batch-size: Batch size for training (default: 16)
    --epochs: Number of epochs to train (default: 50)
    --img-size: Size to resize images to (default: 256)
    --val-split: Validation split ratio (default: 0.2)
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Import project modules
from models.unet_models import build_unet_segmentation, build_unet_classification
from models.training_utils import train_segmentation_model, train_classification_model, plot_training_history
from models.losses import combined_loss, MeanIoUCustom
from utils.dataset_analysis import read_dataset_sample
from utils.helpers import get_default_class_names

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train segmentation and classification models")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output", type=str, required=True, help="Directory to save the trained models")
    parser.add_argument("--model-type", type=str, required=True, choices=["segmentation", "classification", "both"],
                        help="Type of model to train (segmentation, classification, or both)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--img-size", type=int, default=256, help="Size to resize images to")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    return parser.parse_args()

def preprocess_data_for_segmentation(images, masks, img_size):
    """
    Preprocess images and masks for segmentation model training.

    Args:
        images: List of images
        masks: List of masks
        img_size: Size to resize images to

    Returns:
        Preprocessed images and masks as numpy arrays
    """
    # Initialize arrays
    X = np.zeros((len(images), img_size, img_size, 3), dtype=np.float32)
    y = np.zeros((len(masks), img_size, img_size), dtype=np.uint8)

    # Process each image and mask
    for i, (img, mask) in enumerate(zip(images, masks)):
        # Resize image and normalize
        img_resized = cv2.resize(img, (img_size, img_size))
        X[i] = img_resized / 255.0  # Normalize to [0, 1]

        # Resize mask
        mask_resized = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

        # Ensure mask values are within the expected range (0-5)
        mask_resized = np.clip(mask_resized, 0, 5)

        y[i] = mask_resized

    return X, y

def preprocess_data_for_classification(images, masks, img_size):
    """
    Preprocess images and extract class labels for classification model training.
    For classification, we'll use the most common non-background class in each mask as the label.

    Args:
        images: List of images
        masks: List of masks
        img_size: Size to resize images to

    Returns:
        Preprocessed images and class labels as numpy arrays
    """
    # Initialize arrays
    X = np.zeros((len(images), img_size, img_size, 3), dtype=np.float32)
    y = np.zeros(len(masks), dtype=np.uint8)

    # Process each image and mask
    for i, (img, mask) in enumerate(zip(images, masks)):
        # Resize image and normalize
        img_resized = cv2.resize(img, (img_size, img_size))
        X[i] = img_resized / 255.0  # Normalize to [0, 1]

        # Extract the most common non-background class (assuming 0 is background)
        # Count occurrences of each class
        unique, counts = np.unique(mask, return_counts=True)
        class_counts = dict(zip(unique, counts))
        
        # Remove background class (0) if present
        if 0 in class_counts:
            del class_counts[0]
        
        # If no non-background classes, default to class 1
        if not class_counts:
            y[i] = 1
        else:
            # Get the most common non-background class
            y[i] = max(class_counts, key=class_counts.get)

    return X, y

def train_segmentation(dataset, args):
    """
    Train a segmentation model.

    Args:
        dataset: Dictionary containing images and labels
        args: Command-line arguments

    Returns:
        Trained segmentation model and training history
    """
    print("\n=== Training Segmentation Model ===")
    
    # Preprocess data for segmentation
    print("Preprocessing data for segmentation...")
    X, y = preprocess_data_for_segmentation(dataset['images'], dataset['labels'], args.img_size)

    # Determine number of classes from default class names
    class_names = get_default_class_names()
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes} ({', '.join(class_names)})")

    # Convert masks to categorical
    y_cat = to_categorical(y, num_classes=num_classes)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=args.val_split, random_state=42
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")

    # Build model
    print("Building segmentation model...")
    model = build_unet_segmentation(
        input_shape=(args.img_size, args.img_size, 3),
        num_classes=num_classes
    )

    # Train model
    print("Training segmentation model...")
    model_save_path = os.path.join(args.output, "segmentation_model.keras")
    history = train_segmentation_model(
        model=model,
        train_images=X_train,
        train_masks=y_train,
        val_images=X_val,
        val_masks=y_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_save_path=model_save_path
    )

    # Save training history plot
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Segmentation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Segmentation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "segmentation_training_history.png"))
    
    print(f"Segmentation model trained and saved to {model_save_path}")
    print(f"Training history plot saved to {os.path.join(args.output, 'segmentation_training_history.png')}")
    
    return model, history

def train_classification(dataset, args):
    """
    Train a classification model.

    Args:
        dataset: Dictionary containing images and labels
        args: Command-line arguments

    Returns:
        Trained classification model and training history
    """
    print("\n=== Training Classification Model ===")
    
    # Preprocess data for classification
    print("Preprocessing data for classification...")
    X, y = preprocess_data_for_classification(dataset['images'], dataset['labels'], args.img_size)

    # Determine number of classes from default class names
    class_names = get_default_class_names()
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes} ({', '.join(class_names)})")

    # Convert labels to categorical
    y_cat = to_categorical(y, num_classes=num_classes)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=args.val_split, random_state=42
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")

    # Build model
    print("Building classification model...")
    model = build_unet_classification(
        input_shape=(args.img_size, args.img_size, 3),
        num_classes=num_classes
    )

    # Train model
    print("Training classification model...")
    model_save_path = os.path.join(args.output, "classification_model.keras")
    history = train_classification_model(
        model=model,
        train_images=X_train,
        train_labels=y_train,
        val_images=X_val,
        val_labels=y_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_save_path=model_save_path
    )

    # Save training history plot
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "classification_training_history.png"))
    
    print(f"Classification model trained and saved to {model_save_path}")
    print(f"Training history plot saved to {os.path.join(args.output, 'classification_training_history.png')}")
    
    return model, history

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    print(f"Loading dataset from {args.dataset}...")

    # Read dataset
    dataset = read_dataset_sample(args.dataset, max_files=1000)

    if not dataset['images'] or not dataset['labels']:
        print("Error: No images or labels found in the dataset.")
        return

    print(f"Found {len(dataset['images'])} images and {len(dataset['labels'])} labels.")

    # Train models based on the specified model type
    if args.model_type in ["segmentation", "both"]:
        train_segmentation(dataset, args)
    
    if args.model_type in ["classification", "both"]:
        train_classification(dataset, args)

    print("Training complete!")

if __name__ == "__main__":
    main()