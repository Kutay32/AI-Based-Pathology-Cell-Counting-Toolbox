"""
Train Model Example Script

This script demonstrates how to train a model on a dataset using the AI-Based Pathology
Cell Counting Toolbox. It loads data from the dataset, preprocesses it, trains the model,
and saves the trained model.

Usage:
    python train_model_example.py --dataset <path_to_dataset> --output <output_directory>

Options:
    --dataset: Path to the dataset directory (should contain 'images' and 'labels' subdirectories)
    --output: Directory to save the trained model and training results
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
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import project modules
from models.unet import build_attention_unet
from models.training import train_model, plot_training_history
from models.losses import combined_loss, MeanIoUCustom
from utils.dataset_analysis import read_dataset_sample
from utils.helpers import get_default_class_names
from models.active_learning import active_learning_loop

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a model on a dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output", type=str, required=True, help="Directory to save the trained model")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--img-size", type=int, default=256, help="Size to resize images to")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--quantize", action="store_true", help="Quantize model for edge deployment")
    parser.add_argument("--active-learning", action="store_true", help="Enable active learning mode")
    return parser.parse_args()

def preprocess_data(images, masks, img_size):
    """
    Preprocess images and masks for training.

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
        # Clip values to the range 0-5
        mask_resized = np.clip(mask_resized, 0, 5)

        y[i] = mask_resized

    return X, y

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

    # Preprocess data
    print("Preprocessing data...")
    X, y = preprocess_data(dataset['images'], dataset['labels'], args.img_size)

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
    print("Building model...")
    model = build_attention_unet(
        input_shape=(args.img_size, args.img_size, 3),
        num_classes=num_classes
    )

    # Train model
    print("Training model...")
    model_save_path = os.path.join(args.output, "model_weights.keras")

    if args.augment:
        print("Using data augmentation...")
        # Create data generators with augmentation
        data_gen_args = dict(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)

        # Fit the generators
        image_datagen.fit(X_train)
        mask_datagen.fit(y_train)

        # Create generators
        image_generator = image_datagen.flow(X_train, batch_size=args.batch_size, seed=42)
        mask_generator = mask_datagen.flow(y_train, batch_size=args.batch_size, seed=42)

        # Combine generators
        train_generator = zip(image_generator, mask_generator)

        # Train with generators
        history = model.fit(
            train_generator,
            steps_per_epoch=len(X_train) // args.batch_size,
            epochs=args.epochs,
            validation_data=(X_val, y_val),
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, verbose=1),
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
            ]
        )
    else:
        # Train without augmentation
        history = train_model(
            model=model,
            train_images=X_train,
            train_masks=y_train,
            val_images=X_val,
            val_masks=y_val,
            batch_size=args.batch_size,
            epochs=args.epochs,
            model_save_path=model_save_path
        )

    # Plot and save training history
    print("Saving training history...")
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "training_history.png"))

    print(f"Model trained and saved to {model_save_path}")
    print(f"Training history plot saved to {os.path.join(args.output, 'training_history.png')}")

    # Apply active learning if requested
    if args.active_learning:
        print("\nStarting active learning process...")

        # Create a directory for active learning results
        active_learning_dir = os.path.join(args.output, "active_learning")
        os.makedirs(active_learning_dir, exist_ok=True)

        # Split the validation set into a labeled set and an unlabeled pool
        # In a real scenario, you would have a larger unlabeled pool
        initial_labeled_size = min(20, len(X_val) // 5)  # Start with a small labeled set

        # Initial labeled set (simulating initial expert annotations)
        X_labeled = X_val[:initial_labeled_size]
        y_labeled = y_val[:initial_labeled_size]

        # Unlabeled pool (remaining validation data)
        X_unlabeled = X_val[initial_labeled_size:]

        print(f"Initial labeled set: {len(X_labeled)} samples")
        print(f"Unlabeled pool: {len(X_unlabeled)} samples")

        # Run active learning loop
        model, performance_history = active_learning_loop(
            model=model,
            unlabeled_images=X_unlabeled,
            labeled_images=X_labeled,
            labeled_masks=y_labeled,
            iterations=3,  # Number of active learning iterations
            samples_per_iteration=10,  # Number of samples to select per iteration
            fine_tune_epochs=5,  # Number of epochs for fine-tuning in each iteration
            batch_size=args.batch_size,
            output_dir=active_learning_dir
        )

        print("\nActive learning completed.")
        print(f"Final validation accuracy: {performance_history['accuracy'][-1]:.4f}")

        # Save the final model
        final_model_path = os.path.join(active_learning_dir, "final_model.keras")
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")

        # Plot performance history
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(performance_history['iteration'], performance_history['accuracy'])
        plt.title('Validation Accuracy Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(performance_history['iteration'], performance_history['loss'])
        plt.title('Validation Loss Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(active_learning_dir, "active_learning_performance.png"))

    # Quantize model for edge deployment if requested
    if args.quantize:
        print("\nQuantizing model for edge deployment...")

        # Define a representative dataset generator
        def representative_dataset():
            for i in range(min(100, len(X_val))):
                yield [np.expand_dims(X_val[i], axis=0)]

        # Convert the model to TensorFlow Lite format
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Apply quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset

        # Ensure full integer quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        # Convert the model
        quantized_tflite_model = converter.convert()

        # Save the quantized model
        quantized_model_path = os.path.join(args.output, "quantized_model.tflite")
        with open(quantized_model_path, 'wb') as f:
            f.write(quantized_tflite_model)

        print(f"Quantized model saved to {quantized_model_path}")

        # Calculate model size reduction
        original_size = os.path.getsize(model_save_path)
        quantized_size = os.path.getsize(quantized_model_path)
        size_reduction = (1 - quantized_size / original_size) * 100

        print(f"Original model size: {original_size / 1024:.2f} KB")
        print(f"Quantized model size: {quantized_size / 1024:.2f} KB")
        print(f"Size reduction: {size_reduction:.2f}%")

if __name__ == "__main__":
    main()
