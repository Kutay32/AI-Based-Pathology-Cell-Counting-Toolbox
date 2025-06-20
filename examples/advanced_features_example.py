"""
Advanced Features Example Script

This script demonstrates how to use the advanced features of the AI-Based Pathology
Cell Counting Toolbox, including data augmentation, active learning, and edge AI support.

Usage:
    python examples/advanced_features_example.py --dataset <path_to_dataset> --output <output_directory>

Options:
    --dataset: Path to the dataset directory
    --output: Directory to save the results
    --feature: Feature to demonstrate ('augmentation', 'active_learning', 'edge_ai', or 'all')
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import project modules
from models.unet import build_attention_unet
from models.training import train_model
from models.active_learning import active_learning_loop
from utils.dataset_analysis import read_dataset_sample
from utils.helpers import get_default_class_names
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Demonstrate advanced features")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output", type=str, required=True, help="Directory to save the results")
    parser.add_argument("--feature", type=str, default="all", 
                        choices=["augmentation", "active_learning", "edge_ai", "all"],
                        help="Feature to demonstrate")
    return parser.parse_args()

def preprocess_data(images, masks, img_size=256):
    """Preprocess images and masks for training."""
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
        mask_resized = np.clip(mask_resized, 0, 5)  # Ensure values are within range
        y[i] = mask_resized

    return X, y

def demonstrate_data_augmentation(X_train, y_train, X_val, y_val, output_dir, batch_size=8, epochs=5):
    """Demonstrate data augmentation."""
    print("\n=== Demonstrating Data Augmentation ===")
    
    # Create output directory
    augmentation_dir = os.path.join(output_dir, "augmentation")
    os.makedirs(augmentation_dir, exist_ok=True)
    
    # Build model
    model = build_attention_unet(
        input_shape=(X_train.shape[1], X_train.shape[2], 3),
        num_classes=y_train.shape[-1]
    )
    
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
    image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=42)
    mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=42)

    # Combine generators
    train_generator = zip(image_generator, mask_generator)
    
    # Visualize augmented samples
    plt.figure(figsize=(15, 10))
    
    # Get a batch of augmented images and masks
    batch_x, batch_y = next(train_generator)
    
    # Display 5 augmented samples
    for i in range(min(5, batch_x.shape[0])):
        plt.subplot(2, 5, i+1)
        plt.imshow(batch_x[i])
        plt.title(f"Augmented Image {i+1}")
        plt.axis('off')
        
        plt.subplot(2, 5, i+6)
        # Display the mask (take argmax for visualization)
        plt.imshow(np.argmax(batch_y[i], axis=-1), cmap='viridis')
        plt.title(f"Augmented Mask {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(augmentation_dir, "augmented_samples.png"))
    plt.close()
    
    print(f"Augmented samples saved to {os.path.join(augmentation_dir, 'augmented_samples.png')}")
    
    # Train with augmentation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train with generators
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(augmentation_dir, "model.keras"), 
                save_best_only=True
            )
        ]
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy with Data Augmentation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss with Data Augmentation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(augmentation_dir, "training_history.png"))
    plt.close()
    
    print(f"Training history saved to {os.path.join(augmentation_dir, 'training_history.png')}")
    print(f"Model saved to {os.path.join(augmentation_dir, 'model.keras')}")
    
    return model

def demonstrate_active_learning(X_train, y_train, X_val, y_val, output_dir, batch_size=8):
    """Demonstrate active learning."""
    print("\n=== Demonstrating Active Learning ===")
    
    # Create output directory
    active_learning_dir = os.path.join(output_dir, "active_learning")
    os.makedirs(active_learning_dir, exist_ok=True)
    
    # Build model
    model = build_attention_unet(
        input_shape=(X_train.shape[1], X_train.shape[2], 3),
        num_classes=y_train.shape[-1]
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model with a small subset of data
    initial_size = min(20, len(X_train) // 5)
    X_initial = X_train[:initial_size]
    y_initial = y_train[:initial_size]
    
    # Initial training
    model.fit(
        X_initial, y_initial,
        batch_size=batch_size,
        epochs=3,
        verbose=1
    )
    
    # Run active learning
    # Split validation data into labeled and unlabeled sets
    X_labeled = X_val[:10]
    y_labeled = y_val[:10]
    X_unlabeled = X_val[10:]
    
    # Run active learning loop
    model, performance_history = active_learning_loop(
        model=model,
        unlabeled_images=X_unlabeled,
        labeled_images=X_labeled,
        labeled_masks=y_labeled,
        iterations=2,  # Small number for demonstration
        samples_per_iteration=5,
        fine_tune_epochs=2,
        batch_size=batch_size,
        output_dir=active_learning_dir
    )
    
    # Save final model
    model.save(os.path.join(active_learning_dir, "final_model.keras"))
    
    print(f"Active learning results saved to {active_learning_dir}")
    print(f"Final model saved to {os.path.join(active_learning_dir, 'final_model.keras')}")
    
    return model

def demonstrate_edge_ai(model, X_val, output_dir):
    """Demonstrate edge AI support through model quantization."""
    print("\n=== Demonstrating Edge AI Support ===")
    
    # Create output directory
    edge_ai_dir = os.path.join(output_dir, "edge_ai")
    os.makedirs(edge_ai_dir, exist_ok=True)
    
    # Save original model for comparison
    original_model_path = os.path.join(edge_ai_dir, "original_model.keras")
    model.save(original_model_path)
    
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
    quantized_model_path = os.path.join(edge_ai_dir, "quantized_model.tflite")
    with open(quantized_model_path, 'wb') as f:
        f.write(quantized_tflite_model)
    
    # Calculate model size reduction
    original_size = os.path.getsize(original_model_path)
    quantized_size = os.path.getsize(quantized_model_path)
    size_reduction = (1 - quantized_size / original_size) * 100
    
    # Create a report
    with open(os.path.join(edge_ai_dir, "quantization_report.txt"), 'w') as f:
        f.write("Edge AI Model Quantization Report\n")
        f.write("================================\n\n")
        f.write(f"Original model size: {original_size / 1024:.2f} KB\n")
        f.write(f"Quantized model size: {quantized_size / 1024:.2f} KB\n")
        f.write(f"Size reduction: {size_reduction:.2f}%\n\n")
        f.write("The quantized model can be deployed on edge devices for faster inference.\n")
    
    print(f"Quantized model saved to {quantized_model_path}")
    print(f"Original model size: {original_size / 1024:.2f} KB")
    print(f"Quantized model size: {quantized_size / 1024:.2f} KB")
    print(f"Size reduction: {size_reduction:.2f}%")
    
    return quantized_model_path

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading dataset from {args.dataset}...")
    
    # Read dataset
    dataset = read_dataset_sample(args.dataset, max_files=100)  # Limit to 100 files for demonstration
    
    if not dataset['images'] or not dataset['labels']:
        print("Error: No images or labels found in the dataset.")
        return
    
    print(f"Found {len(dataset['images'])} images and {len(dataset['labels'])} labels.")
    
    # Preprocess data
    print("Preprocessing data...")
    X, y = preprocess_data(dataset['images'], dataset['labels'])
    
    # Determine number of classes
    class_names = get_default_class_names()
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes} ({', '.join(class_names)})")
    
    # Convert masks to categorical
    y_cat = to_categorical(y, num_classes=num_classes)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Demonstrate selected features
    if args.feature == "augmentation" or args.feature == "all":
        model = demonstrate_data_augmentation(X_train, y_train, X_val, y_val, args.output)
    
    if args.feature == "active_learning" or args.feature == "all":
        model = demonstrate_active_learning(X_train, y_train, X_val, y_val, args.output)
    
    if args.feature == "edge_ai" or args.feature == "all":
        # If we haven't trained a model yet, create one
        if 'model' not in locals():
            model = build_attention_unet(
                input_shape=(X_train.shape[1], X_train.shape[2], 3),
                num_classes=y_train.shape[-1]
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            # Train briefly just for demonstration
            model.fit(X_train[:20], y_train[:20], epochs=1, verbose=0)
        
        demonstrate_edge_ai(model, X_val, args.output)
    
    print("\nDemonstration completed successfully!")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    import cv2  # Import here to avoid issues if not available
    main()