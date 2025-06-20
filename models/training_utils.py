"""
Training utilities for segmentation and classification models.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def train_segmentation_model(model, train_images, train_masks, val_images, val_masks, 
                            batch_size=16, epochs=50, learning_rate=1e-4, 
                            model_save_path="segmentation_model.keras", save_both_formats=False):
    """
    Train a segmentation model with appropriate callbacks and learning rate.

    Args:
        model: The segmentation model to train
        train_images: Training images
        train_masks: Training masks (categorical)
        val_images: Validation images
        val_masks: Validation masks (categorical)
        batch_size: Batch size for training
        epochs: Number of epochs to train
        learning_rate: Initial learning rate
        model_save_path: Path to save the best model weights
        save_both_formats: If True, save the model in both .h5 and .keras formats

    Returns:
        Training history
    """
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    checkpoint = ModelCheckpoint(
        model_save_path, 
        save_best_only=True, 
        verbose=1
    )
    earlystop = EarlyStopping(
        patience=10, 
        restore_best_weights=True
    )
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-6, 
        verbose=1
    )

    # Train the model
    history = model.fit(
        train_images, train_masks,
        validation_data=(val_images, val_masks),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint, earlystop, lr_scheduler]
    )

    # If save_both_formats is True, save the model in both formats
    if save_both_formats:
        try:
            from model_format_utils import save_model_dual_format
            # Get base path without extension
            base_path = os.path.splitext(model_save_path)[0]
            print(f"Saving model in both .h5 and .keras formats with base path {base_path}")
            h5_path, keras_path = save_model_dual_format(model, base_path)
            print(f"Model saved in .h5 format at {h5_path}")
            print(f"Model saved in .keras format at {keras_path}")
        except ImportError:
            print("model_format_utils not found, skipping dual format saving")
            print(f"Model saved only in the format specified by {model_save_path}")

    return history

def train_classification_model(model, train_images, train_labels, val_images, val_labels, 
                              batch_size=16, epochs=50, learning_rate=1e-4, 
                              model_save_path="classification_model.keras", save_both_formats=False):
    """
    Train a classification model with appropriate callbacks and learning rate.

    Args:
        model: The classification model to train
        train_images: Training images
        train_labels: Training labels (one-hot encoded)
        val_images: Validation images
        val_labels: Validation labels (one-hot encoded)
        batch_size: Batch size for training
        epochs: Number of epochs to train
        learning_rate: Initial learning rate
        model_save_path: Path to save the best model weights
        save_both_formats: If True, save the model in both .h5 and .keras formats

    Returns:
        Training history
    """
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    checkpoint = ModelCheckpoint(
        model_save_path, 
        save_best_only=True, 
        verbose=1
    )
    earlystop = EarlyStopping(
        patience=10, 
        restore_best_weights=True
    )
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-6, 
        verbose=1
    )

    # Train the model
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint, earlystop, lr_scheduler]
    )

    # If save_both_formats is True, save the model in both formats
    if save_both_formats:
        try:
            from model_format_utils import save_model_dual_format
            # Get base path without extension
            base_path = os.path.splitext(model_save_path)[0]
            print(f"Saving model in both .h5 and .keras formats with base path {base_path}")
            h5_path, keras_path = save_model_dual_format(model, base_path)
            print(f"Model saved in .h5 format at {h5_path}")
            print(f"Model saved in .keras format at {keras_path}")
        except ImportError:
            print("model_format_utils not found, skipping dual format saving")
            print(f"Model saved only in the format specified by {model_save_path}")

    return history

def plot_training_history(history, is_classification=False):
    """
    Plot training and validation accuracy and loss.

    Args:
        history: Training history object
        is_classification: Whether the history is from a classification model
    """
    # Plot accuracy
    plt.figure(figsize=(12, 5))
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
    plt.show()

def load_trained_model(model_path, custom_objects=None, model_type=None):
    """
    Load a trained model with custom objects.

    Args:
        model_path: Path to the saved model (.h5 or .keras)
        custom_objects: Dictionary of custom objects
        model_type: Type of model ('segmentation' or 'classification')

    Returns:
        Loaded model
    """
    # Try to import from model_format_utils first
    try:
        from model_format_utils import load_model_any_format
        print(f"Using model_format_utils.load_model_any_format to load {model_path}")
        return load_model_any_format(model_path, custom_objects, model_type)
    except ImportError:
        # Fall back to original implementation if model_format_utils is not available
        print(f"model_format_utils not found, using original implementation to load {model_path}")
        from tensorflow.keras.models import load_model
        from keras.config import enable_unsafe_deserialization

        enable_unsafe_deserialization()

        model = load_model(model_path, custom_objects=custom_objects)

        # Add a property to identify the model type
        if model_type:
            model._model_type = model_type

        return model
