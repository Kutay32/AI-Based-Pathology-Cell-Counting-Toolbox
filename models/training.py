"""
Training utilities for cell segmentation models.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def train_model(model, train_images, train_masks, val_images, val_masks, 
                batch_size=16, epochs=50, learning_rate=1e-4, 
                model_save_path="model_weights.keras"):
    """
    Train a segmentation model with appropriate callbacks and learning rate.
    
    Args:
        model: The model to train
        train_images: Training images
        train_masks: Training masks
        val_images: Validation images
        val_masks: Validation masks
        batch_size: Batch size for training
        epochs: Number of epochs to train
        learning_rate: Initial learning rate
        model_save_path: Path to save the best model weights
        
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
    
    return history

def plot_training_history(history):
    """
    Plot training and validation accuracy and loss.
    
    Args:
        history: Training history object
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

def load_trained_model(model_path, custom_objects=None):
    """
    Load a trained model with custom objects.
    
    Args:
        model_path: Path to the saved model
        custom_objects: Dictionary of custom objects
        
    Returns:
        Loaded model
    """
    from tensorflow.keras.models import load_model
    from keras.config import enable_unsafe_deserialization
    
    enable_unsafe_deserialization()
    
    model = load_model(model_path, custom_objects=custom_objects)
    return model