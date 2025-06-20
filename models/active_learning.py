"""
Active Learning Module for Pathology Cell Counting Toolbox

This module implements active learning capabilities for continuous model improvement
through feedback loops with pathologist annotations. It includes functions for:
1. Uncertainty-based sample selection
2. Model fine-tuning with new annotations
3. Performance tracking over feedback iterations
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def entropy_based_uncertainty(predictions):
    """
    Calculate uncertainty based on prediction entropy.
    Higher entropy means higher uncertainty.
    
    Args:
        predictions: Model predictions (softmax outputs)
        
    Returns:
        Uncertainty scores for each prediction
    """
    # Calculate entropy: -sum(p_i * log(p_i))
    epsilon = 1e-10  # To avoid log(0)
    entropy = -np.sum(predictions * np.log(predictions + epsilon), axis=-1)
    return entropy

def select_uncertain_samples(predictions, images, top_k=10):
    """
    Select the most uncertain samples for annotation.
    
    Args:
        predictions: Model predictions
        images: Corresponding images
        top_k: Number of samples to select
        
    Returns:
        Indices of selected samples and their uncertainty scores
    """
    uncertainty_scores = entropy_based_uncertainty(predictions)
    
    # Get indices of top_k most uncertain predictions
    top_indices = np.argsort(uncertainty_scores)[-top_k:]
    
    return top_indices, uncertainty_scores[top_indices]

def fine_tune_model(model, new_images, new_masks, val_images=None, val_masks=None, 
                   learning_rate=1e-5, epochs=10, batch_size=8, model_save_path=None):
    """
    Fine-tune the model with newly annotated samples.
    
    Args:
        model: The model to fine-tune
        new_images: Newly annotated images
        new_masks: Corresponding masks
        val_images: Validation images
        val_masks: Validation masks
        learning_rate: Learning rate for fine-tuning
        epochs: Number of epochs for fine-tuning
        batch_size: Batch size for fine-tuning
        model_save_path: Path to save the fine-tuned model
        
    Returns:
        Fine-tuned model and training history
    """
    # If no validation data is provided, use a portion of the new data
    if val_images is None or val_masks is None:
        val_split = 0.2
        split_idx = int(len(new_images) * (1 - val_split))
        val_images = new_images[split_idx:]
        val_masks = new_masks[split_idx:]
        new_images = new_images[:split_idx]
        new_masks = new_masks[:split_idx]
    
    # Compile the model with a lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]
    
    # Add model checkpoint if save path is provided
    if model_save_path:
        callbacks.append(ModelCheckpoint(model_save_path, save_best_only=True, verbose=1))
    
    # Fine-tune the model
    history = model.fit(
        new_images, new_masks,
        validation_data=(val_images, val_masks),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return model, history

def active_learning_loop(model, unlabeled_images, labeled_images=None, labeled_masks=None,
                        iterations=5, samples_per_iteration=10, fine_tune_epochs=10,
                        batch_size=8, output_dir=None):
    """
    Run an active learning loop with simulated expert feedback.
    In a real-world scenario, this would involve interaction with pathologists.
    
    Args:
        model: Initial model
        unlabeled_images: Pool of unlabeled images
        labeled_images: Initial set of labeled images (if any)
        labeled_masks: Initial set of labeled masks (if any)
        iterations: Number of active learning iterations
        samples_per_iteration: Number of samples to select per iteration
        fine_tune_epochs: Number of epochs for fine-tuning in each iteration
        batch_size: Batch size for fine-tuning
        output_dir: Directory to save results
        
    Returns:
        Fine-tuned model and performance history
    """
    if labeled_images is None or labeled_masks is None:
        labeled_images = np.array([])
        labeled_masks = np.array([])
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        iteration_dir = os.path.join(output_dir, f"active_learning_{timestamp}")
        os.makedirs(iteration_dir, exist_ok=True)
    
    # Performance tracking
    performance_history = {
        'iteration': [],
        'accuracy': [],
        'loss': []
    }
    
    # Active learning iterations
    for i in range(iterations):
        print(f"\nActive Learning Iteration {i+1}/{iterations}")
        
        # Get predictions for unlabeled images
        predictions = model.predict(unlabeled_images)
        
        # Select the most uncertain samples
        selected_indices, uncertainty_scores = select_uncertain_samples(
            predictions, unlabeled_images, top_k=samples_per_iteration
        )
        
        print(f"Selected {len(selected_indices)} samples with uncertainty scores: {uncertainty_scores}")
        
        # In a real-world scenario, these samples would be sent to pathologists for annotation
        # Here we simulate this by using ground truth masks (which would come from experts)
        # This is a placeholder - in a real implementation, you would integrate with a UI for expert annotation
        
        # For demonstration, we'll assume we have access to ground truth masks for the selected samples
        # In practice, this would be replaced with actual expert annotations
        selected_images = unlabeled_images[selected_indices]
        
        # Simulate expert annotation (in real-world, this would be done by pathologists)
        # This is where you would integrate with your UI to collect expert feedback
        # For this example, we'll just use a placeholder function
        selected_masks = simulate_expert_annotation(selected_images)
        
        # Add newly labeled samples to the labeled dataset
        if len(labeled_images) == 0:
            labeled_images = selected_images
            labeled_masks = selected_masks
        else:
            labeled_images = np.concatenate([labeled_images, selected_images])
            labeled_masks = np.concatenate([labeled_masks, selected_masks])
        
        # Remove labeled samples from unlabeled pool
        unlabeled_images = np.delete(unlabeled_images, selected_indices, axis=0)
        
        # Fine-tune the model with the updated labeled dataset
        model_save_path = os.path.join(iteration_dir, f"model_iteration_{i+1}.keras") if output_dir else None
        model, history = fine_tune_model(
            model, labeled_images, labeled_masks,
            learning_rate=1e-5, epochs=fine_tune_epochs, batch_size=batch_size,
            model_save_path=model_save_path
        )
        
        # Track performance
        performance_history['iteration'].append(i+1)
        performance_history['accuracy'].append(history.history['val_accuracy'][-1])
        performance_history['loss'].append(history.history['val_loss'][-1])
        
        print(f"Iteration {i+1} completed. Validation accuracy: {performance_history['accuracy'][-1]:.4f}")
        
        # Save performance plot
        if output_dir:
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
            plt.savefig(os.path.join(iteration_dir, "active_learning_performance.png"))
    
    return model, performance_history

def simulate_expert_annotation(images):
    """
    Placeholder function to simulate expert annotation.
    In a real-world scenario, this would be replaced with actual pathologist input.
    
    Args:
        images: Images to be annotated
        
    Returns:
        Simulated masks (placeholder)
    """
    # This is just a placeholder - in a real implementation, this would be replaced
    # with actual expert annotations through a UI
    
    # For demonstration purposes, we'll create random masks
    # In practice, these would come from pathologists
    num_classes = 6  # Assuming 6 classes including background
    masks = np.zeros((len(images), images.shape[1], images.shape[2], num_classes))
    
    # Create random one-hot encoded masks (this is just a placeholder)
    for i in range(len(images)):
        # Create a random class mask (this would be the expert annotation in reality)
        random_mask = np.random.randint(0, num_classes, size=(images.shape[1], images.shape[2]))
        
        # Convert to one-hot encoding
        for c in range(num_classes):
            masks[i, :, :, c] = (random_mask == c).astype(np.float32)
    
    return masks