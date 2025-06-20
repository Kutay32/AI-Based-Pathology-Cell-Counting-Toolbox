"""
Wrapper script for using model.h5 for classification.

This script provides functions to load and use model.h5 for classification
without relying on the project's import structure.

Usage:
    from model_h5_wrapper import load_model_h5, classify_image

    # Load model
    model = load_model_h5("model.h5")

    # Classify image
    result = classify_image(model, image)
    print(f"Class: {result['predicted_class']}, Confidence: {result['probability']}")
"""

import os
import sys
import logging
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D as OriginalDepthwiseConv2D
from tensorflow.keras.layers import SpatialDropout2D as OriginalSpatialDropout2D
from tensorflow.keras.layers import Conv2DTranspose as OriginalConv2DTranspose
from keras.config import enable_unsafe_deserialization

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom DepthwiseConv2D layer that ignores 'groups' parameter
class CustomDepthwiseConv2D(OriginalDepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        # Remove 'groups' parameter if present
        if 'groups' in kwargs:
            logger.info("Removing 'groups' parameter from DepthwiseConv2D")
            kwargs.pop('groups')
        super().__init__(*args, **kwargs)

# Custom SpatialDropout2D layer that ignores incompatible parameters
class CustomSpatialDropout2D(OriginalSpatialDropout2D):
    def __init__(self, *args, **kwargs):
        # Remove incompatible parameters if present
        if 'trainable' in kwargs:
            logger.info("Removing 'trainable' parameter from SpatialDropout2D")
            kwargs.pop('trainable')
        if 'noise_shape' in kwargs:
            logger.info("Removing 'noise_shape' parameter from SpatialDropout2D")
            kwargs.pop('noise_shape')
        if 'seed' in kwargs:
            logger.info("Removing 'seed' parameter from SpatialDropout2D")
            kwargs.pop('seed')
        super().__init__(*args, **kwargs)

# Custom Conv2DTranspose layer that ignores 'groups' parameter
class CustomConv2DTranspose(OriginalConv2DTranspose):
    def __init__(self, *args, **kwargs):
        # Remove 'groups' parameter if present
        if 'groups' in kwargs:
            logger.info("Removing 'groups' parameter from Conv2DTranspose")
            kwargs.pop('groups')
        super().__init__(*args, **kwargs)

def load_model_h5(model_path):
    """
    Load model.h5 with compatibility layers.

    Args:
        model_path: Path to model.h5

    Returns:
        Loaded model
    """
    enable_unsafe_deserialization()

    # Check if model exists
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Custom objects for compatibility
    custom_objects = {
        'DepthwiseConv2D': CustomDepthwiseConv2D,
        'SpatialDropout2D': CustomSpatialDropout2D,
        'Conv2DTranspose': CustomConv2DTranspose
    }

    logger.info(f"Loading model from {model_path} with compatibility layers")
    model = load_model(model_path, compile=False, custom_objects=custom_objects)

    # Set model type to classification
    model._model_type = 'classification'

    return model

def preprocess_image(image, target_size=(512, 512), target_channels=6):
    """
    Preprocess image for model input.

    Args:
        image: Input image (numpy array or path to image file)
        target_size: Target size for resizing
        target_channels: Number of channels expected by the model

    Returns:
        Preprocessed image
    """
    # Load image if path is provided
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        if image is None:
            raise ValueError(f"Failed to load image from {image}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image
    resized_image = cv2.resize(image, target_size)

    # Normalize pixel values
    normalized_image = resized_image / 255.0

    # Handle channel mismatch
    current_channels = normalized_image.shape[-1] if len(normalized_image.shape) == 3 else 1

    if current_channels != target_channels:
        logger.info(f"Converting image from {current_channels} to {target_channels} channels")

        if current_channels == 1 and target_channels > 1:
            # Convert grayscale to multi-channel by repeating
            normalized_image = np.repeat(normalized_image[..., np.newaxis], target_channels, axis=-1)
        elif current_channels == 3 and target_channels == 6:
            # For 3 to 6 channels, we'll duplicate the RGB channels
            # This is a simple approach - in a real application, you might want a more sophisticated conversion
            rgb_channels = normalized_image
            normalized_image = np.concatenate([rgb_channels, rgb_channels], axis=-1)
        elif current_channels == 3 and target_channels > 3:
            # For 3 to more than 3 channels, we'll add additional channels with zeros
            additional_channels = np.zeros(normalized_image.shape[:-1] + (target_channels - current_channels,))
            normalized_image = np.concatenate([normalized_image, additional_channels], axis=-1)
        else:
            # For other conversions, we'll use a simple approach
            logger.warning(f"Using simple channel conversion from {current_channels} to {target_channels}")
            if current_channels > target_channels:
                # Take only the first target_channels
                normalized_image = normalized_image[..., :target_channels]
            else:
                # Add zeros for additional channels
                additional_channels = np.zeros(normalized_image.shape[:-1] + (target_channels - current_channels,))
                normalized_image = np.concatenate([normalized_image, additional_channels], axis=-1)

    # Add batch dimension
    input_image = np.expand_dims(normalized_image, axis=0)

    return input_image

def classify_image(model, image):
    """
    Classify image using model.h5.

    Args:
        model: Loaded model
        image: Input image (numpy array or path to image file)

    Returns:
        Dictionary with classification results
    """
    # Preprocess image
    input_image = preprocess_image(image)

    # Run prediction
    prediction = model.predict(input_image)

    # Handle different output formats
    if isinstance(prediction, list):
        # If model has multiple outputs, use the first one for classification
        class_probs = prediction[0][0]
    else:
        class_probs = prediction[0]

    # Get class with highest probability
    predicted_class = np.argmax(class_probs)
    probability = float(class_probs[predicted_class])

    # Return as dictionary
    return {
        'predicted_class': int(predicted_class),
        'probability': probability,
        'probabilities': class_probs.tolist()
    }

# Example usage
if __name__ == "__main__":
    try:
        # Parse command line arguments
        if len(sys.argv) < 2:
            print("Usage: python model_h5_wrapper.py <image_path>")
            sys.exit(1)

        image_path = sys.argv[1]

        # Load model
        model = load_model_h5("model.h5")

        # Classify image
        result = classify_image(model, image_path)

        # Print results
        print("\nClassification Results:")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['probability']:.4f}")

        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)
