"""
Inference utilities for segmentation and classification models.
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.config import enable_unsafe_deserialization
from tensorflow.keras.layers import DepthwiseConv2D as OriginalDepthwiseConv2D
from tensorflow.keras.layers import SpatialDropout2D as OriginalSpatialDropout2D
from tensorflow.keras.layers import Conv2DTranspose as OriginalConv2DTranspose
import logging

# Configure logging
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
        logger.info(f"Using model_format_utils.load_model_any_format to load {model_path}")
        return load_model_any_format(model_path, custom_objects, model_type)
    except ImportError:
        # Fall back to original implementation if model_format_utils is not available
        logger.info(f"model_format_utils not found, using original implementation to load {model_path}")
        enable_unsafe_deserialization()

        # Add compatibility layers to custom_objects
        if custom_objects is None:
            custom_objects = {}

        # Add our custom compatibility layers
        compatibility_layers = {
            'DepthwiseConv2D': CustomDepthwiseConv2D,
            'SpatialDropout2D': CustomSpatialDropout2D,
            'Conv2DTranspose': CustomConv2DTranspose
        }

        # Merge with any user-provided custom objects
        for key, value in compatibility_layers.items():
            if key not in custom_objects:
                custom_objects[key] = value

        logger.info(f"Loading model from {model_path} with compatibility layers")
        model = load_model(model_path, custom_objects=custom_objects)

        # Add a property to identify the model type if provided
        if model_type:
            model._model_type = model_type
        # Try to infer model type from model architecture if not provided
        elif not hasattr(model, '_model_type'):
            # Check if the model has a classification head (Dense layer at the end)
            if any(isinstance(layer, tf.keras.layers.Dense) for layer in model.layers[-3:]):
                model._model_type = 'classification'
            else:
                model._model_type = 'segmentation'

        return model

def preprocess_image_for_model(image, model):
    """
    Preprocess an image for model inference.

    Args:
        image: Input image
        model: Model to use for inference

    Returns:
        Preprocessed image ready for model input
    """
    # Resize image to match model input shape
    input_shape = model.input_shape[1:3]
    resized_image = cv2.resize(image, input_shape)

    # Normalize pixel values
    normalized_image = resized_image / 255.0

    # Convert to grayscale if the model expects 1 channel input
    if model.input_shape[-1] == 1 and len(normalized_image.shape) == 3 and normalized_image.shape[-1] == 3:
        normalized_image = cv2.cvtColor(normalized_image, cv2.COLOR_RGB2GRAY)
        normalized_image = np.expand_dims(normalized_image, axis=-1)

    # Add batch dimension
    input_image = np.expand_dims(normalized_image, axis=0)

    return input_image

def predict_segmentation(model, image, confidence_threshold=0.3, classification_result=None):
    """
    Run segmentation prediction on an image.

    Args:
        model: Segmentation model
        image: Input image
        confidence_threshold: Minimum confidence threshold for class prediction (default: 0.3)
        classification_result: Optional classification result to guide segmentation

    Returns:
        Predicted segmentation mask
    """
    # Check if model is a segmentation model
    if hasattr(model, '_model_type') and model._model_type != 'segmentation':
        raise ValueError("Model is not a segmentation model")

    # Preprocess image
    input_image = preprocess_image_for_model(image, model)

    # Run prediction
    prediction = model.predict(input_image)

    # Create a mask with background (class 0) by default
    pred_probs = prediction[0]
    pred_mask = np.zeros(pred_probs.shape[:-1], dtype=np.uint8)

    # Print information about the prediction
    print(f"Prediction shape: {pred_probs.shape}")
    print(f"Max probability: {np.max(pred_probs)}")
    print(f"Min probability: {np.min(pred_probs)}")
    print(f"Mean probability: {np.mean(pred_probs)}")

    # Check if we have any valid predictions
    if np.max(pred_probs) < confidence_threshold:
        print(f"Warning: All probabilities are below the confidence threshold ({confidence_threshold})")
        # Lower the threshold to get at least some predictions
        confidence_threshold = max(0.1, np.max(pred_probs) * 0.8)
        print(f"Lowering threshold to {confidence_threshold}")

    # Apply class-specific thresholds if classification result is provided
    if classification_result is not None and 'predicted_class' in classification_result:
        # Get the predicted class from classification
        predicted_class = classification_result['predicted_class']
        print(f"Using classification result: class={predicted_class}, probability={classification_result.get('probability', 'N/A')}")

        # Use a lower threshold for the class predicted by classification
        class_thresholds = np.ones(pred_probs.shape[-1]) * confidence_threshold

        # Lower threshold for the predicted class (but keep it reasonable)
        class_thresholds[predicted_class] = max(0.1, confidence_threshold * 0.5)
        print(f"Class thresholds: {class_thresholds}")

        # Apply class-specific thresholds
        for class_id in range(1, pred_probs.shape[-1]):  # Skip background
            class_mask = (np.argmax(pred_probs, axis=-1) == class_id) & (pred_probs[..., class_id] > class_thresholds[class_id])
            pred_mask[class_mask] = class_id
            # Print the number of pixels assigned to this class
            print(f"Class {class_id}: {np.sum(class_mask)} pixels")
    else:
        # Standard approach with uniform threshold
        max_probs = np.max(pred_probs, axis=-1)
        max_classes = np.argmax(pred_probs, axis=-1)

        # Where probability exceeds threshold, assign the predicted class
        pred_mask = np.where(max_probs > confidence_threshold, max_classes, pred_mask)

        # Print the number of pixels assigned to each class
        for class_id in range(1, pred_probs.shape[-1]):  # Skip background
            print(f"Class {class_id}: {np.sum(pred_mask == class_id)} pixels")

    # Ensure we have at least some non-background pixels
    if np.sum(pred_mask > 0) == 0:
        print("Warning: No non-background pixels in the mask")
        # Assign the most probable class to at least some pixels
        max_classes = np.argmax(pred_probs, axis=-1)
        max_probs = np.max(pred_probs, axis=-1)

        # Get the top 10% of pixels by probability
        threshold = np.percentile(max_probs, 90)
        high_prob_mask = max_probs > threshold

        # Assign these pixels to their most probable class
        pred_mask[high_prob_mask] = max_classes[high_prob_mask]
        print(f"Assigned {np.sum(high_prob_mask)} pixels to their most probable class")

        # Print the number of pixels assigned to each class
        for class_id in range(1, pred_probs.shape[-1]):  # Skip background
            print(f"Class {class_id}: {np.sum(pred_mask == class_id)} pixels")

    return pred_mask

def predict_classification(model, image):
    """
    Run classification prediction on an image.

    Args:
        model: Classification model
        image: Input image

    Returns:
        Predicted class probabilities
    """
    # Check if model is a classification model
    if hasattr(model, '_model_type') and model._model_type != 'classification':
        raise ValueError("Model is not a classification model")

    # Preprocess image
    input_image = preprocess_image_for_model(image, model)

    # Run prediction
    prediction = model.predict(input_image)

    # Return class probabilities
    return prediction[0]

def get_predicted_class(model, image):
    """
    Get the predicted class for an image.

    Args:
        model: Classification model
        image: Input image

    Returns:
        Dictionary containing predicted class index, probability, and all class probabilities
    """
    # Get class probabilities
    class_probs = predict_classification(model, image)

    # Get class with highest probability
    predicted_class = np.argmax(class_probs)
    probability = class_probs[predicted_class]

    # Return as dictionary for better compatibility with UI code
    return {
        'predicted_class': predicted_class,
        'probability': probability,
        'probabilities': class_probs
    }
