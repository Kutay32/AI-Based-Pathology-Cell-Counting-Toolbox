"""
Utility functions for handling model formats (.h5 and .keras).

This module provides functions to:
1. Convert models between .h5 and .keras formats
2. Load models from either format
3. Save models in either format

Usage:
    from model_format_utils import convert_model_format, load_model_any_format, save_model_dual_format

    # Convert model from one format to another
    convert_model_format("model.h5", "model.keras")

    # Load model from either format
    model = load_model_any_format("model.h5")  # or "model.keras"

    # Save model in both formats
    save_model_dual_format(model, "model_base")  # Creates model_base.h5 and model_base.keras
"""

import os
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.config import enable_unsafe_deserialization
from tensorflow.keras.layers import DepthwiseConv2D as OriginalDepthwiseConv2D
from tensorflow.keras.layers import SpatialDropout2D as OriginalSpatialDropout2D
from tensorflow.keras.layers import Conv2DTranspose as OriginalConv2DTranspose

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom compatibility layers (same as in model_h5_wrapper.py and inference.py)
class CustomDepthwiseConv2D(OriginalDepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        # Remove 'groups' parameter if present
        if 'groups' in kwargs:
            logger.info("Removing 'groups' parameter from DepthwiseConv2D")
            kwargs.pop('groups')
        super().__init__(*args, **kwargs)

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

class CustomConv2DTranspose(OriginalConv2DTranspose):
    def __init__(self, *args, **kwargs):
        # Remove 'groups' parameter if present
        if 'groups' in kwargs:
            logger.info("Removing 'groups' parameter from Conv2DTranspose")
            kwargs.pop('groups')
        super().__init__(*args, **kwargs)

def get_custom_objects():
    """
    Get dictionary of custom objects for model loading.
    
    Returns:
        Dictionary of custom objects
    """
    return {
        'DepthwiseConv2D': CustomDepthwiseConv2D,
        'SpatialDropout2D': CustomSpatialDropout2D,
        'Conv2DTranspose': CustomConv2DTranspose
    }

def load_model_any_format(model_path, custom_objects=None, model_type=None):
    """
    Load a model from either .h5 or .keras format.
    
    Args:
        model_path: Path to the model file (.h5 or .keras)
        custom_objects: Dictionary of custom objects (optional)
        model_type: Type of model ('segmentation' or 'classification') (optional)
        
    Returns:
        Loaded model
    """
    # Check if file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Enable unsafe deserialization for compatibility
    enable_unsafe_deserialization()
    
    # Merge custom objects
    if custom_objects is None:
        custom_objects = {}
    
    # Add our compatibility layers
    for key, value in get_custom_objects().items():
        if key not in custom_objects:
            custom_objects[key] = value
    
    # Load the model
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path, custom_objects=custom_objects, compile=False)
    
    # Add model type if provided
    if model_type:
        model._model_type = model_type
    # Try to infer model type if not provided
    elif not hasattr(model, '_model_type'):
        # Check if the model has a classification head (Dense layer at the end)
        if any(isinstance(layer, tf.keras.layers.Dense) for layer in model.layers[-3:]):
            model._model_type = 'classification'
        else:
            model._model_type = 'segmentation'
    
    return model

def convert_model_format(input_path, output_path, custom_objects=None):
    """
    Convert a model from one format to another (.h5 to .keras or vice versa).
    
    Args:
        input_path: Path to the input model file
        output_path: Path to save the converted model
        custom_objects: Dictionary of custom objects (optional)
        
    Returns:
        Path to the converted model
    """
    # Load the model
    model = load_model_any_format(input_path, custom_objects)
    
    # Save in the new format
    logger.info(f"Saving model to {output_path}")
    model.save(output_path, save_format='tf' if output_path.endswith('.keras') else 'h5')
    
    return output_path

def save_model_dual_format(model, base_path, custom_objects=None):
    """
    Save a model in both .h5 and .keras formats.
    
    Args:
        model: The model to save
        base_path: Base path for saving (without extension)
        custom_objects: Dictionary of custom objects (optional)
        
    Returns:
        Tuple of paths to the saved models (h5_path, keras_path)
    """
    # Save in .h5 format
    h5_path = f"{base_path}.h5"
    logger.info(f"Saving model in .h5 format to {h5_path}")
    model.save(h5_path, save_format='h5')
    
    # Save in .keras format
    keras_path = f"{base_path}.keras"
    logger.info(f"Saving model in .keras format to {keras_path}")
    model.save(keras_path, save_format='tf')
    
    return h5_path, keras_path

def check_model_compatibility(model_path):
    """
    Check if a model is compatible with both .h5 and .keras formats.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary with compatibility information
    """
    try:
        # Try to load the model
        model = load_model_any_format(model_path)
        
        # Get base path without extension
        base_path = os.path.splitext(model_path)[0]
        
        # Try to save in both formats
        temp_h5_path = f"{base_path}_temp.h5"
        temp_keras_path = f"{base_path}_temp.keras"
        
        # Save in .h5 format
        h5_compatible = True
        try:
            model.save(temp_h5_path, save_format='h5')
            # Clean up
            if os.path.exists(temp_h5_path):
                os.remove(temp_h5_path)
        except Exception as e:
            logger.warning(f"Model is not compatible with .h5 format: {str(e)}")
            h5_compatible = False
        
        # Save in .keras format
        keras_compatible = True
        try:
            model.save(temp_keras_path, save_format='tf')
            # Clean up
            if os.path.exists(temp_keras_path):
                os.remove(temp_keras_path)
        except Exception as e:
            logger.warning(f"Model is not compatible with .keras format: {str(e)}")
            keras_compatible = False
        
        return {
            'model_path': model_path,
            'h5_compatible': h5_compatible,
            'keras_compatible': keras_compatible,
            'fully_compatible': h5_compatible and keras_compatible
        }
    except Exception as e:
        logger.error(f"Error checking model compatibility: {str(e)}")
        return {
            'model_path': model_path,
            'h5_compatible': False,
            'keras_compatible': False,
            'fully_compatible': False,
            'error': str(e)
        }

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python model_format_utils.py <model_path> [output_format]")
        print("Example: python model_format_utils.py model.h5 keras")
        print("Example: python model_format_utils.py model.keras h5")
        print("If output_format is not specified, the model will be checked for compatibility with both formats.")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_format = sys.argv[2].lower()
        if output_format not in ['h5', 'keras']:
            print("Error: output_format must be 'h5' or 'keras'")
            sys.exit(1)
        
        # Get base path without extension
        base_path = os.path.splitext(model_path)[0]
        
        # Convert to the specified format
        if output_format == 'h5':
            output_path = f"{base_path}.h5"
        else:
            output_path = f"{base_path}.keras"
        
        # Convert the model
        try:
            convert_model_format(model_path, output_path)
            print(f"Model successfully converted to {output_path}")
        except Exception as e:
            print(f"Error converting model: {str(e)}")
            sys.exit(1)
    else:
        # Check compatibility
        result = check_model_compatibility(model_path)
        
        print("\nModel Compatibility Check:")
        print(f"Model: {result['model_path']}")
        print(f"Compatible with .h5 format: {result['h5_compatible']}")
        print(f"Compatible with .keras format: {result['keras_compatible']}")
        print(f"Fully compatible with both formats: {result['fully_compatible']}")
        
        if 'error' in result:
            print(f"Error: {result['error']}")