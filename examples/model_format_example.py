"""
Example script demonstrating how to use model_format_utils.py to:
1. Convert models between .h5 and .keras formats
2. Load models from either format
3. Save models in both formats simultaneously

Usage:
    python model_format_example.py <model_path>

Example:
    python model_format_example.py model.h5
    python model_format_example.py Efficent_pet_203_clf-end.h5
"""

import os
import sys
import logging
import tensorflow as tf
from model_format_utils import (
    load_model_any_format,
    convert_model_format,
    save_model_dual_format,
    check_model_compatibility
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(model_path):
    """
    Main function to demonstrate model format conversion and dual-format saving.
    
    Args:
        model_path: Path to the model file (.h5 or .keras)
    """
    # Check if file exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    
    # Get base name without extension
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # Step 1: Check model compatibility
    logger.info("Checking model compatibility...")
    compatibility = check_model_compatibility(model_path)
    
    logger.info(f"Model: {compatibility['model_path']}")
    logger.info(f"Compatible with .h5 format: {compatibility['h5_compatible']}")
    logger.info(f"Compatible with .keras format: {compatibility['keras_compatible']}")
    logger.info(f"Fully compatible with both formats: {compatibility['fully_compatible']}")
    
    if 'error' in compatibility:
        logger.error(f"Compatibility check error: {compatibility['error']}")
        return False
    
    # Step 2: Load the model
    logger.info(f"Loading model from {model_path}...")
    try:
        model = load_model_any_format(model_path)
        logger.info("Model loaded successfully")
        
        # Print model summary
        logger.info("Model summary:")
        model.summary(print_fn=logger.info)
        
        # Print model type
        model_type = getattr(model, '_model_type', 'unknown')
        logger.info(f"Model type: {model_type}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False
    
    # Step 3: Convert to the other format
    other_format = '.keras' if model_path.endswith('.h5') else '.h5'
    other_path = f"{os.path.splitext(model_path)[0]}{other_format}"
    
    logger.info(f"Converting model to {other_format} format...")
    try:
        converted_path = convert_model_format(model_path, other_path)
        logger.info(f"Model successfully converted to {converted_path}")
    except Exception as e:
        logger.error(f"Error converting model: {str(e)}")
        # Continue with the example even if conversion fails
    
    # Step 4: Save in both formats simultaneously
    dual_format_base = f"{os.path.splitext(model_path)[0]}_dual"
    
    logger.info(f"Saving model in both formats with base path {dual_format_base}...")
    try:
        h5_path, keras_path = save_model_dual_format(model, dual_format_base)
        logger.info(f"Model saved in .h5 format at {h5_path}")
        logger.info(f"Model saved in .keras format at {keras_path}")
    except Exception as e:
        logger.error(f"Error saving model in dual formats: {str(e)}")
        return False
    
    # Step 5: Verify that both formats can be loaded
    logger.info("Verifying that both formats can be loaded...")
    
    try:
        # Load .h5 version
        h5_model = load_model_any_format(h5_path)
        logger.info(f"Successfully loaded .h5 model from {h5_path}")
        
        # Load .keras version
        keras_model = load_model_any_format(keras_path)
        logger.info(f"Successfully loaded .keras model from {keras_path}")
        
        logger.info("Both formats loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error verifying model loading: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model_format_example.py <model_path>")
        print("Example: python model_format_example.py model.h5")
        sys.exit(1)
    
    model_path = sys.argv[1]
    success = main(model_path)
    
    if success:
        print("\nSuccess! The model has been successfully converted and saved in both .h5 and .keras formats.")
        print("You can now use either format for your cell counting and heatmap generation.")
    else:
        print("\nThere were some issues during the process. Please check the logs for details.")