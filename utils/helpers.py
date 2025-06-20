"""
Helper functions and utilities for the AI-Based Pathology Cell Counting Toolbox.
"""

import os
import numpy as np
import cv2
import json
import pandas as pd
from datetime import datetime

def load_image(file_path):
    """
    Load an image from file.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Loaded image in RGB format
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    
    # Load image
    image = cv2.imread(file_path)
    
    # Convert from BGR to RGB
    if image is not None and len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def save_image(image, file_path):
    """
    Save an image to file.
    
    Args:
        image: Image to save
        file_path: Path to save the image
    """
    # Convert from RGB to BGR if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save image
    cv2.imwrite(file_path, image)

def save_results_to_csv(df, file_path):
    """
    Save results DataFrame to CSV.
    
    Args:
        df: DataFrame with results
        file_path: Path to save the CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(file_path, index=False)
    
    print(f"Results saved to {file_path}")

def save_results_to_json(results, file_path):
    """
    Save results dictionary to JSON.
    
    Args:
        results: Dictionary with results
        file_path: Path to save the JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Convert numpy arrays to lists
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        else:
            return obj
    
    # Convert results
    results_converted = convert_numpy(results)
    
    # Save to JSON
    with open(file_path, 'w') as f:
        json.dump(results_converted, f, indent=4)
    
    print(f"Results saved to {file_path}")

def generate_report_filename(base_dir="results", prefix="report", extension=".csv"):
    """
    Generate a unique filename for a report based on current date and time.
    
    Args:
        base_dir: Base directory for reports
        prefix: Prefix for the filename
        extension: File extension
        
    Returns:
        Full path for the report file
    """
    # Create directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    filename = f"{prefix}_{timestamp}{extension}"
    
    return os.path.join(base_dir, filename)

def batch_process_images(image_dir, process_func, output_dir=None, extensions=None):
    """
    Process all images in a directory using the provided function.
    
    Args:
        image_dir: Directory containing images
        process_func: Function to process each image
        output_dir: Directory to save results (if None, use image_dir)
        extensions: List of file extensions to process (if None, process all)
        
    Returns:
        List of results from processing each image
    """
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    if output_dir is None:
        output_dir = image_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Default extensions
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    
    # Get all image files
    image_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(root, file))
    
    # Process each image
    results = []
    for image_file in image_files:
        try:
            # Load image
            image = load_image(image_file)
            
            # Process image
            result = process_func(image)
            
            # Add filename to result
            if isinstance(result, dict):
                result['filename'] = os.path.basename(image_file)
            
            # Add to results
            results.append(result)
            
            print(f"Processed {image_file}")
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
    
    return results

def get_default_class_names():
    """
    Get default class names for cell types.
    
    Returns:
        List of default class names
    """
    return ["Background", "Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]