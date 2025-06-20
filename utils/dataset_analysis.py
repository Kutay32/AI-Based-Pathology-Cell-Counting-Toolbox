"""
Dataset Analysis Module

This module provides functionality for analyzing and integrating datasets into the
AI-Based Pathology Cell Counting Toolbox. It includes functions to read and analyze
image datasets, extract information about their structure, pixel levels, and bit depth.

Functions:
    analyze_dataset: Analyzes a dataset directory and returns statistics about the images
    read_dataset_sample: Reads a sample of images from a dataset directory
    get_image_info: Extracts information about an image file
    integrate_dataset: Integrates a dataset into the project
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

def get_image_info(image_path):
    """
    Extract information about an image file.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Dictionary containing image information
    """
    try:
        # Read the image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            return {
                'path': image_path,
                'error': 'Failed to read image'
            }
        
        # Convert BGR to RGB if the image is color
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get basic image information
        height, width = image.shape[:2]
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        dtype = image.dtype
        bit_depth = 8 * image.itemsize
        
        # Calculate pixel statistics
        min_val = np.min(image)
        max_val = np.max(image)
        mean_val = np.mean(image)
        std_val = np.std(image)
        
        # Calculate histogram for each channel
        histograms = []
        if channels == 1:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            histograms.append(hist.flatten())
        else:
            for i in range(channels):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                histograms.append(hist.flatten())
        
        return {
            'path': image_path,
            'height': height,
            'width': width,
            'channels': channels,
            'dtype': str(dtype),
            'bit_depth': bit_depth,
            'min_value': min_val,
            'max_value': max_val,
            'mean_value': mean_val,
            'std_value': std_val,
            'histograms': histograms
        }
    except Exception as e:
        return {
            'path': image_path,
            'error': str(e)
        }

def read_dataset_sample(dataset_dir, max_files=100):
    """
    Read a sample of images from a dataset directory.
    
    Args:
        dataset_dir (str): Path to the dataset directory
        max_files (int): Maximum number of files to read per folder
        
    Returns:
        dict: Dictionary containing image samples and their information
    """
    result = {
        'images': [],
        'labels': [],
        'image_info': [],
        'label_info': []
    }
    
    # Check if the directory exists
    if not os.path.exists(dataset_dir):
        print(f"Directory not found: {dataset_dir}")
        return result
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(dataset_dir):
        # Skip if this is not an 'images' or 'labels' directory
        if not (os.path.basename(root) == 'images' or os.path.basename(root) == 'labels'):
            continue
        
        # Get the folder type (images or labels)
        folder_type = os.path.basename(root)
        
        # Get a list of image files
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        # Limit the number of files
        if len(image_files) > max_files:
            image_files = image_files[:max_files]
        
        # Process each image file
        for file_name in image_files:
            file_path = os.path.join(root, file_name)
            
            # Get image information
            info = get_image_info(file_path)
            
            # Read the image
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if image is not None:
                # Convert BGR to RGB if the image is color
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Add to the appropriate list
                if folder_type == 'images':
                    result['images'].append(image)
                    result['image_info'].append(info)
                else:  # labels
                    result['labels'].append(image)
                    result['label_info'].append(info)
    
    return result

def analyze_dataset(dataset_dir, max_files=100):
    """
    Analyze a dataset directory and return statistics about the images.
    
    Args:
        dataset_dir (str): Path to the dataset directory
        max_files (int): Maximum number of files to analyze per folder
        
    Returns:
        dict: Dictionary containing dataset statistics
    """
    # Initialize statistics
    stats = {
        'folder_structure': [],
        'image_count': 0,
        'label_count': 0,
        'image_sizes': [],
        'image_channels': [],
        'image_bit_depths': [],
        'label_values': set(),
        'pixel_value_ranges': {
            'min': float('inf'),
            'max': float('-inf'),
            'mean': 0,
            'std': 0
        }
    }
    
    # Collect folder structure
    for root, dirs, files in os.walk(dataset_dir):
        rel_path = os.path.relpath(root, dataset_dir)
        if rel_path == '.':
            rel_path = ''
        
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        if image_files:
            stats['folder_structure'].append({
                'path': rel_path,
                'file_count': len(image_files),
                'sample_files': image_files[:5]
            })
    
    # Sample and analyze images
    sample_data = read_dataset_sample(dataset_dir, max_files)
    
    # Update statistics based on sample data
    stats['image_count'] = len(sample_data['images'])
    stats['label_count'] = len(sample_data['labels'])
    
    # Process image information
    pixel_values = []
    
    for info in sample_data['image_info']:
        if 'error' not in info:
            stats['image_sizes'].append((info['height'], info['width']))
            stats['image_channels'].append(info['channels'])
            stats['image_bit_depths'].append(info['bit_depth'])
            
            # Update pixel value ranges
            stats['pixel_value_ranges']['min'] = min(stats['pixel_value_ranges']['min'], info['min_value'])
            stats['pixel_value_ranges']['max'] = max(stats['pixel_value_ranges']['max'], info['max_value'])
            
            # Collect pixel values for mean and std calculation
            pixel_values.append(info['mean_value'])
    
    # Process label information
    for info in sample_data['label_info']:
        if 'error' not in info:
            # For labels, we're interested in the unique values (class IDs)
            if 'min_value' in info and 'max_value' in info:
                stats['label_values'].add(info['min_value'])
                stats['label_values'].add(info['max_value'])
    
    # Calculate mean and std of pixel values
    if pixel_values:
        stats['pixel_value_ranges']['mean'] = np.mean(pixel_values)
        stats['pixel_value_ranges']['std'] = np.std(pixel_values)
    
    # Convert label_values set to a sorted list for better readability
    stats['label_values'] = sorted(list(stats['label_values']))
    
    return stats

def integrate_dataset(dataset_dir, max_files=100):
    """
    Integrate a dataset into the project.
    
    Args:
        dataset_dir (str): Path to the dataset directory
        max_files (int): Maximum number of files to integrate per folder
        
    Returns:
        dict: Dictionary containing integration results
    """
    # Analyze the dataset
    stats = analyze_dataset(dataset_dir, max_files)
    
    # Sample data from the dataset
    sample_data = read_dataset_sample(dataset_dir, max_files)
    
    # Create a summary report
    summary = {
        'dataset_path': dataset_dir,
        'statistics': stats,
        'sample_count': {
            'images': len(sample_data['images']),
            'labels': len(sample_data['labels'])
        }
    }
    
    return summary

def visualize_dataset_sample(dataset_dir, max_files=10, save_path=None):
    """
    Visualize a sample of images from the dataset.
    
    Args:
        dataset_dir (str): Path to the dataset directory
        max_files (int): Maximum number of files to visualize
        save_path (str, optional): Path to save the visualization
        
    Returns:
        None
    """
    # Read a sample of images
    sample_data = read_dataset_sample(dataset_dir, max_files)
    
    if not sample_data['images']:
        print("No images found in the dataset.")
        return
    
    # Determine the number of rows and columns for the grid
    n_images = min(len(sample_data['images']), max_files)
    n_cols = min(5, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Create a figure
    plt.figure(figsize=(15, 3 * n_rows))
    
    # Plot each image
    for i in range(n_images):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(sample_data['images'][i])
        plt.title(f"Image {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def generate_dataset_report(dataset_dir, max_files=100, save_path=None):
    """
    Generate a comprehensive report about the dataset.
    
    Args:
        dataset_dir (str): Path to the dataset directory
        max_files (int): Maximum number of files to analyze per folder
        save_path (str, optional): Path to save the report
        
    Returns:
        str: HTML report
    """
    # Analyze the dataset
    stats = analyze_dataset(dataset_dir, max_files)
    
    # Create a DataFrame for folder structure
    folder_df = pd.DataFrame(stats['folder_structure'])
    
    # Create a DataFrame for image statistics
    image_stats = {
        'Total Images': stats['image_count'],
        'Total Labels': stats['label_count'],
        'Unique Image Sizes': len(set(stats['image_sizes'])),
        'Image Channels': list(set(stats['image_channels'])),
        'Image Bit Depths': list(set(stats['image_bit_depths'])),
        'Min Pixel Value': stats['pixel_value_ranges']['min'],
        'Max Pixel Value': stats['pixel_value_ranges']['max'],
        'Mean Pixel Value': stats['pixel_value_ranges']['mean'],
        'Std Pixel Value': stats['pixel_value_ranges']['std'],
        'Unique Label Values': stats['label_values']
    }
    
    # Convert to HTML
    html = "<h1>Dataset Analysis Report</h1>"
    html += f"<h2>Dataset: {dataset_dir}</h2>"
    
    # Image statistics
    html += "<h3>Image Statistics</h3>"
    html += "<table border='1'>"
    for key, value in image_stats.items():
        html += f"<tr><td>{key}</td><td>{value}</td></tr>"
    html += "</table>"
    
    # Folder structure
    html += "<h3>Folder Structure</h3>"
    html += folder_df.to_html(index=False)
    
    # Save the report if a save path is provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write(html)
        print(f"Report saved to {save_path}")
    
    return html