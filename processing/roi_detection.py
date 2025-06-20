"""
Advanced Region of Interest (ROI) detection for histopathology images.

This module provides robust algorithms for detecting regions of interest
in histopathology images, with support for different stain types and
tissue characteristics.
"""

import numpy as np
import cv2
from skimage import filters, morphology, color, exposure

def normalize_image(image):
    """
    Normalize image to 0-1 range.

    Args:
        image: Input image

    Returns:
        Normalized image
    """
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32)

    if image.max() > 1.0:
        image = image / 255.0

    return image

def detect_tissue_hsv(image, saturation_threshold=0.05, value_threshold=0.8):
    """
    Detect tissue regions using HSV color space thresholding.

    This method works well for H&E stained images by focusing on
    the saturation channel to distinguish tissue from background.

    Args:
        image: RGB input image
        saturation_threshold: Minimum saturation to be considered tissue
        value_threshold: Maximum value (brightness) to be considered tissue

    Returns:
        Binary mask of detected tissue regions
    """
    # Normalize image
    image = normalize_image(image)

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Extract channels
    h, s, v = cv2.split(hsv)

    # Create mask based on saturation and value
    mask = ((s > saturation_threshold) & (v < value_threshold)).astype(np.uint8)

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

def detect_tissue_adaptive(image, block_size=51, c=0.02):
    """
    Detect tissue regions using adaptive thresholding.

    This method is more robust to variations in staining intensity
    and works well for various stain types.

    Args:
        image: Input image
        block_size: Size of the local neighborhood for adaptive thresholding
        c: Constant subtracted from the mean

    Returns:
        Binary mask of detected tissue regions
    """
    # Convert to grayscale if RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Normalize
    gray = normalize_image(gray)

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        (gray * 255).astype(np.uint8),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c
    )

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask / 255.0

def detect_tissue_otsu(image):
    """
    Detect tissue regions using Otsu's thresholding.

    This method works well for images with good contrast between
    tissue and background.

    Args:
        image: Input image

    Returns:
        Binary mask of detected tissue regions
    """
    # Convert to grayscale if RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Normalize
    gray = normalize_image(gray)

    # Apply Otsu's thresholding
    _, binary = cv2.threshold(
        (gray * 255).astype(np.uint8),
        0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask / 255.0

def detect_tissue_hed(image):
    """
    Detect tissue regions using Hematoxylin channel from color deconvolution.

    This method is specifically designed for H&E stained images and
    focuses on the Hematoxylin component which stains cell nuclei.

    Args:
        image: RGB input image

    Returns:
        Binary mask of detected tissue regions
    """
    # Normalize image
    image = normalize_image(image)

    # Convert RGB to Hematoxylin-Eosin-DAB
    from skimage.color import rgb2hed
    hed = rgb2hed(image)

    # Get Hematoxylin channel
    h_channel = hed[:, :, 0]

    # Rescale for visualization
    h_norm = exposure.rescale_intensity(h_channel, out_range=(0, 1))

    # Threshold the Hematoxylin channel
    thresh = filters.threshold_otsu(h_norm)
    binary = (h_norm > thresh).astype(np.uint8)

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

def filter_regions_by_size(mask, min_size_factor=0.01, max_size_factor=0.95):
    """
    Filter regions in the mask by size.

    Args:
        mask: Binary mask
        min_size_factor: Minimum region size as a fraction of the image size
        max_size_factor: Maximum region size as a fraction of the image size

    Returns:
        Filtered binary mask
    """
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8)
    )

    # Create output mask
    filtered_mask = np.zeros_like(mask)

    # Calculate size thresholds
    image_size = mask.shape[0] * mask.shape[1]
    min_size = min_size_factor * image_size
    max_size = max_size_factor * image_size

    # Filter regions by size
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if min_size < area < max_size:
            filtered_mask[labels == i] = 1

    return filtered_mask

def detect_roi(image, method='adaptive', **kwargs):
    """
    Detect regions of interest in the image using the specified method.

    Args:
        image: Input image
        method: ROI detection method ('hsv', 'adaptive', 'otsu', 'hed')
        **kwargs: Additional parameters for the specific method

    Returns:
        Binary mask of ROIs
    """
    # Normalize image
    image = normalize_image(image)

    # Apply the selected method
    if method == 'hsv':
        mask = detect_tissue_hsv(image, **kwargs)
    elif method == 'adaptive':
        mask = detect_tissue_adaptive(image, **kwargs)
    elif method == 'otsu':
        mask = detect_tissue_otsu(image)
    elif method == 'hed':
        mask = detect_tissue_hed(image)
    else:
        raise ValueError(f"Unsupported ROI detection method: {method}")

    # Filter regions by size
    min_size_factor = kwargs.get('min_size_factor', 0.01)
    max_size_factor = kwargs.get('max_size_factor', 0.95)
    mask = filter_regions_by_size(mask, min_size_factor, max_size_factor)

    return mask

def auto_detect_best_roi(image):
    """
    Automatically detect the best ROI detection method for the given image.

    This function tries multiple methods and selects the one that produces
    the most reasonable result based on heuristics.

    Args:
        image: Input image

    Returns:
        Binary mask of ROIs and the name of the selected method
    """
    # Try all methods
    methods = {
        'hsv': detect_tissue_hsv(image),
        'adaptive': detect_tissue_adaptive(image),
        'otsu': detect_tissue_otsu(image),
        'hed': detect_tissue_hed(image)
    }

    # Score each method based on heuristics
    scores = {}
    for name, mask in methods.items():
        # Calculate coverage (percentage of image covered by ROI)
        coverage = np.mean(mask)

        # Penalize too small or too large coverage
        if coverage < 0.05:
            scores[name] = 0  # Too small
        elif coverage > 0.9:
            scores[name] = 0  # Too large
        else:
            # Score based on how close coverage is to an ideal value (around 30%)
            scores[name] = 1.0 - abs(coverage - 0.3) / 0.3

    # Select the method with the highest score
    best_method = max(scores, key=scores.get)

    return methods[best_method], best_method

def visualize_roi(image, roi_mask, alpha=0.5, color='red'):
    """
    Visualize the ROI mask overlaid on the original image.

    Args:
        image: Original image
        roi_mask: Binary ROI mask
        alpha: Transparency of the overlay
        color: Color of the overlay ('red', 'green', 'blue', or RGB tuple)

    Returns:
        Visualization image with ROI overlay
    """
    # Normalize image
    image = normalize_image(image)

    # Ensure roi_mask is binary
    roi_mask = (roi_mask > 0).astype(np.float32)

    # Create RGB mask based on specified color
    mask_rgb = np.zeros((*roi_mask.shape, 3), dtype=np.float32)

    if color == 'red':
        mask_rgb[..., 0] = roi_mask  # Red channel
    elif color == 'green':
        mask_rgb[..., 1] = roi_mask  # Green channel
    elif color == 'blue':
        mask_rgb[..., 2] = roi_mask  # Blue channel
    elif isinstance(color, tuple) and len(color) == 3:
        # Use custom RGB color
        for i in range(3):
            mask_rgb[..., i] = roi_mask * color[i]
    else:
        # Default to red if color is not recognized
        mask_rgb[..., 0] = roi_mask  # Red channel

    # Create overlay
    overlay = image.copy()

    # Apply mask with alpha blending
    mask_3d = np.expand_dims(roi_mask, axis=-1)
    overlay = (1 - alpha * mask_3d) * overlay + alpha * mask_rgb

    # Ensure overlay is in valid range
    overlay = np.clip(overlay, 0, 1)

    return overlay
