"""
Image preprocessing and enhancement functions for pathology images.
"""

import numpy as np
import cv2
from skimage import exposure, color, filters
import matplotlib.pyplot as plt

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

def adaptive_histogram_equalization(image, clip_limit=0.03):
    """
    Apply adaptive histogram equalization to enhance image contrast.

    Args:
        image: Input image
        clip_limit: Clipping limit for contrast enhancement

    Returns:
        Enhanced image
    """
    # Convert to grayscale if RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Normalize to 0-1
    gray = normalize_image(gray)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply((gray * 255).astype(np.uint8))

    # Convert back to original range
    enhanced = enhanced.astype(np.float32) / 255.0

    # If original was RGB, merge the enhanced channel
    if len(image.shape) == 3 and image.shape[2] == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[..., 2] = enhanced * 255  # Value channel
        enhanced_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return enhanced_rgb

    return enhanced

def gaussian_filtering(image, sigma=1.0):
    """
    Apply Gaussian filtering to reduce noise.

    Args:
        image: Input image
        sigma: Standard deviation for Gaussian kernel

    Returns:
        Filtered image
    """
    return filters.gaussian(image, sigma=sigma, preserve_range=True)

def color_deconvolution(image, stain_type='H&E'):
    """
    Perform color deconvolution to separate stain components.

    Args:
        image: Input RGB image
        stain_type: Type of stain ('H&E' or 'DAB')

    Returns:
        Dictionary of separated stain components
    """
    from skimage.color import rgb2hed

    # Normalize image
    image = normalize_image(image)

    if stain_type == 'H&E':
        # Convert RGB to Hematoxylin-Eosin-DAB
        hed = rgb2hed(image)

        # Get individual stain channels
        h_channel = hed[:, :, 0]
        e_channel = hed[:, :, 1]
        d_channel = hed[:, :, 2]

        # Rescale for visualization
        h_norm = exposure.rescale_intensity(h_channel, out_range=(0, 1))
        e_norm = exposure.rescale_intensity(e_channel, out_range=(0, 1))
        d_norm = exposure.rescale_intensity(d_channel, out_range=(0, 1))

        return {
            'Hematoxylin': h_norm,
            'Eosin': e_norm,
            'DAB': d_norm
        }

    elif stain_type == 'DAB':
        # For DAB staining, we focus on the brown component
        # This is a simplified approach
        b_channel = image[:, :, 0]
        g_channel = image[:, :, 1]
        r_channel = image[:, :, 2]

        # DAB is primarily in the blue-green channels
        dab_channel = (b_channel + g_channel) / 2 - r_channel
        dab_norm = exposure.rescale_intensity(dab_channel, out_range=(0, 1))

        return {
            'DAB': dab_norm
        }

    else:
        raise ValueError(f"Unsupported stain type: {stain_type}")

def detect_roi(image, threshold=0.1):
    """
    Automatically detect regions of interest in the image.

    Args:
        image: Input image
        threshold: Threshold for ROI detection

    Returns:
        Binary mask of ROIs
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
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Find contours
    contours, _ = cv2.findContours(
        binary, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Create ROI mask
    roi_mask = np.zeros_like(gray)

    # Filter contours by area
    min_area = threshold * image.shape[0] * image.shape[1]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            cv2.drawContours(roi_mask, [contour], 0, 1, -1)

    return roi_mask

def visualize_preprocessing(original, processed, title="Image Preprocessing"):
    """
    Visualize original and processed images side by side.

    Args:
        original: Original image
        processed: Processed image
        title: Plot title
    """
    plt.figure(figsize=(12, 5))

    # Normalize original image to 0-1 range for display
    normalized_original = original.copy()
    if normalized_original.max() > 1.0:
        normalized_original = normalized_original / 255.0

    # Normalize processed image to 0-1 range for display
    normalized_processed = processed.copy()
    if normalized_processed.max() > 1.0:
        normalized_processed = normalized_processed / 255.0

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(normalized_original)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Processed")
    plt.imshow(normalized_processed)
    plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def preprocess_image(image, enhance=True, denoise=True, detect_regions=False):
    """
    Apply a complete preprocessing pipeline to the image.

    Args:
        image: Input image
        enhance: Whether to apply contrast enhancement
        denoise: Whether to apply denoising
        detect_regions: Whether to detect ROIs

    Returns:
        Preprocessed image and optional ROI mask
    """
    # Normalize
    image = normalize_image(image)

    # Denoise
    if denoise:
        image = gaussian_filtering(image, sigma=1.0)

    # Enhance contrast
    if enhance:
        image = adaptive_histogram_equalization(image)

    # Detect ROIs
    roi_mask = None
    if detect_regions:
        roi_mask = detect_roi(image)

    return image, roi_mask
