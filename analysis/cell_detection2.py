"""
Cell detection and analysis functions for pathology images using OpenCV.
"""

import numpy as np
import cv2
import pandas as pd
import tensorflow as tf

def extract_cell_features(image, labeled_mask, class_id):
    """
    Extract morphological features and bounding boxes for each cell.
    Returns features with bbox coordinates.

    Args:
        image: Input image
        labeled_mask: Labeled mask with unique IDs for each cell
        class_id: Class ID for the cells

    Returns:
        List of dictionaries containing cell features
    """
    features = []

    # Find unique labels in the mask (excluding 0 which is background)
    unique_labels = np.unique(labeled_mask)
    unique_labels = unique_labels[unique_labels > 0]

    # For each cell (label)
    for label in unique_labels:
        # Create a binary mask for this cell
        cell_mask = (labeled_mask == label).astype(np.uint8)

        # Find contours for this cell
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        # Get the largest contour (should be only one for a single cell)
        contour = max(contours, key=cv2.contourArea)

        # Calculate area
        area = cv2.contourArea(contour)

        if area < 3:  # Filter out very small regions
            continue

        # Calculate bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate perimeter
        perimeter = cv2.arcLength(contour, True)

        # Calculate moments for eccentricity
        moments = cv2.moments(contour)

        # Calculate eccentricity using central moments
        if moments['m00'] != 0:
            # Calculate central moments
            mu20 = moments['mu20'] / moments['m00']
            mu02 = moments['mu02'] / moments['m00']
            mu11 = moments['mu11'] / moments['m00']

            # Calculate eccentricity
            diff = mu20 - mu02
            eccentricity = np.sqrt(diff*diff + 4*mu11*mu11) / (mu20 + mu02)
        else:
            eccentricity = 0

        # Calculate circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)

        # Calculate mean intensity
        mean_intensity = cv2.mean(image, mask=cell_mask)[0]

        features.append({
            'Class': class_id,
            'Area': area,
            'BBox': (x, y, w, h),  # Add bbox to features
            'Perimeter': perimeter,
            'Eccentricity': eccentricity,
            'Circularity': circularity,
            'Mean Intensity': mean_intensity
        })

    return features

def split_touching_cells(pred_mask, class_id):
    """
    Split touching cells using watershed algorithm with OpenCV.

    Args:
        pred_mask: Predicted segmentation mask
        class_id: Class ID to process

    Returns:
        Labeled mask with separated cells
    """
    # Create binary mask for the class
    binary = (pred_mask == class_id).astype(np.uint8)

    # Check if there are any pixels of this class
    if np.sum(binary) == 0:
        print(f"Warning: No pixels found for class {class_id} in the mask")
        # Return a blank mask of the same shape
        return np.zeros_like(binary)

    # Apply morphological operations to clean the mask
    # FIXED: Reduced kernel size to be less aggressive for small cells
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Ensure the opening operation didn't remove all pixels
    if np.sum(opening) == 0:
        print(f"Warning: Opening operation removed all pixels for class {class_id}")
        # Fall back to the original binary mask
        opening = binary

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=2)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # Ensure the distance transform has non-zero values
    if dist_transform.max() == 0:
        print(f"Warning: Distance transform is all zeros for class {class_id}")
        # Return connected components of the binary mask directly
        ret, markers = cv2.connectedComponents(binary)
        return markers

    # Threshold to find sure foreground
    # Use a lower threshold to ensure we get some foreground
    # FIXED: Better threshold handling for small cells
    if dist_transform.max() > 0:
        # Use relative threshold but with minimum floor value
        threshold_value = max(0.1 * dist_transform.max(), 0.5)  # Reduced multiplier and increased minimum
    else:
        threshold_value = 0
    ret, sure_fg = cv2.threshold(dist_transform, threshold_value, 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # Ensure the thresholding operation didn't remove all pixels
    if np.sum(sure_fg) == 0:
        print(f"Warning: Thresholding removed all pixels for class {class_id}")
        # Fall back to a lower threshold
        ret, sure_fg = cv2.threshold(dist_transform, 0.1, 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        # If still no foreground, use the opening
        if np.sum(sure_fg) == 0:
            sure_fg = opening

    # Finding unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that background is 1 instead of 0
    markers = markers + 1

    # Mark the unknown region with 0
    markers[unknown == 1] = 0

    # Apply watershed
    # Convert binary to BGR for watershed
    binary_bgr = cv2.cvtColor(binary*255, cv2.COLOR_GRAY2BGR)
    cv2.watershed(binary_bgr, markers)

    # Watershed marks boundaries with -1, convert to positive labels
    markers[markers == -1] = 0

    return markers

def count_cells(pred_mask, class_names=None, classification_result=None):
    """
    Count cells for each class in the predicted mask using OpenCV.
    If classification_result is provided, use it to define the cell type.

    Args:
        pred_mask: Predicted segmentation mask
        class_names: List of class names
        classification_result: Optional dictionary with classification results

    Returns:
        Dictionary with cell counts for each class
    """
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(pred_mask.max() + 1)]

    counts = {}

    # Initialize counts for all classes to 0
    for class_id in range(1, len(class_names)):
        counts[class_names[class_id]] = 0

    # Always count cells for each class separately, regardless of classification result
    for class_id in range(1, len(class_names)):  # Skip background (0)
        instance_mask = split_touching_cells(pred_mask, class_id)

        # FIXED: Proper cell counting using unique labels
        unique = np.unique(instance_mask)
        num_cells = len(unique) - 1 if 0 in unique else len(unique)
        counts[class_names[class_id]] = num_cells

    # If classification result is provided, we still keep the counts for all classes
    # but we can highlight the predicted class in the UI if needed
    if (classification_result and 'predicted_class' in classification_result and
            'probability' in classification_result):
        # Store the predicted class in the classification result for reference
        classification_result['all_classes_counted'] = True

        # Note: We're not adjusting counts based on whole-image classification here
        # Individual cell classification is handled in summarize_prediction

    return counts

def analyze_cell_morphology(image, pred_mask, class_names=None):
    """
    Analyze cell morphology for all classes in the predicted mask using OpenCV.

    Args:
        image: Input image
        pred_mask: Predicted segmentation mask
        class_names: List of class names

    Returns:
        DataFrame with cell features
    """
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(pred_mask.max() + 1)]

    all_features = []
    for class_id in range(1, len(class_names)):  # Skip background (0)
        instance_mask = split_touching_cells(pred_mask, class_id)
        features = extract_cell_features(image, instance_mask, class_id)
        for f in features:
            if class_names:
                f['ClassName'] = class_names[class_id]
        all_features.extend(features)

    return pd.DataFrame(all_features)

def calculate_cell_density(pred_mask, image_area, class_names=None, classification_result=None):
    """
    Calculate cell density (cells per unit area) for each class using OpenCV-based cell counting.
    If classification_result is provided, use it to define the cell type.

    Args:
        pred_mask: Predicted segmentation mask
        image_area: Total area of the image
        class_names: List of class names
        classification_result: Optional dictionary with classification results

    Returns:
        Dictionary with cell density for each class
    """
    counts = count_cells(pred_mask, class_names, classification_result)
    densities = {class_name: count / image_area for class_name, count in counts.items()}
    return densities

def classify_cells(image, cell_features, classification_model, class_names=None):
    """
    Classify individual cells using a classification model.

    This function takes a list of cell features (with bounding boxes) and an image,
    and uses a classification model to predict the cell type for each cell. It extracts
    each cell using its bounding box, preprocesses it for the model, and runs the
    classification prediction. The results are added to the cell features as
    'PredictedClass', 'PredictedClassName', and 'ClassProbability'.

    Args:
        image: Input image
        cell_features: List of dictionaries containing cell features with bounding boxes
        classification_model: Classification model to use
        class_names: List of class names

    Returns:
        Updated cell features with predicted cell types
    """
    if not cell_features:
        return cell_features

    if class_names is None:
        from utils.helpers import get_default_class_names
        class_names = get_default_class_names()

    # Create a copy of the features to avoid modifying the original
    updated_features = []

    for feature in cell_features:
        # Extract bounding box
        x, y, w, h = feature['BBox']

        # Ensure coordinates are within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)

        # Skip if bounding box is invalid
        if w <= 0 or h <= 0:
            updated_features.append(feature.copy())
            continue

        # Extract cell image
        cell_image = image[y:y+h, x:x+w]

        # Skip if cell image is empty
        if cell_image.size == 0:
            updated_features.append(feature.copy())
            continue

        # Preprocess cell image for classification
        # Resize to match model input shape
        input_shape = classification_model.input_shape[1:3]
        resized_cell = cv2.resize(cell_image, input_shape)

        # Normalize pixel values
        normalized_cell = resized_cell / 255.0

        # Add batch dimension
        input_cell = np.expand_dims(normalized_cell, axis=0)

        # Run classification
        try:
            prediction = classification_model.predict(input_cell, verbose=0)
            predicted_class = np.argmax(prediction[0])
            probability = prediction[0][predicted_class]

            # Create a copy of the feature and update it
            updated_feature = feature.copy()
            updated_feature['PredictedClass'] = predicted_class
            updated_feature['PredictedClassName'] = class_names[predicted_class]
            updated_feature['ClassProbability'] = probability
            updated_features.append(updated_feature)
        except Exception as e:
            print(f"Error classifying cell: {str(e)}")
            updated_features.append(feature.copy())

    return updated_features

def summarize_prediction(image, pred_mask, index=0, class_names=None, classification_result=None, classification_model=None):
    """
    Generate a comprehensive summary of the prediction using OpenCV-based cell detection.
    Always analyzes cells for each class separately, regardless of classification result.

    If classification_model is provided, this function will also classify individual cells
    by their cell type using the model. It extracts each detected cell, classifies it using
    the model, and updates the cell features with the classification results. The class
    summary is also updated with counts of cells classified as each type.

    This function provides a complete pipeline for cell detection and classification:
    1. Segments cells using the segmentation mask
    2. Extracts features for each cell
    3. Classifies each cell by type (if classification_model is provided)
    4. Generates a summary of cell counts and classifications

    Args:
        image: Input image
        pred_mask: Predicted segmentation mask
        index: Image index (for batch processing)
        class_names: List of class names
        classification_result: Optional dictionary with classification results from whole-image classification
        classification_model: Optional classification model for individual cell type classification

    Returns:
        Tuple of (cell_features, class_summary) where:
        - cell_features: List of dictionaries containing features for each cell, including classification results
        - class_summary: List of dictionaries containing summary statistics for each class
    """
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(pred_mask.max() + 1)]

    all_features = []
    class_summary = []
    image_area = image.shape[0] * image.shape[1]

    # Get initial cell counts using the count_cells function (which now always counts cells for each class separately)
    counts = count_cells(pred_mask, class_names, classification_result)

    # Always analyze cells for each class separately, regardless of classification result
    for class_id in range(1, len(class_names)):  # Skip background (0)
        instance_mask = split_touching_cells(pred_mask, class_id)

        # FIXED: Proper cell counting using unique labels
        unique = np.unique(instance_mask)
        num_cells = len(unique) - 1 if 0 in unique else len(unique)

        # Add debug visualization for cases where we have mask pixels but no cells detected
        if num_cells == 0 and np.any(pred_mask == class_id):
            print(f"Warning: Class {class_id} ({class_names[class_id]}) has mask pixels but 0 cells detected")

        if num_cells > 0:
            features = extract_cell_features(image, instance_mask, class_id)
            for f in features:
                f['Image'] = index
                f['ClassName'] = class_names[class_id]
                # Store original class ID for reference
                f['OriginalClass'] = class_id
            all_features.extend(features)

            # Calculate density
            density = num_cells / image_area

            class_summary.append({
                'Class': class_id,
                'ClassName': class_names[class_id],
                'Cell Count': num_cells,
                'Average Area': np.mean([f['Area'] for f in features]) if features else 0,
                'Density (cells per pixel^2)': density
            })
        else:
            class_summary.append({
                'Class': class_id,
                'ClassName': class_names[class_id],
                'Cell Count': 0,
                'Average Area': 0.0,
                'Density (cells per pixel^2)': 0.0
            })

    # If classification result is provided, we can highlight the predicted class in the UI if needed
    if (classification_result and 'predicted_class' in classification_result and
            'probability' in classification_result):
        # Store the predicted class in the classification result for reference
        classification_result['all_classes_analyzed'] = True

    # If classification model is provided, classify individual cells
    if classification_model is not None and all_features:
        all_features = classify_cells(image, all_features, classification_model, class_names)

        # Update class summary with classification results
        class_counts = {}
        for feature in all_features:
            if 'PredictedClassName' in feature:
                class_name = feature['PredictedClassName']
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1

                # Update the cell's class to match its predicted class
                feature['class'] = feature['PredictedClass']
                feature['class_name'] = feature['PredictedClassName']

        # Reset counts based on classification results
        for summary in class_summary:
            class_name = summary['ClassName']
            # Update the main cell count to match the classified count
            summary['Original Cell Count'] = summary['Cell Count']  # Store original count for reference
            summary['Cell Count'] = class_counts.get(class_name, 0)
            summary['Classified Count'] = class_counts.get(class_name, 0)

    return all_features, class_summary
