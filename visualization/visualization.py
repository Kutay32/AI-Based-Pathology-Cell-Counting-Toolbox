"""
Visualization functions for pathology image analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import cv2
from matplotlib.figure import Figure

def visualize_prediction(image, true_mask=None, pred_mask=None, class_names=None, show_dice=True, cmap='tab10', threshold=None, classification_result=None):
    """
    Visualize original image, true mask, and predicted mask.

    Args:
        image: Input image
        true_mask: Ground truth mask (optional)
        pred_mask: Predicted mask (optional)
        class_names: List of class names
        show_dice: Whether to show Dice scores
        cmap: Colormap for masks
        threshold: Threshold value for binary mask visualization (optional)
        classification_result: Optional dictionary with classification results
    """
    n_extra = (true_mask is not None) + (pred_mask is not None)
    show_classification = classification_result is not None and 'predicted_class' in classification_result and 'probability' in classification_result
    n_plots = 1 + n_extra + int(show_classification)
    idx = 1

    plt.figure(figsize=(5 * n_plots, 5))

    # 1. Original
    plt.subplot(1, n_plots, idx)
    plt.title("Original")
    normalized_image = image.copy()
    if normalized_image.max() > 1.0:
        normalized_image = normalized_image / 255.0
    if image.shape[-1] == 1:
        plt.imshow(normalized_image.squeeze(), cmap='gray')
    else:
        plt.imshow(normalized_image)
    plt.axis("off")
    idx += 1

    # 2. Classification (if present)
    if show_classification:
        plt.subplot(1, n_plots, idx)
        predicted_class = classification_result['predicted_class']
        probability = classification_result['probability']
        if class_names and predicted_class < len(class_names):
            class_name = class_names[predicted_class]
        else:
            class_name = f"Class {predicted_class}"
        plt.imshow(normalized_image if image.shape[-1] != 1 else normalized_image.squeeze(), cmap='gray' if image.shape[-1] == 1 else None)
        plt.title(f"Classification: {class_name}\nConfidence: {probability:.2f}", fontsize=10)
        plt.axis("off")
        idx += 1

    # 3. True mask
    if true_mask is not None:
        plt.subplot(1, n_plots, idx)
        plt.title("True Mask")
        plt.imshow(true_mask, cmap=cmap, vmin=0, vmax=(len(class_names)-1 if class_names else None))
        plt.axis("off")
        idx += 1

    # 4. Predicted mask
    if pred_mask is not None:
        plt.subplot(1, n_plots, idx)
        plt.title("Predicted Mask")
        if threshold is not None:
            binary_mask = (pred_mask > threshold).astype(np.uint8)
            plt.imshow(binary_mask, cmap='binary', vmin=0, vmax=1)
        else:
            plt.imshow(pred_mask, cmap=cmap, vmin=0, vmax=(len(class_names)-1 if class_names else None))
        plt.axis("off")

    # Show Dice if present
    if (true_mask is not None and pred_mask is not None and show_dice):
        from models.losses import compute_dice
        dice_scores = compute_dice(true_mask, pred_mask, num_classes=len(class_names))
        text = "\n".join([f"{class_names[i]}: {dice_scores[i]:.2f}" for i in range(len(class_names))])
        plt.gcf().text(0.92, 0.5, text, fontsize=10, va='center')

    plt.tight_layout()
    plt.show()
def plot_class_based_analysis(df, class_names=None):
    """
    Plot class-based analysis of cell features.

    Args:
        df: DataFrame with cell features
        class_names: List of class names
    """
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    sns.histplot(df['Area'], kde=True, bins=30, color='skyblue')
    plt.title('Cell Area Distribution')

    plt.subplot(2, 2, 2)
    sns.histplot(df['Perimeter'], kde=True, bins=30, color='orange')
    plt.title('Cell Perimeter Distribution')

    plt.subplot(2, 2, 3)
    sns.histplot(df['Circularity'], kde=True, bins=30, color='green')
    plt.title('Cell Circularity Distribution')

    plt.subplot(2, 2, 4)
    sns.histplot(df['Eccentricity'], kde=True, bins=30, color='red')
    plt.title('Cell Eccentricity Distribution')

    plt.tight_layout()
    plt.show()

    # Box plots by class
    plt.figure(figsize=(14, 6))
    sns.boxplot(x='ClassName', y='Area', data=df, palette='Set2')
    plt.title('Cell Area by Class')
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(14, 6))
    sns.boxplot(x='ClassName', y='Perimeter', data=df, palette='Set2')
    plt.title('Cell Perimeter by Class')
    plt.xticks(rotation=45)
    plt.show()

def visualize_cell_detection(image, labeled_mask, class_id=None, class_names=None, preprocessed_image=None, show=False):
    """
    Visualize detected cells with unique colors, cell boundaries, and bounding boxes.

    Args:
        image: Input image
        labeled_mask: Labeled mask with unique IDs for each cell
        class_id: Class ID for the cells
        class_names: List of class names
        preprocessed_image: Optional preprocessed image to use instead of original
        show: Whether to show the figure (default: False)

    Returns:
        Figure object with the visualization
    """
    fig = plt.figure(figsize=(15, 6))

    # Original image
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    # Normalize image to 0-1 range for display
    normalized_image = image.copy()
    if normalized_image.max() > 1.0:
        normalized_image = normalized_image / 255.0

    if image.shape[-1] == 1:
        plt.imshow(normalized_image.squeeze(), cmap='gray')
    else:
        plt.imshow(normalized_image)
    plt.axis("off")

    # Get class name for display
    class_name = "Unknown"
    if class_id is not None and class_names is not None and class_id < len(class_names):
        class_name = class_names[class_id]

    # Middle subplot: Cell boundaries
    plt.subplot(1, 3, 2)
    plt.title(f"Cell Boundaries - {class_name}")

    # Find boundaries of labeled cells
    boundaries = (labeled_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(boundaries, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a boundary overlay
    boundary_overlay = normalized_image.copy()
    if boundary_overlay.shape[-1] == 1:
        boundary_overlay = cv2.cvtColor((boundary_overlay * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        boundary_overlay = boundary_overlay / 255.0

    # Draw boundaries in red
    cv2.drawContours(boundary_overlay, contours, -1, (1.0, 0, 0), 1)

    plt.imshow(boundary_overlay)
    plt.axis("off")

    # Right subplot: Bounding boxes on preprocessed image
    plt.subplot(1, 3, 3)
    plt.title(f"Cell BBox - {class_name}")

    # Use preprocessed image if provided, otherwise use original
    display_img = preprocessed_image.copy() if preprocessed_image is not None else image.copy()
    if display_img.max() <= 1.0:
        display_img = (display_img * 255).astype(np.uint8)
    if len(display_img.shape) == 2:
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

    # Draw bounding boxes
    unique_labels = np.unique(labeled_mask)
    unique_labels = unique_labels[unique_labels > 0]

    for label in unique_labels:
        cell_mask = (labeled_mask == label).astype(np.uint8)
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Add class label with name instead of just ID
            label_text = class_name if class_name != "Unknown" else f"Class {class_id}"
            cv2.putText(display_img, label_text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()

    if show:
        plt.show()

    return fig

def plot_cell_counts(counts, class_names=None):
    """
    Plot cell counts for each class.

    Args:
        counts: Dictionary with cell counts for each class
        class_names: List of class names
    """
    plt.figure(figsize=(10, 6))

    classes = list(counts.keys())
    values = list(counts.values())

    bars = plt.bar(classes, values, color='skyblue')

    plt.title('Cell Counts by Class')
    plt.xlabel('Cell Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def plot_cell_density_heatmap(image, pred_mask, class_id, window_size=50, class_names=None):
    """
    Create a heatmap of cell density with OpenCV bounding boxes.

    Args:
        image: Input image
        pred_mask: Predicted segmentation mask
        class_id: Class ID to visualize
        window_size: Size of the sliding window for density calculation
        class_names: List of class names for better labeling

    Returns:
        Figure object with the heatmap visualization
    """
    # Create binary mask for the specified class
    if np.max(pred_mask) > 1:
        # Multi-class mask
        binary_mask = (pred_mask == class_id).astype(np.uint8)
    else:
        # Binary mask - use as is
        binary_mask = pred_mask.astype(np.uint8)

    # Initialize density map
    h, w = binary_mask.shape
    density_map = np.zeros((h // window_size, w // window_size))

    # Calculate density in each window
    for i in range(0, h, window_size):
        for j in range(0, w, window_size):
            i_end = min(i + window_size, h)
            j_end = min(j + window_size, w)
            window = binary_mask[i:i_end, j:j_end]
            density_map[i // window_size, j // window_size] = np.sum(window) / (window_size * window_size)

    # Apply Gaussian blur to smooth the density map
    from scipy.ndimage import gaussian_filter
    density_map = gaussian_filter(density_map, sigma=1.0)

    # Create figure
    fig = plt.figure(figsize=(15, 5))

    # Original Image
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("Original Image")
    # Normalize image to 0-1 range for display
    normalized_image = image.copy()
    if normalized_image.max() > 1.0:
        normalized_image = normalized_image / 255.0

    if image.shape[-1] == 1:
        ax1.imshow(normalized_image.squeeze(), cmap='gray')
    else:
        ax1.imshow(normalized_image)
    ax1.axis("off")

    # Segmentation Mask with OpenCV Bounding Boxes
    ax2 = fig.add_subplot(1, 3, 2)
    # Use class name if available, otherwise use class ID
    if class_names is not None and class_id < len(class_names):
        class_label = class_names[class_id]
    else:
        class_label = f"Class {class_id}"

    ax2.set_title(f"Cell Detection ({class_label})")

    # Create a display image for OpenCV operations
    display_img = image.copy()
    if display_img.max() <= 1.0:
        display_img = (display_img * 255).astype(np.uint8)
    if len(display_img.shape) == 2:
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (noise)
    min_area = 50
    contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # Draw bounding boxes and labels on the image
    for contour in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Draw rectangle
        cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add label
        cv2.putText(display_img, f"{class_label}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the image with bounding boxes
    ax2.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    ax2.axis("off")

    # Add cell count information
    cell_count = len(contours)
    ax2.text(0.05, 0.95, f"Cell Count: {cell_count}",
             transform=ax2.transAxes, color='white', fontsize=10,
             bbox=dict(facecolor='black', alpha=0.7))

    # Density Heatmap
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title(f"Cell Density Heatmap ({class_label})")

    # Use a more visually appealing colormap
    heatmap = ax3.imshow(density_map, cmap='hot', interpolation='bicubic')

    # Add colorbar with better formatting
    cbar = fig.colorbar(heatmap, ax=ax3, label='Cell Density')
    cbar.ax.tick_params(labelsize=8)

    # Add density statistics
    mean_density = np.mean(density_map)
    max_density = np.max(density_map)
    ax3.text(0.05, 0.95, f"Mean Density: {mean_density:.4f}\nMax Density: {max_density:.4f}",
             transform=ax3.transAxes, color='white', fontsize=10,
             bbox=dict(facecolor='black', alpha=0.7))

    ax3.axis("off")

    fig.tight_layout()
    return fig


def generate_summary_report(class_summary, class_names=None):
    """
    Generate a summary report of cell counts and densities.

    Args:
        class_summary: List of dictionaries with class summaries
        class_names: List of class names

    Returns:
        DataFrame with summary report
    """
    df = pd.DataFrame(class_summary)

    if 'ClassName' not in df.columns and class_names is not None:
        df['ClassName'] = df['Class'].apply(lambda x: class_names[x])

    # Print summary
    print("=== Cell Count Summary ===")
    for _, row in df.iterrows():
        print(f"{row['ClassName']}: {row['Cell Count']} cells, "
              f"Average Area: {row['Average Area']:.2f}, "
              f"Density: {row['Density (cells per pixel^2)']:.6f}")

    # Plot summary
    plt.figure(figsize=(10, 6))
    sns.barplot(x='ClassName', y='Cell Count', hue='ClassName', data=df, palette='viridis', legend=False)
    plt.title('Cell Counts by Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return df
import matplotlib.pyplot as plt
import numpy as np

def plot_cells_on_image(app, image, cell_list, show_numbers=True):
    """
    Draw detected cells on an image with colored contours and optional numbering.
    Assumes cell_list is a list of dicts, each with 'centroid', `contour`, `class` keys.

    Args:
        app: The application instance
        image: The image to draw cells on
        cell_list: List of cells with contours and centroids
        show_numbers: Whether to show cell numbers

    Returns:
        Figure object with the cell visualization
    """
    # Check if cell_list is empty
    if not cell_list:
        # Create a figure with a message
        fig = plt.figure(figsize=(14, 14))
        plt.imshow(image)
        plt.text(image.shape[1]/2, image.shape[0]/2, "No cells detected", 
                 ha='center', va='center', fontsize=20, color='white',
                 bbox=dict(facecolor='black', alpha=0.7, pad=10))
        plt.axis('off')
        plt.tight_layout()
        return fig

    # Define default colors in case self.class_colors is not available or not properly formatted
    default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']

    fig = plt.figure(figsize=(14, 14))
    plt.imshow(image)
    for idx, cell in enumerate(cell_list):
        centroid = cell['centroid']  # (y, x)
        contour = np.array(cell['contour'])  # shape (N, 2)
        cell_class = cell.get('class', 0)

        # Get class name - first check if it's directly provided in the cell
        if 'class_name' in cell:
            class_name = cell['class_name']
        # Otherwise get it from app.class_names if available
        elif hasattr(app, 'class_names') and cell_class < len(app.class_names):
            class_name = app.class_names[cell_class]
        else:
            # Fallback to a generic class name
            class_name = f"Class {cell_class}"

        # Handle different formats of class_colors (list or dictionary)
        if hasattr(app, 'class_colors'):
            if isinstance(app.class_colors, dict):
                # If class_colors is a dictionary, try to get color by class name
                # or use a default color based on class index
                color = app.class_colors.get(class_name, default_colors[cell_class % len(default_colors)])
            else:
                # If class_colors is a list, use modulo to ensure index is within bounds
                color = app.class_colors[cell_class % len(app.class_colors)]
        else:
            # Fallback to default colors if app.class_colors is not available
            color = default_colors[cell_class % len(default_colors)]

        # Add class name label to the cell
        plt.text(contour[:, 1].mean(), contour[:, 0].min() - 5, class_name, 
                 color='white', fontsize=8, ha='center', va='center',
                 bbox=dict(facecolor=color, alpha=0.7, pad=1))

        # Draw contour
        plt.plot(contour[:, 1], contour[:, 0], color=color, linewidth=2)

        # Draw centroid
        plt.scatter([centroid[1]], [centroid[0]], color='yellow', edgecolors='black', s=40, zorder=5)

        # Draw cell number
        if show_numbers:
            plt.text(centroid[1], centroid[0], str(idx+1), color='white', fontsize=12, ha='center', va='center',
                     bbox=dict(facecolor='black', edgecolor='none', alpha=0.6, pad=1))

    plt.axis('off')
    plt.tight_layout()
    return fig

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

def plot_density_heatmap(app, image, cell_list, bins=100, sigma=6):
    """
    Plots a smooth 2D density heatmap of cell centroids overlaid on the image.

    Args:
        app: The application instance
        image: The image to overlay the heatmap on
        cell_list: List of cells with centroids
        bins: Number of bins for the histogram
        sigma: Sigma for Gaussian filter

    Returns:
        Figure object with the heatmap visualization
    """
    if not cell_list:
        fig = plt.figure(figsize=(14, 14))
        plt.imshow(image, alpha=1)
        plt.text(image.shape[1]/2, image.shape[0]/2, "No cells detected for density heatmap",
                 ha='center', va='center', fontsize=20, color='white',
                 bbox=dict(facecolor='black', alpha=0.7, pad=10))
        plt.axis('off')
        plt.tight_layout()
        return fig

    fig = plt.figure(figsize=(14, 14))
    plt.imshow(image, alpha=1, zorder=1)

    try:
        # Print debug info about cell_list
        print(f"DEBUG: cell_list length: {len(cell_list)}")
        if cell_list:
            print(f"DEBUG: First cell keys: {cell_list[0].keys()}")
            print(f"DEBUG: First cell class: {cell_list[0].get('class', 'N/A')}")

        centroids = np.array([c['centroid'] for c in cell_list if 'centroid' in c])
        print("DEBUG: centroids.shape:", centroids.shape)
        # Defensive shape check
        if centroids.size == 0 or len(centroids.shape) < 2 or centroids.shape[1] < 2:
            plt.text(image.shape[1]/2, image.shape[0]/2, "Invalid centroid data for density heatmap",
                    ha='center', va='center', fontsize=16, color='white',
                    bbox=dict(facecolor='black', alpha=0.7, pad=10))
            plt.axis('off')
            plt.tight_layout()
            return fig

        # Clip bins to avoid tiny images + many bins
        bins_x = min(bins, image.shape[1])
        bins_y = min(bins, image.shape[0])
        print(f"DEBUG: Using bins_x={bins_x}, bins_y={bins_y}")
        heatmap, xedges, yedges = np.histogram2d(
            centroids[:, 0],        # row (y)
            centroids[:, 1],        # col (x)
            bins=[bins_y, bins_x],
            range=[[0, image.shape[0]], [0, image.shape[1]]]
        )
        print("DEBUG: heatmap shape:", heatmap.shape)
    except Exception as e:
        print(f"Error creating density heatmap: {e}")
        plt.text(image.shape[1]/2, image.shape[0]/2, f"Error creating density heatmap: {str(e)}",
                ha='center', va='center', fontsize=20, color='white',
                bbox=dict(facecolor='black', alpha=0.7, pad=10))
        plt.axis('off')
        plt.tight_layout()
        return fig

    heatmap_smooth = gaussian_filter(heatmap, sigma=sigma)

    extent = [0, image.shape[1], image.shape[0], 0]
    plt.imshow(
        heatmap_smooth.T,
        extent=extent,
        cmap='jet',
        alpha=0.5,
        zorder=2
    )
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Cell density', rotation=270, labelpad=15)
    plt.axis('off')
    plt.tight_layout()
    return fig

def visualize_segmentation_stats(image, seg_op, inst_op, pred_sep_inst, class_names=None):
    """
    Visualize segmentation results with confidence and area statistics.

    Args:
        image: Input image
        seg_op: Semantic segmentation output
        inst_op: Instance segmentation output
        pred_sep_inst: Separated instances
        class_names: List of class names

    Returns:
        Figure object with the visualization
    """
    from matplotlib.gridspec import GridSpec
    import pandas as pd

    # Create figure
    fig = Figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)

    # Calculate statistics
    stats = calculate_segmentation_stats(seg_op, inst_op, pred_sep_inst)

    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Original Image")
    ax1.imshow(image)
    ax1.axis('off')

    # Segmentation mask
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Segmentation Mask")
    ax2.imshow(seg_op, cmap='tab10')
    ax2.axis('off')

    # Instance boundaries
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title("Instance Boundaries")
    ax3.imshow(image)
    ax3.contour(inst_op, colors='red', linewidths=0.5)
    ax3.axis('off')

    # Class statistics
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.set_title("Class Statistics")

    # Create data for table
    class_data = []
    for class_idx in stats['class_counts']:
        if class_names and class_idx < len(class_names):
            class_name = class_names[class_idx]
        else:
            class_name = f"Class {class_idx}"

        area_pct = stats['class_areas'][class_idx] / stats['total_area'] * 100 if stats['total_area'] > 0 else 0
        confidence = stats['class_confidences'][class_idx] if class_idx in stats['class_confidences'] else 0.0

        class_data.append({
            'Class': class_name,
            'Count': stats['class_counts'][class_idx],
            'Area %': f"{area_pct:.2f}%",
            'Confidence': f"{confidence:.2f}"
        })

    # Create table
    if class_data:
        df = pd.DataFrame(class_data)
        ax4.axis('tight')
        ax4.axis('off')
        table = ax4.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
    else:
        ax4.text(0.5, 0.5, "No class data available", ha='center', va='center')

    # Instance details
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_title("Instance Details")

    # Create data for instance table
    instance_data = []
    max_instances = min(10, len(stats['instance_details']))
    for i, instance in enumerate(stats['instance_details'][:max_instances]):
        class_idx = instance['class']
        if class_names and class_idx < len(class_names):
            class_name = class_names[class_idx]
        else:
            class_name = f"Class {class_idx}"

        instance_data.append({
            'ID': instance['id'],
            'Class': class_name,
            'Area': instance['area'],
            'Conf': f"{instance['confidence']:.2f}"
        })

    # Create table
    if instance_data:
        df = pd.DataFrame(instance_data)
        ax5.axis('tight')
        ax5.axis('off')
        table = ax5.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Add note if there are more instances
        if len(stats['instance_details']) > max_instances:
            ax5.text(0.5, 0.05, f"... and {len(stats['instance_details']) - max_instances} more instances", 
                    ha='center', va='center', fontsize=8, transform=ax5.transAxes)
    else:
        ax5.text(0.5, 0.5, "No instance data available", ha='center', va='center')

    fig.tight_layout()
    return fig

def calculate_segmentation_stats(seg_op, inst_op, pred_sep_inst):
    """
    Calculate statistics from segmentation data.

    Args:
        seg_op: Semantic segmentation output
        inst_op: Instance segmentation output
        pred_sep_inst: Separated instances

    Returns:
        Dictionary with statistics
    """
    import numpy as np

    # Initialize statistics dictionary
    stats = {
        'class_counts': {},  # Number of objects per class
        'class_areas': {},   # Area (pixel count) per class
        'class_confidences': {},  # Confidence scores per class
        'instance_details': [],  # Details for each instance
        'total_count': 0,    # Total number of objects
        'total_area': 0      # Total area of all objects
    }

    # Count unique instance IDs (excluding background which is 0)
    unique_instances = np.unique(pred_sep_inst)
    unique_instances = unique_instances[unique_instances > 0]  # Exclude background

    # Total count of objects
    stats['total_count'] = len(unique_instances)

    # Calculate class statistics
    for instance_id in unique_instances:
        # Create mask for this instance
        instance_mask = (pred_sep_inst == instance_id)

        # Get the most common class for this instance
        if np.any(instance_mask):
            # Handle case where instance_mask is 3D (has multiple channels)
            if seg_op.ndim == 2 and instance_mask.ndim == 3:
                # Sum along the channel dimension to get a 2D mask
                instance_mask = np.any(instance_mask, axis=2)

            # Get semantic segmentation values for this instance
            if seg_op.ndim == instance_mask.ndim:
                instance_sem = seg_op[instance_mask]

                # Get the most common class (excluding background)
                if len(instance_sem) > 0:
                    class_counts = np.bincount(instance_sem.flatten())
                    if len(class_counts) > 1:  # Ensure we have classes beyond background
                        # Get the most common non-background class
                        class_idx = np.argmax(class_counts[1:]) + 1 if len(class_counts) > 1 else 0

                        # Update class count
                        if class_idx not in stats['class_counts']:
                            stats['class_counts'][class_idx] = 0
                            stats['class_areas'][class_idx] = 0
                            stats['class_confidences'][class_idx] = []

                        stats['class_counts'][class_idx] += 1

                        # Update class area
                        area = np.sum(instance_mask)
                        stats['class_areas'][class_idx] += area
                        stats['total_area'] += area

                        # Calculate confidence score for this instance
                        confidence = np.sum(instance_sem == class_idx) / len(instance_sem)
                        stats['class_confidences'][class_idx].append(confidence)

                        # Store instance details
                        instance_details = {
                            'id': instance_id,
                            'class': class_idx,
                            'area': area,
                            'confidence': confidence,
                            # Calculate centroid (center of mass)
                            'centroid': np.array(np.where(instance_mask)).mean(axis=1) if instance_mask.ndim == 2 else None
                        }
                        stats['instance_details'].append(instance_details)

    # Calculate average confidence per class
    for class_idx in stats['class_confidences']:
        if stats['class_confidences'][class_idx]:
            stats['class_confidences'][class_idx] = np.mean(stats['class_confidences'][class_idx])
        else:
            stats['class_confidences'][class_idx] = 0.0

    return stats

def plot_classification_confidence(predicted_class, class_probabilities, class_names, calibration_temperature=None):
    """
    Create a bar chart showing confidence rates for different classes.

    Args:
        predicted_class: Index of the predicted class
        class_probabilities: Array of probabilities for each class
        class_names: List of class names
        calibration_temperature: Temperature used for calibrating confidence scores (optional)

    Returns:
        Figure object with the confidence visualization
    """
    # Create a new figure
    fig = Figure(figsize=(14, 8))
    ax = fig.add_subplot(111)

    # If class_probabilities is a single value, convert it to an array with zeros for other classes
    if not isinstance(class_probabilities, (list, np.ndarray)) or len(class_probabilities) == 1:
        probs = np.zeros(len(class_names))
        probs[predicted_class] = class_probabilities if isinstance(class_probabilities, (list, np.ndarray)) else class_probabilities
        class_probabilities = probs

    # Ensure class_probabilities has the same length as class_names
    if len(class_probabilities) != len(class_names):
        # Pad with zeros or truncate
        if len(class_probabilities) < len(class_names):
            class_probabilities = np.pad(class_probabilities, 
                                        (0, len(class_names) - len(class_probabilities)), 
                                        'constant')
        else:
            class_probabilities = class_probabilities[:len(class_names)]

    # Create bar chart
    bars = ax.bar(range(len(class_names)), class_probabilities, color='skyblue')

    # Highlight the predicted class
    if 0 <= predicted_class < len(class_names):
        bars[predicted_class].set_color('red')

    # Add labels and title
    ax.set_xlabel('Class')
    ax.set_ylabel('Confidence')
    ax.set_title('Classification Confidence by Class')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')

    # Add confidence values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')

    # Add a horizontal line at 0.5 confidence
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)

    # Highlight the predicted class with text
    if 0 <= predicted_class < len(class_names):
        predicted_name = class_names[predicted_class]
        confidence = class_probabilities[predicted_class]

        # Create text with prediction information
        text = f"Predicted: {predicted_name}\nConfidence: {confidence:.2f}"

        # Add calibration temperature information if provided
        if calibration_temperature is not None:
            text += f"\nCalibration Temp: {calibration_temperature:.1f}"

        ax.text(0.02, 0.95, text,
                transform=ax.transAxes, fontsize=16,
                bbox=dict(facecolor='white', alpha=0.8))

    # Set y-axis limit to slightly above the maximum probability
    ax.set_ylim(0, max(class_probabilities) * 1.1)

    fig.tight_layout()
    return fig
