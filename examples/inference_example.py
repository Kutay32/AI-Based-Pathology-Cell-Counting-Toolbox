"""
Inference Example Script

This script demonstrates how to use both segmentation and classification models
for inference on images.

Usage:
    python inference_example.py --image <path_to_image> --segmentation-model <path_to_segmentation_model> --classification-model <path_to_classification_model> --output <output_directory>

Options:
    --image: Path to the input image
    --segmentation-model: Path to the trained segmentation model
    --classification-model: Path to the trained classification model
    --output: Directory to save the results
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Import project modules
from models.inference import load_trained_model, predict_segmentation, get_predicted_class
from models.losses import combined_loss, MeanIoUCustom
from preprocessing.image_enhancement import preprocess_image
from utils.helpers import load_image, get_default_class_names
from analysis.cell_detection2 import summarize_prediction

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with segmentation and classification models")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--segmentation-model", type=str, required=True, help="Path to the trained segmentation model")
    parser.add_argument("--classification-model", type=str, required=True, help="Path to the trained classification model")
    parser.add_argument("--output", type=str, required=True, help="Directory to save the results")
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Load image
    try:
        print(f"Loading image from {args.image}...")
        image = load_image(args.image)
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return

    # Preprocess image
    print("Preprocessing image...")
    preprocessed_image, _ = preprocess_image(image, enhance=True, denoise=True)

    # Load segmentation model
    try:
        print(f"Loading segmentation model from {args.segmentation_model}...")
        segmentation_model = load_trained_model(
            args.segmentation_model,
            custom_objects={
                'combined_loss': combined_loss,
                'MeanIoUCustom': MeanIoUCustom
            },
            model_type='segmentation'
        )
    except Exception as e:
        print(f"Error loading segmentation model: {str(e)}")
        return

    # Load classification model
    try:
        print(f"Loading classification model from {args.classification_model}...")
        classification_model = load_trained_model(
            args.classification_model,
            custom_objects={
                'combined_loss': combined_loss,
                'MeanIoUCustom': MeanIoUCustom
            },
            model_type='classification'
        )
    except Exception as e:
        print(f"Error loading classification model: {str(e)}")
        return

    # Get class names
    class_names = get_default_class_names()

    # Run classification prediction first
    print("Running classification prediction...")
    classification_result = get_predicted_class(classification_model, preprocessed_image)
    predicted_class = classification_result['predicted_class']
    probability = classification_result['probability']
    # Get all class probabilities if available
    class_probabilities = classification_result.get('probabilities', None)
    predicted_class_name = class_names[predicted_class] if predicted_class < len(class_names) else f"Class {predicted_class}"

    # Run segmentation prediction with classification result
    print("Running segmentation prediction...")
    segmentation_mask = predict_segmentation(segmentation_model, preprocessed_image, classification_result=classification_result)

    # Print classification results
    print("\n=== Classification Results ===")
    print(f"Predicted class: {predicted_class_name}")
    print(f"Confidence: {probability:.2f}")

    # Perform cell detection and classification
    print("\n=== Cell Detection and Classification ===")
    cell_features, class_summary = summarize_prediction(
        preprocessed_image, 
        segmentation_mask, 
        class_names=class_names, 
        classification_result=classification_result,
        classification_model=classification_model
    )

    # Print cell detection and classification results
    print(f"Detected {len(cell_features)} cells")
    print("\nClass Summary:")
    for summary in class_summary:
        print(f"  {summary['ClassName']}: {summary['Cell Count']} cells detected")
        if 'Classified Count' in summary:
            print(f"    {summary['Classified Count']} cells classified as this type")

    # Save visualization
    print(f"\nSaving visualization to {args.output}...")
    plt.figure(figsize=(15, 10))

    # Original image
    plt.subplot(2, 2, 1)
    plt.title("Original")
    plt.imshow(image)
    plt.axis("off")

    # Preprocessed image with classification result
    plt.subplot(2, 2, 2)
    plt.title(f"Classification: {predicted_class_name}\nConfidence: {probability:.2f}")
    plt.imshow(preprocessed_image)
    plt.axis("off")

    # Segmentation mask
    plt.subplot(2, 2, 3)
    plt.title("Segmentation")
    plt.imshow(segmentation_mask, cmap='tab10', vmin=0, vmax=len(class_names)-1)
    plt.axis("off")

    # Cell detection and classification visualization
    plt.subplot(2, 2, 4)
    plt.title("Cell Detection and Classification")
    plt.imshow(preprocessed_image)

    # Draw bounding boxes and labels for classified cells
    for feature in cell_features:
        if 'BBox' in feature and 'PredictedClassName' in feature:
            x, y, w, h = feature['BBox']
            class_name = feature['PredictedClassName']
            probability = feature.get('ClassProbability', 0)

            # Get color for this class
            class_id = class_names.index(class_name) if class_name in class_names else 0
            color_map = plt.cm.get_cmap('tab10', len(class_names))
            color = color_map(class_id)[:3]  # RGB

            # Convert to 0-255 range for rectangle
            color = tuple(int(c * 255) for c in color)

            # Draw rectangle on the image
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2)
            plt.gca().add_patch(rect)

            # Add text label
            plt.text(x, y-5, f"{class_name} ({probability:.2f})", 
                     color='white', fontsize=8, 
                     bbox=dict(facecolor=color, alpha=0.7))

    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "inference_results.png"))

    print(f"Results saved to {os.path.join(args.output, 'inference_results.png')}")
    print("Inference complete!")

if __name__ == "__main__":
    main()
