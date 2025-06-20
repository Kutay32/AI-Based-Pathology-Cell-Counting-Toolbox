"""
AI-Based Pathology Cell Counting Toolbox

This is the main entry point for the application. It provides a command-line interface
and launches the GUI for the AI-Based Pathology Cell Counting Toolbox.

Usage:
    python main.py [--cli] [--image PATH] [--segmentation-model PATH] [--classification-model PATH] [--output PATH] [--confidence-threshold THRESHOLD]
    python main.py --analyze-dataset [--dataset PATH] [--max-files N] [--output PATH]

Options:
    --cli                      Run in command-line mode instead of GUI mode
    --image PATH               Path to the input image
    --model PATH               Path to the pre-trained model (for backward compatibility)
    --segmentation-model PATH  Path to the pre-trained segmentation model
    --classification-model PATH Path to the pre-trained classification model
    --output PATH              Path to save the output results
    --confidence-threshold THRESHOLD  Confidence threshold for cell detection (default: 0.3)
    --analyze-dataset          Analyze a dataset and generate a report
    --dataset PATH             Path to the dataset directory
    --max-files N              Maximum number of files to analyze per folder (default: 100)

Notes:
    When both classification and segmentation models are provided:

    1. The classification result is used to guide the segmentation process. This helps ensure 
       consistency between the overall image classification and the cell-level segmentation, 
       particularly for cases where the classification model detects a specific cell type 
       (e.g., neoplastic) but the segmentation model might have lower confidence for that cell type.

    2. The classification result is also used to define the cell types and counts. If the 
       classification model predicts a specific cell type with high confidence (>0.7), all 
       detected cells will be assigned to that cell type, regardless of their segmentation class.
       This ensures that the cell counts reflect the classification model's prediction, which
       is often more accurate for overall tissue type determination.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import warnings

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from ui.app import main

from models.losses import MeanIoUCustom, combined_loss
from models.inference import load_trained_model, predict_segmentation, get_predicted_class
from preprocessing.image_enhancement import preprocess_image
from analysis.cell_detection2 import count_cells, analyze_cell_morphology, summarize_prediction
from visualization.visualization import visualize_prediction, generate_summary_report
from utils.helpers import load_image, save_results_to_csv, generate_report_filename, get_default_class_names
from utils.dataset_analysis import analyze_dataset, visualize_dataset_sample, generate_dataset_report
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QScrollArea
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pathology_toolbox.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress unnecessary warnings
warnings.filterwarnings('ignore', category=UserWarning)


class PathologyAnalyzer:
    """Main class for pathology image analysis."""

    def __init__(self):
        """Initialize the analyzer."""
        self.segmentation_model = None
        self.classification_model = None
        self.segmentation_class_names = get_default_class_names()
        self.classification_class_names = get_default_class_names()
        self.custom_objects = {
            'combined_loss': combined_loss,
            'MeanIoUCustom': MeanIoUCustom
        }

    def load_models(self, segmentation_model_path: str,
                    classification_model_path: Optional[str] = None) -> bool:
        """
        Load segmentation and classification models.

        Args:
            segmentation_model_path: Path to segmentation model
            classification_model_path: Optional path to classification model

        Returns:
            bool: True if models loaded successfully
        """
        try:
            # Load segmentation model
            if not Path(segmentation_model_path).exists():
                raise FileNotFoundError(f"Segmentation model not found: {segmentation_model_path}")

            logger.info(f"Loading segmentation model from {segmentation_model_path}")
            self.segmentation_model = load_trained_model(
                segmentation_model_path,
                custom_objects=self.custom_objects,
                model_type='segmentation'
            )

            # Load classification model if provided
            if classification_model_path:
                if not Path(classification_model_path).exists():
                    logger.warning(f"Classification model not found: {classification_model_path}")
                    return True  # Continue with segmentation only

                logger.info(f"Loading classification model from {classification_model_path}")
                self.classification_model = load_trained_model(
                    classification_model_path,
                    custom_objects={
                        'combined_loss': combined_loss,
                        'MeanIoUCustom': MeanIoUCustom
                    },
                    model_type='classification'
                )

            return True

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False

    def analyze_image(self, image_path: str, confidence_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Analyze a single image.

        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence threshold for class prediction (default: 0.3)

        Returns:
            Dict containing analysis results
        """
        if self.segmentation_model is None:
            raise ValueError("Segmentation model not loaded")

        try:
            # Load and validate image
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            logger.info(f"Loading image from {image_path}")
            image = load_image(image_path)

            if image is None:
                raise ValueError("Failed to load image")

            # Preprocess image
            logger.info("Preprocessing image")
            preprocessed_image, enhancement_info = preprocess_image(
                image, enhance=True, denoise=True
            )

            # Run classification first if model available
            classification_result = None
            if self.classification_model:
                logger.info("Running classification prediction")
                result = get_predicted_class(
                    self.classification_model, preprocessed_image
                )
                predicted_class = result['predicted_class']
                class_probability = result['probability']
                # Store full class probabilities if available
                class_probabilities = result.get('probabilities', None)

                predicted_class_name = (
                    self.classification_class_names[predicted_class]
                    if predicted_class < len(self.classification_class_names)
                    else f"Class {predicted_class}"
                )
                classification_result = {
                    'predicted_class': predicted_class,
                    'class_name': predicted_class_name,
                    'probability': class_probability,
                    'probabilities': class_probabilities
                }
                logger.info(f"Classification result: {predicted_class_name} (confidence: {class_probability:.2f})")

            # 2. Segmentation, using classification result as guidance if desired
            logger.info("Running segmentation prediction")
            mask = predict_segmentation(
                self.segmentation_model,
                preprocessed_image,
                classification_result=classification_result
            )
            # mask is your final segmentation mask

            # Count cells using classification result if available
            logger.info("Counting cells")
            # Use classification class names if classification model is available
            class_names_to_use = self.classification_class_names if self.classification_model else self.segmentation_class_names
            counts = count_cells(mask, class_names_to_use, classification_result)

            # Generate morphology analysis using classification result if available
            logger.info("Analyzing cell morphology")
            all_features, class_summary = summarize_prediction(
                preprocessed_image, mask, class_names=class_names_to_use, 
                classification_result=classification_result
            )

            return {
                'original_image': image,
                'preprocessed_image': preprocessed_image,
                'segmentation_mask': mask,
                'cell_counts': counts,
                'classification_result': classification_result,
                'morphology_features': all_features,
                'class_summary': class_summary,
                'enhancement_info': enhancement_info
            }

        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            raise


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="AI-Based Pathology Cell Counting Toolbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --cli --image sample.jpg --segmentation-model model.keras
  python main.py --analyze-dataset --dataset ./data --output ./results
  python main.py  # Run GUI mode
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--cli", action="store_true",
                            help="Run in command-line mode")
    mode_group.add_argument("--analyze-dataset", action="store_true",
                            help="Analyze a dataset and generate a report")

    # File paths
    parser.add_argument("--image", type=str,
                        help="Path to the input image")
    parser.add_argument("--model", type=str, default="segmentation_model.keras",
                        help="Path to the pre-trained model (for backward compatibility)")
    parser.add_argument("--segmentation-model", type=str, default="segmentation_model.keras",
                        help="Path to the pre-trained segmentation model")
    parser.add_argument("--classification-model", type=str,
                        help="Path to the pre-trained classification model")
    parser.add_argument("--output", type=str,
                        help="Path to save the output results")
    parser.add_argument("--dataset", type=str,
                        help="Path to the dataset directory")

    # Configuration
    parser.add_argument("--max-files", type=int, default=100,
                        help="Maximum number of files to analyze per folder")
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                        help="Confidence threshold for cell detection (default: 0.3)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--no-gui-fallback", action="store_true",
                        help="Don't fall back to CLI if GUI fails")

    return parser


def validate_cli_args(args: argparse.Namespace) -> bool:
    """
    Validate command-line arguments for CLI mode.

    Args:
        args: Parsed command-line arguments

    Returns:
        bool: True if arguments are valid
    """
    if not args.image:
        logger.error("Image path is required in CLI mode")
        print("Usage: python main.py --cli --image PATH [options]")
        return False

    if not Path(args.image).exists():
        logger.error(f"Image file not found: {args.image}")
        return False

    # Determine segmentation model path (backward compatibility)
    segmentation_model_path = args.segmentation_model
    if args.model != "segmentation_model.keras" and args.segmentation_model == "segmentation_model.keras":
        segmentation_model_path = args.model

    if not Path(segmentation_model_path).exists():
        logger.error(f"Segmentation model not found: {segmentation_model_path}")
        return False

    return True


def save_analysis_results(results: Dict[str, Any], output_dir: str,
                          image_name: str) -> None:
    """
    Save analysis results to files.

    Args:
        results: Analysis results dictionary
        output_dir: Output directory path
        image_name: Name of the analyzed image
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save visualization
    logger.info("Saving visualization")
    save_visualization(results, output_path, image_name)

    # Save cell counts
    counts_file = output_path / f"{image_name}_cell_counts.csv"
    counts_df = pd.DataFrame([
        {"Class": k, "Count": v}
        for k, v in results['cell_counts'].items()
        if k != "Background"
    ])
    save_results_to_csv(counts_df, str(counts_file))
    logger.info(f"Cell counts saved to {counts_file}")

    # Save classification results if available
    if results['classification_result']:
        classification_file = output_path / f"{image_name}_classification.csv"
        classification_df = pd.DataFrame([{
            "Predicted_Class": results['classification_result']['class_name'],
            "Confidence": results['classification_result']['probability']
        }])
        save_results_to_csv(classification_df, str(classification_file))
        logger.info(f"Classification results saved to {classification_file}")


def save_visualization(results: Dict[str, Any], output_path: Path,
                       image_name: str) -> None:
    """Save visualization plots."""
    classification_result = results['classification_result']

    if classification_result:
        # 4 subplots with classification
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(results['original_image'])
        axes[0].set_title("Original", fontsize=14)
        axes[0].axis("off")

        axes[1].imshow(results['preprocessed_image'])
        axes[1].set_title("Preprocessed", fontsize=14)
        axes[1].axis("off")

        axes[2].imshow(results['preprocessed_image'])
        axes[2].set_title(
            f"Classification: {classification_result['class_name']}\n"
            f"Confidence: {classification_result['probability']:.3f}",
            fontsize=14
        )
        axes[2].axis("off")

        axes[3].imshow(results['segmentation_mask'], cmap='tab10')
        axes[3].set_title("Segmentation", fontsize=14)
        axes[3].axis("off")
    else:
        # 3 subplots without classification
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(results['original_image'])
        axes[0].set_title("Original", fontsize=14)
        axes[0].axis("off")

        axes[1].imshow(results['preprocessed_image'])
        axes[1].set_title("Preprocessed", fontsize=14)
        axes[1].axis("off")

        axes[2].imshow(results['segmentation_mask'], cmap='tab10')
        axes[2].set_title("Segmentation", fontsize=14)
        axes[2].axis("off")

    plt.tight_layout()
    visualization_file = output_path / f"{image_name}_analysis.png"
    plt.savefig(visualization_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Visualization saved to {visualization_file}")


def run_cli_mode(args: argparse.Namespace) -> bool:
    """
    Run the application in command-line mode.

    Args:
        args: Parsed command-line arguments

    Returns:
        bool: True if successful
    """
    logger.info("Running in command-line mode")

    # Validate arguments
    if not validate_cli_args(args):
        return False

    try:
        # Initialize analyzer
        analyzer = PathologyAnalyzer()

        # Determine model paths
        segmentation_model_path = args.segmentation_model
        if args.model != "segmentation_model.keras" and args.segmentation_model == "segmentation_model.keras":
            segmentation_model_path = args.model

        # Load models
        if not analyzer.load_models(segmentation_model_path, args.classification_model):
            return False

        # Analyze image with specified confidence threshold
        results = analyzer.analyze_image(args.image, confidence_threshold=args.confidence_threshold)

        # Print results
        print_analysis_results(results)

        # Save results if output path provided
        if args.output:
            image_name = Path(args.image).stem
            save_analysis_results(results, args.output, image_name)

        logger.info("CLI analysis completed successfully")
        return True

    except Exception as e:
        logger.error(f"CLI mode failed: {str(e)}")
        return False


def print_analysis_results(results: Dict[str, Any]) -> None:
    """Print analysis results to console."""
    print("\n" + "=" * 50)
    print("ANALYSIS RESULTS")
    print("=" * 50)

    # Classification results
    if results['classification_result']:
        print("\n📊 CLASSIFICATION RESULTS:")
        print(f"   Predicted class: {results['classification_result']['class_name']}")
        print(f"   Confidence: {results['classification_result']['probability']:.3f}")

    # Cell count results
    print("\n🔢 CELL COUNT RESULTS:")
    total_cells = 0
    for class_name, count in results['cell_counts'].items():
        if class_name != "Background":
            print(f"   {class_name}: {count} cells")
            total_cells += count
    print(f"   Total cells: {total_cells}")

    # Summary statistics
    if results['class_summary']:
        print("\n📈 MORPHOLOGY SUMMARY:")
        for class_name, summary in results['class_summary'].items():
            if class_name != "Background" and summary:
                print(f"   {class_name}:")
                if 'mean_area' in summary:
                    print(f"     Average area: {summary['mean_area']:.1f} pixels")
                if 'mean_perimeter' in summary:
                    print(f"     Average perimeter: {summary['mean_perimeter']:.1f} pixels")


def run_gui_mode(args: argparse.Namespace) -> bool:
    """
    Run the application in GUI mode.

    Args:
        args: Parsed command-line arguments

    Returns:
        bool: True if successful
    """
    logger.info("Running in GUI mode")

    try:
        from ui.app import main as run_app
        run_app()
        return True
    except ImportError as e:
        logger.error(f"GUI dependencies not available: {str(e)}")
        if not args.no_gui_fallback:
            logger.info("Falling back to CLI mode. Use --help for CLI options.")
            return False
        else:
            logger.error("GUI fallback disabled. Install PyQt5: pip install PyQt5")
            return False
    except Exception as e:
        logger.error(f"GUI mode failed: {str(e)}")
        return False


def run_dataset_analysis_mode(args: argparse.Namespace) -> bool:
    """
    Run dataset analysis mode.

    Args:
        args: Parsed command-line arguments

    Returns:
        bool: True if successful
    """
    logger.info("Running dataset analysis mode")

    if not args.dataset:
        logger.error("Dataset path is required for dataset analysis")
        print("Usage: python main.py --analyze-dataset --dataset PATH [options]")
        return False

    if not Path(args.dataset).exists():
        logger.error(f"Dataset directory not found: {args.dataset}")
        return False

    try:
        logger.info(f"Analyzing dataset at {args.dataset}")
        logger.info(f"Maximum files per folder: {args.max_files}")

        # Analyze dataset
        stats = analyze_dataset(args.dataset, args.max_files)

        # Print statistics
        print_dataset_statistics(stats)

        # Save results if output path provided
        if args.output:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)

            # Generate visualization
            visualization_path = output_path / "dataset_visualization.png"
            logger.info(f"Generating visualization: {visualization_path}")
            visualize_dataset_sample(args.dataset, min(10, args.max_files), str(visualization_path))

            # Generate report
            report_path = output_path / "dataset_report.html"
            logger.info(f"Generating report: {report_path}")
            generate_dataset_report(args.dataset, args.max_files, str(report_path))

        logger.info("Dataset analysis completed successfully")
        return True

    except Exception as e:
        logger.error(f"Dataset analysis failed: {str(e)}")
        return False


def print_dataset_statistics(stats: Dict[str, Any]) -> None:
    """Print dataset statistics to console."""
    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    print(f"📁 Total images: {stats['image_count']}")
    print(f"🏷️  Total labels: {stats['label_count']}")
    print(f"📏 Unique image sizes: {len(set(stats['image_sizes']))}")
    print(f"🎨 Image channels: {list(set(stats['image_channels']))}")
    print(f"🔢 Image bit depths: {list(set(stats['image_bit_depths']))}")
    print(f"📊 Pixel value range: {stats['pixel_value_ranges']['min']} - {stats['pixel_value_ranges']['max']}")
    print(f"🎯 Unique label values: {stats['label_values']}")


def main() -> int:
    """
    Main function.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        # Parse arguments
        parser = setup_argument_parser()
        args = parser.parse_args()

        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Run appropriate mode
        success = False
        if args.analyze_dataset:
            success = run_dataset_analysis_mode(args)
        elif args.cli:
            success = run_cli_mode(args)
        else:
            success = run_gui_mode(args)

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    main()
