"""
Ultimate Analysis Mode for histopathology images.

This module integrates all the enhanced components (preprocessing, ROI detection,
segmentation, classification, and analysis) into a comprehensive analysis pipeline.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
import time

def convert_numpy_types(obj):
    """
    Convert numpy types to Python native types for JSON serialization.

    Args:
        obj: Object to convert

    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj

# Import enhanced components
from processing.preprocessing import ImagePreprocessor
from processing.roi_detection import detect_roi, auto_detect_best_roi, visualize_roi
from processing.postprocessing import CellCounter
from analysis.quantitative import QuantitativeAnalyzer
from utils.visualization import ResultVisualizer

class UltimateAnalysis:
    """
    Comprehensive analysis pipeline for histopathology images.

    This class integrates all the enhanced components into a single pipeline
    that provides robust preprocessing, ROI detection, cell segmentation,
    classification, and quantitative analysis.
    """

    def __init__(self, config=None):
        """
        Initialize the UltimateAnalysis pipeline.

        Args:
            config: Configuration dictionary or path to JSON config file
        """
        # Load configuration
        self.config = self._load_config(config)

        # Initialize components
        self.preprocessor = ImagePreprocessor(
            stain_matrix=np.array(self.config.get('stain_matrix')) if 'stain_matrix' in self.config else None
        )

        self.cell_counter = CellCounter(
            class_names=self.config.get('class_names'),
            min_cell_size=self.config.get('min_cell_size', 50),
            max_cell_size=self.config.get('max_cell_size', 1000)
        )

        self.analyzer = QuantitativeAnalyzer(
            pixel_size=self.config.get('pixel_size', 0.25)
        )

        self.visualizer = ResultVisualizer(
            class_colors=self.config.get('class_colors')
        )

        # Initialize models
        self.seg_model = None
        self.cls_model = None

    def _load_config(self, config):
        """
        Load configuration from dictionary or JSON file.

        Args:
            config: Configuration dictionary or path to JSON config file

        Returns:
            Configuration dictionary
        """
        if config is None:
            # Default configuration
            return {
                'seg_model_path': "segmentation_model.keras",
                'cls_model_path': "classification_model.keras",
                'result_dir': "results",
                'pixel_size': 0.25,
                'min_cell_size': 50,
                'max_cell_size': 1000,
                'roi_detection_method': 'auto',
                'separate_touching_cells': True,
                'morphological_cleanup': True,
                'clustering_distance': 50,
                'generate_pdf_report': True
            }
        elif isinstance(config, str):
            # Load from JSON file
            try:
                with open(config) as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading configuration from {config}: {e}")
                return self._load_config(None)  # Use default config
        else:
            # Use provided dictionary
            return config

    def load_models(self, seg_model_path=None, cls_model_path=None):
        """
        Load segmentation and classification models.

        Args:
            seg_model_path: Path to segmentation model
            cls_model_path: Path to classification model
        """
        from models.model_utils import load_models

        # Use paths from config if not provided
        seg_model_path = seg_model_path or self.config.get('seg_model_path')
        cls_model_path = cls_model_path or self.config.get('cls_model_path')

        try:
            self.seg_model, self.cls_model = load_models(seg_model_path, cls_model_path)
            print(f"Models loaded successfully from {seg_model_path} and {cls_model_path}")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def preprocess(self, image_path):
        """
        Preprocess the image with enhanced methods.

        Args:
            image_path: Path to the input image

        Returns:
            Tuple of (h_channel, enhanced_image, original_image, applied_steps)
        """
        print(f"Preprocessing image: {image_path}")

        # Load and preprocess the image
        h_channel, enhanced_image, applied_steps = self.preprocessor.preprocess_pipeline(image_path)

        # Load original image for reference
        original_image = self.preprocessor.load_image(image_path)

        # Print applied preprocessing steps
        print(f"Applied preprocessing steps: {', '.join(applied_steps)}")

        return h_channel, enhanced_image, original_image, applied_steps

    def detect_roi(self, image, method=None):
        """
        Detect regions of interest in the image.

        Args:
            image: Input image
            method: ROI detection method ('auto', 'hsv', 'adaptive', 'otsu', 'hed')

        Returns:
            Binary mask of ROIs
        """
        # Use method from config if not provided
        method = method or self.config.get('roi_detection_method', 'auto')

        print(f"Detecting ROI using method: {method}")

        if method == 'auto':
            roi_mask, best_method = auto_detect_best_roi(image)
            print(f"Auto-selected ROI method: {best_method}")
        else:
            roi_mask = detect_roi(image, method=method)

        return roi_mask

    def segment_cells(self, h_channel, roi_mask=None):
        """
        Segment cells in the image.

        Args:
            h_channel: Hematoxylin channel from preprocessing
            roi_mask: Optional ROI mask to limit segmentation

        Returns:
            Binary mask of segmented cells
        """
        from models.model_utils import predict_cells

        print("Segmenting cells")

        # Ensure models are loaded
        if self.seg_model is None:
            if not self.load_models():
                raise ValueError("Segmentation model not loaded")

        # Apply ROI mask if provided
        if roi_mask is not None:
            # Expand dimensions to match h_channel
            if len(roi_mask.shape) == 2 and len(h_channel.shape) == 3:
                roi_mask = np.expand_dims(roi_mask, axis=-1)

            # Apply mask
            masked_h_channel = h_channel * roi_mask
        else:
            masked_h_channel = h_channel

        # Segment cells
        cell_mask = predict_cells(
            self.seg_model,
            masked_h_channel,
            min_size=self.config.get('min_cell_size', 50),
            morphological_cleanup=self.config.get('morphological_cleanup', True)
        )

        return cell_mask

    def classify_cells(self, h_channel, cell_mask):
        """
        Classify segmented cells.

        Args:
            h_channel: Hematoxylin channel from preprocessing
            cell_mask: Binary mask of segmented cells

        Returns:
            Tuple of (counts, cell_details)
        """
        print("Classifying cells")

        # Ensure models are loaded
        if self.cls_model is None:
            if not self.load_models():
                raise ValueError("Classification model not loaded")

        # Prepare input for classification
        # Add batch dimension (axis=0) and ensure we have channel dimension (axis=-1)
        if len(h_channel.shape) == 2:
            # If h_channel is 2D (height, width), expand to 3D with 3 channels
            h_channel_3d = np.stack([h_channel, h_channel, h_channel], axis=-1)
            input_data = np.expand_dims(h_channel_3d, axis=0)
        elif len(h_channel.shape) == 3 and h_channel.shape[2] == 1:
            # If h_channel is 3D but with only 1 channel, repeat to 3 channels
            h_channel_3d = np.concatenate([h_channel, h_channel, h_channel], axis=2)
            input_data = np.expand_dims(h_channel_3d, axis=0)
        elif len(h_channel.shape) == 3 and h_channel.shape[2] == 3:
            # If h_channel already has 3 channels, just add batch dimension
            input_data = np.expand_dims(h_channel, axis=0)
        else:
            raise ValueError(f"Unexpected h_channel shape: {h_channel.shape}. Expected (height, width) or (height, width, channels)")

        # Run classification
        class_probs = self.cls_model.predict(input_data)[0]

        # Process detections
        counts, cell_details = self.cell_counter.process_detections(
            cell_mask, class_probs
        )

        # Apply watershed to separate touching cells if needed
        if self.config.get('separate_touching_cells', True):
            print("Separating touching cells")
            separated_mask = self.cell_counter.separate_clumped_cells(cell_mask)

            # Recount with separated mask
            counts, cell_details = self.cell_counter.process_detections(
                separated_mask, class_probs
            )
            cell_mask = separated_mask

        # Remove duplicate detections
        counts, cell_details = self.cell_counter.cluster_cells(
            cell_details,
            eps=self.config.get('clustering_distance', 50)
        )

        return counts, cell_details, cell_mask

    def analyze_cells(self, cell_mask, roi_mask=None):
        """
        Perform quantitative analysis on segmented cells.

        Args:
            cell_mask: Binary mask of segmented cells
            roi_mask: Optional ROI mask for density calculation

        Returns:
            Dictionary of analysis results
        """
        print("Analyzing cells")

        # Calculate density
        density = self.analyzer.calculate_density(cell_mask, roi_mask)

        # Extract morphological features
        morphology = self.analyzer.morphological_features(cell_mask)

        # Detect anomalies
        anomalies = self.analyzer.detect_anomalies(morphology, density)

        # Perform spatial analysis
        spatial = self.analyzer.spatial_analysis(cell_mask)

        # Combine all analysis results
        analysis_results = {
            'cell_density': density,
            'morphology': morphology,
            **anomalies,
            **spatial
        }

        return analysis_results

    def visualize_results(self, original_image, enhanced_image, h_channel, cell_mask, counts, analysis_results):
        """
        Generate visualizations of the analysis results.

        Args:
            original_image: Original input image
            enhanced_image: Enhanced image from preprocessing
            h_channel: Hematoxylin channel from preprocessing
            cell_mask: Binary mask of segmented cells
            counts: Dictionary of cell counts by class
            analysis_results: Dictionary of analysis results

        Returns:
            Matplotlib figure with visualizations
        """
        print("Generating visualizations")

        # Generate dashboard visualization
        fig = self.visualizer.plot_dashboard(
            original_image,
            h_channel,
            cell_mask,
            counts,
            analysis_results
        )

        return fig

    def generate_report(self, image_path, counts, analysis_results, output_path=None):
        """
        Generate a PDF report of the analysis results.

        Args:
            image_path: Path to the input image
            counts: Dictionary of cell counts by class
            analysis_results: Dictionary of analysis results
            output_path: Path to save the PDF report

        Returns:
            Path to the generated report
        """
        if not self.config.get('generate_pdf_report', True):
            return None

        # Use default output path if not provided
        if output_path is None:
            result_dir = self.config.get('result_dir', 'results')
            os.makedirs(result_dir, exist_ok=True)

            base_name = os.path.basename(image_path).split('.')[0]
            output_path = os.path.join(result_dir, f"{base_name}_report.pdf")

        print(f"Generating PDF report: {output_path}")

        # Generate report
        self.visualizer.generate_report(
            image_path,
            counts,
            analysis_results,
            output_path
        )

        return output_path

    def save_results(self, image_path, counts, analysis_results, fig=None):
        """
        Save analysis results to files.

        Args:
            image_path: Path to the input image
            counts: Dictionary of cell counts by class
            analysis_results: Dictionary of analysis results
            fig: Optional matplotlib figure to save

        Returns:
            Dictionary of saved file paths
        """
        result_dir = self.config.get('result_dir', 'results')
        os.makedirs(result_dir, exist_ok=True)

        base_name = os.path.basename(image_path).split('.')[0]

        # Save results
        saved_files = {}

        # Save figure
        if fig is not None:
            fig_path = os.path.join(result_dir, f"{base_name}_results.png")
            fig.savefig(fig_path)
            saved_files['figure'] = fig_path

        # Save counts and analysis results as JSON
        json_path = os.path.join(result_dir, f"{base_name}_analysis.json")
        with open(json_path, 'w') as f:
            # Convert numpy types to Python native types for JSON serialization
            serializable_data = convert_numpy_types({
                'counts': counts,
                'analysis': {k: v for k, v in analysis_results.items() if not isinstance(v, np.ndarray) and k != 'morphology'}
            })
            json.dump(serializable_data, f, indent=4)
        saved_files['json'] = json_path

        # Generate PDF report
        if self.config.get('generate_pdf_report', True):
            pdf_path = self.generate_report(image_path, counts, analysis_results)
            saved_files['pdf'] = pdf_path

        return saved_files

    def analyze_image(self, image_path, roi_method=None, save_results=True):
        """
        Perform complete analysis of an image.

        Args:
            image_path: Path to the input image
            roi_method: ROI detection method ('auto', 'hsv', 'adaptive', 'otsu', 'hed')
            save_results: Whether to save results to files

        Returns:
            Dictionary of analysis results
        """
        start_time = time.time()

        # Step 1: Preprocessing
        h_channel, enhanced_image, original_image = self.preprocess(image_path)

        # Step 2: ROI detection
        roi_mask = self.detect_roi(enhanced_image, method=roi_method)

        # Step 3: Cell segmentation
        cell_mask = self.segment_cells(h_channel, roi_mask)

        # Step 4: Cell classification
        counts, cell_details, final_mask = self.classify_cells(h_channel, cell_mask)

        # Step 5: Quantitative analysis
        analysis_results = self.analyze_cells(final_mask, roi_mask)

        # Step 6: Visualization
        fig = self.visualize_results(
            original_image,
            enhanced_image,
            h_channel,
            final_mask,
            counts,
            analysis_results
        )

        # Step 7: Save results
        saved_files = {}
        if save_results:
            saved_files = self.save_results(
                image_path,
                counts,
                analysis_results,
                fig
            )

        # Combine all results
        results = {
            'counts': counts,
            'cell_details': cell_details,
            'analysis': analysis_results,
            'processing_time': time.time() - start_time,
            'saved_files': saved_files
        }

        print(f"Analysis complete in {results['processing_time']:.2f} seconds")
        print(f"Total cells: {sum(counts.values())}")
        for cell_type, count in counts.items():
            print(f"{cell_type}: {count}")
        print(f"Cell density: {analysis_results['cell_density']:.2f} cells/mm²")

        return results

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Ultimate Analysis for Histopathology Images')
    parser.add_argument('--image', type=str, default='dataset/pannuke_processed/fold1/images/1_0.png',required=True, help='Path to the input image')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file')
    parser.add_argument('--roi', type=str, default='auto', help='ROI detection method (auto, hsv, adaptive, otsu, hed)')
    args = parser.parse_args()

    # Initialize the analysis pipeline
    analyzer = UltimateAnalysis(args.config)

    # Load models
    analyzer.load_models()

    # Analyze the image
    results = analyzer.analyze_image(args.image, roi_method=args.roi)

    # Print anomaly alerts
    if results['analysis'].get('density_anomaly', False):
        print("\nWARNING: High cell density detected!")
    if results['analysis'].get('size_anomaly', False):
        print("\nWARNING: Abnormal cell sizes detected!")
    if results['analysis'].get('shape_anomaly', False):
        print("\nWARNING: Abnormal cell shapes detected!")

    # Print spatial analysis results
    print("\nSpatial Analysis:")
    print(f"Nearest neighbor distance: {results['analysis'].get('nearest_neighbor_distance', 0):.2f} μm")
    print(f"Clustering index: {results['analysis'].get('clustering_index', 0):.2f}")
    if results['analysis'].get('is_clustered', False):
        print("Cells show significant clustering")
    else:
        print("Cells are distributed randomly")
