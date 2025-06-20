"""
Ultimate Analysis Mode UI components for the Pathology Cell Counting Toolbox.

This module provides UI components for the Ultimate Analysis Mode, which integrates
all the enhanced components (preprocessing, ROI detection, segmentation, classification,
and analysis) into a comprehensive analysis pipeline.
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import time
import traceback
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QComboBox, QTabWidget, 
    QSpinBox, QCheckBox, QGroupBox, QGridLayout, QMessageBox,
    QDoubleSpinBox, QRadioButton, QButtonGroup, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ultimate analysis components
from processing.ultimate_analysis import UltimateAnalysis
from processing.roi_detection import visualize_roi

class AnalysisWorker(QThread):
    """
    Worker thread for running the analysis in the background.
    """
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, analyzer, image_path, roi_method):
        super().__init__()
        self.analyzer = analyzer
        self.image_path = image_path
        self.roi_method = roi_method

    def run(self):
        try:
            # Step 1: Preprocessing
            self.progress_signal.emit(10, "Preprocessing image...")
            h_channel, enhanced_image, original_image, applied_steps = self.analyzer.preprocess(self.image_path)
            # Update progress with applied steps
            self.progress_signal.emit(15, f"Applied preprocessing steps: {', '.join(applied_steps)}")

            # Step 2: ROI detection
            self.progress_signal.emit(25, "Detecting regions of interest...")
            roi_mask = self.analyzer.detect_roi(enhanced_image, method=self.roi_method)

            # Step 3: Cell segmentation
            self.progress_signal.emit(40, "Segmenting cells...")
            cell_mask = self.analyzer.segment_cells(h_channel, roi_mask)

            # Step 4: Cell classification
            self.progress_signal.emit(60, "Classifying cells...")
            counts, cell_details, final_mask = self.analyzer.classify_cells(h_channel, cell_mask)

            # Step 5: Quantitative analysis
            self.progress_signal.emit(75, "Analyzing cells...")
            analysis_results = self.analyzer.analyze_cells(final_mask, roi_mask)

            # Step 6: Visualization
            self.progress_signal.emit(90, "Generating visualizations...")
            fig = self.analyzer.visualize_results(
                original_image,
                enhanced_image,
                h_channel,
                final_mask,
                counts,
                analysis_results
            )

            # Step 7: Save results
            self.progress_signal.emit(95, "Saving results...")
            saved_files = self.analyzer.save_results(
                self.image_path,
                counts,
                analysis_results,
                fig
            )

            # Combine all results
            results = {
                'counts': counts,
                'cell_details': cell_details,
                'analysis': analysis_results,
                'saved_files': saved_files,
                'original_image': original_image,
                'enhanced_image': enhanced_image,
                'h_channel': h_channel,
                'roi_mask': roi_mask,
                'cell_mask': final_mask,
                'fig': fig,
                'preprocessing_steps': applied_steps
            }

            self.progress_signal.emit(100, "Analysis complete!")
            self.finished_signal.emit(results)

        except Exception as e:
            self.error_signal.emit(f"Error during analysis: {str(e)}\n{traceback.format_exc()}")

class UltimateAnalysisTab(QWidget):
    """
    Tab for the Ultimate Analysis Mode in the Pathology Cell Counting Toolbox.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize variables
        self.image_path = None
        self.analyzer = None
        self.results = None

        # Create UI components
        self.create_ui()

        # Initialize analyzer
        self.initialize_analyzer()

    def create_ui(self):
        """Create the UI components for the Ultimate Analysis tab."""
        layout = QVBoxLayout(self)

        # Top toolbar
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)

        # Load image button
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        toolbar_layout.addWidget(self.load_button)

        # Analyze button
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.analyze_image)
        self.analyze_button.setEnabled(False)
        toolbar_layout.addWidget(self.analyze_button)

        # ROI detection method
        roi_group = QGroupBox("ROI Detection Method")
        roi_layout = QHBoxLayout(roi_group)

        self.roi_method_combo = QComboBox()
        self.roi_method_combo.addItems(["Auto", "HSV", "Adaptive", "Otsu", "H&E"])
        roi_layout.addWidget(self.roi_method_combo)

        toolbar_layout.addWidget(roi_group)

        # Advanced options
        options_group = QGroupBox("Analysis Options")
        options_layout = QGridLayout(options_group)

        # Min cell size
        self.min_cell_size_spin = QSpinBox()
        self.min_cell_size_spin.setRange(10, 500)
        self.min_cell_size_spin.setValue(50)
        options_layout.addWidget(QLabel("Min Cell Size:"), 0, 0)
        options_layout.addWidget(self.min_cell_size_spin, 0, 1)

        # Max cell size
        self.max_cell_size_spin = QSpinBox()
        self.max_cell_size_spin.setRange(100, 5000)
        self.max_cell_size_spin.setValue(1000)
        options_layout.addWidget(QLabel("Max Cell Size:"), 0, 2)
        options_layout.addWidget(self.max_cell_size_spin, 0, 3)

        # Pixel size
        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(0.01, 10.0)
        self.pixel_size_spin.setValue(0.25)
        self.pixel_size_spin.setSingleStep(0.01)
        options_layout.addWidget(QLabel("Pixel Size (μm):"), 1, 0)
        options_layout.addWidget(self.pixel_size_spin, 1, 1)

        # Separate touching cells
        self.separate_cells_checkbox = QCheckBox("Separate Touching Cells")
        self.separate_cells_checkbox.setChecked(True)
        options_layout.addWidget(self.separate_cells_checkbox, 1, 2)

        # Morphological cleanup
        self.morphological_cleanup_checkbox = QCheckBox("Morphological Cleanup")
        self.morphological_cleanup_checkbox.setChecked(True)
        options_layout.addWidget(self.morphological_cleanup_checkbox, 1, 3)

        # Generate PDF report
        self.generate_pdf_checkbox = QCheckBox("Generate PDF Report")
        self.generate_pdf_checkbox.setChecked(True)
        options_layout.addWidget(self.generate_pdf_checkbox, 2, 0)

        toolbar_layout.addWidget(options_group)

        # Add toolbar to main layout
        layout.addWidget(toolbar)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% - %v")
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        # Create tabs for different views
        self.view_tabs = QTabWidget()
        self.view_tabs.currentChanged.connect(self.on_tab_changed)

        # Original Image Tab
        self.original_tab = QWidget()
        original_layout = QVBoxLayout(self.original_tab)

        # Add view controls for original tab
        original_controls = QWidget()
        original_controls_layout = QHBoxLayout(original_controls)
        self.show_roi_on_original = QCheckBox("Show ROI Overlay")
        self.show_roi_on_original.setChecked(False)
        self.show_roi_on_original.stateChanged.connect(self.update_original_view)
        original_controls_layout.addWidget(self.show_roi_on_original)
        original_layout.addWidget(original_controls)

        self.original_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        original_layout.addWidget(self.original_canvas)
        self.view_tabs.addTab(self.original_tab, "Original Image")

        # ROI Detection Tab
        self.roi_tab = QWidget()
        roi_layout = QVBoxLayout(self.roi_tab)

        # Add view controls for ROI tab
        roi_controls = QWidget()
        roi_controls_layout = QHBoxLayout(roi_controls)
        self.roi_method_selector = QComboBox()
        self.roi_method_selector.addItems(["Auto", "HSV", "Adaptive", "Otsu", "H&E"])
        self.roi_method_selector.currentIndexChanged.connect(self.update_roi_view)
        roi_controls_layout.addWidget(QLabel("ROI Method:"))
        roi_controls_layout.addWidget(self.roi_method_selector)
        roi_layout.addWidget(roi_controls)

        self.roi_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        roi_layout.addWidget(self.roi_canvas)
        self.view_tabs.addTab(self.roi_tab, "ROI Detection")

        # Cell Detection Tab
        self.cell_tab = QWidget()
        cell_layout = QVBoxLayout(self.cell_tab)
        self.cell_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        cell_layout.addWidget(self.cell_canvas)
        self.view_tabs.addTab(self.cell_tab, "Cell Detection")

        # Analysis Results Tab
        self.results_tab = QWidget()
        results_layout = QVBoxLayout(self.results_tab)

        # Results options
        results_options = QWidget()
        results_options_layout = QHBoxLayout(results_options)

        self.class_combo = QComboBox()
        results_options_layout.addWidget(QLabel("Cell Class:"))
        results_options_layout.addWidget(self.class_combo)

        self.view_combo = QComboBox()
        self.view_combo.addItems(["Dashboard", "Cell Counts", "Density Heatmap", "Morphology", "Spatial Analysis"])
        self.view_combo.currentIndexChanged.connect(self.update_results_view)
        results_options_layout.addWidget(QLabel("View:"))
        results_options_layout.addWidget(self.view_combo)

        results_options_layout.addStretch()
        results_layout.addWidget(results_options)

        self.results_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        results_layout.addWidget(self.results_canvas)

        # Results text area
        self.results_text = QLabel("No results available")
        results_layout.addWidget(self.results_text)

        self.view_tabs.addTab(self.results_tab, "Analysis Results")

        # Add tabs to main layout
        layout.addWidget(self.view_tabs)

    def initialize_analyzer(self):
        """Initialize the UltimateAnalysis object."""
        try:
            # Create default configuration
            config = {
                'seg_model_path': "segmentation_model.keras",
                'cls_model_path': "classification_model.keras",
                'result_dir': "results",
                'pixel_size': self.pixel_size_spin.value(),
                'min_cell_size': self.min_cell_size_spin.value(),
                'max_cell_size': self.max_cell_size_spin.value(),
                'roi_detection_method': self.roi_method_combo.currentText().lower(),
                'separate_touching_cells': self.separate_cells_checkbox.isChecked(),
                'morphological_cleanup': self.morphological_cleanup_checkbox.isChecked(),
                'generate_pdf_report': self.generate_pdf_checkbox.isChecked()
            }

            # Initialize analyzer
            self.analyzer = UltimateAnalysis(config)

            # Load models
            if not self.analyzer.load_models():
                QMessageBox.warning(
                    self,
                    "Model Loading Error",
                    "Failed to load models. Please check that the model files exist."
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Initialization Error",
                f"Failed to initialize Ultimate Analysis: {str(e)}\n{traceback.format_exc()}"
            )

    def load_image(self):
        """Load an image from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff)"
        )

        if file_path:
            try:
                # Store the image path
                self.image_path = file_path

                # Display the image
                self.original_canvas.figure.clear()
                ax = self.original_canvas.figure.add_subplot(111)

                # Load and display the image
                image = cv2.imread(file_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Normalize image to 0-1 range for display
                normalized_image = image.copy()
                if normalized_image.max() > 1.0:
                    normalized_image = normalized_image / 255.0

                ax.imshow(normalized_image)
                ax.set_title("Original Image")
                ax.axis('off')
                self.original_canvas.draw()

                # Enable analyze button
                self.analyze_button.setEnabled(True)

                # Switch to original image tab
                self.view_tabs.setCurrentIndex(0)

                # Update status
                self.status_label.setText(f"Loaded image: {os.path.basename(file_path)}")

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Loading Image",
                    f"Failed to load image: {str(e)}\n{traceback.format_exc()}"
                )

    def analyze_image(self):
        """Analyze the loaded image using the Ultimate Analysis pipeline."""
        if self.image_path is None:
            return

        try:
            # Update configuration with current UI settings
            config = {
                'seg_model_path': "segmentation_model.keras",
                'cls_model_path': "classification_model.keras",
                'result_dir': "results",
                'pixel_size': self.pixel_size_spin.value(),
                'min_cell_size': self.min_cell_size_spin.value(),
                'max_cell_size': self.max_cell_size_spin.value(),
                'roi_detection_method': self.roi_method_combo.currentText().lower(),
                'separate_touching_cells': self.separate_cells_checkbox.isChecked(),
                'morphological_cleanup': self.morphological_cleanup_checkbox.isChecked(),
                'generate_pdf_report': self.generate_pdf_checkbox.isChecked()
            }

            # Update analyzer configuration
            self.analyzer.config = config

            # Get ROI method
            roi_method = self.roi_method_combo.currentText().lower()
            if roi_method == "h&e":
                roi_method = "hed"  # Convert to method name used in code

            # Disable UI during analysis
            self.setEnabled(False)
            self.progress_bar.setValue(0)
            self.status_label.setText("Starting analysis...")

            # Create worker thread
            self.worker = AnalysisWorker(self.analyzer, self.image_path, roi_method)
            self.worker.progress_signal.connect(self.update_progress)
            self.worker.finished_signal.connect(self.analysis_finished)
            self.worker.error_signal.connect(self.analysis_error)

            # Start analysis
            self.worker.start()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Starting Analysis",
                f"Failed to start analysis: {str(e)}\n{traceback.format_exc()}"
            )
            self.setEnabled(True)

    def update_progress(self, value, message):
        """Update the progress bar and status message."""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def on_tab_changed(self, index):
        """Handle tab changes and update the views accordingly."""
        if self.results is None:
            return

        # Get the previous tab index
        previous_index = getattr(self, '_previous_tab_index', None)

        # Update the view based on the selected tab
        if index == 0:  # Original Image tab
            # If we're coming from the ROI Detection tab, automatically show the ROI overlay
            if previous_index == 1 and 'roi_mask' in self.results:
                self.show_roi_on_original.setChecked(True)
            self.update_original_view()
        elif index == 1:  # ROI Detection tab
            self.update_roi_view()
        elif index == 2:  # Cell Detection tab
            self.update_cell_view()
        elif index == 3:  # Analysis Results tab
            self.update_results_view()

        # Store the current index for next time
        self._previous_tab_index = index

    def update_original_view(self):
        """Update the original image view with or without ROI overlay."""
        if self.results is None:
            return

        self.original_canvas.figure.clear()
        ax = self.original_canvas.figure.add_subplot(111)

        # Get the original image
        original_image = self.results['original_image'].copy()
        if np.max(original_image) > 1.0:
            original_image = original_image / 255.0

        # Check if ROI overlay should be shown
        if hasattr(self, 'show_roi_on_original') and self.show_roi_on_original.isChecked() and 'roi_mask' in self.results:
            # Normalize ROI mask
            roi_mask = self.results['roi_mask'].astype(np.float32)
            if np.max(roi_mask) > 1.0:
                roi_mask = roi_mask / 255.0

            # Create overlay
            from processing.roi_detection import visualize_roi
            overlay = visualize_roi(original_image, roi_mask, alpha=0.3, color='red')
            ax.imshow(overlay)
        else:
            # Show original image without overlay
            ax.imshow(original_image)

        ax.set_title("Original Image")
        ax.axis('off')
        self.original_canvas.draw()

    def update_roi_view(self):
        """Update the ROI detection view with the selected method."""
        if self.results is None or 'original_image' not in self.results:
            return

        # Get the selected ROI method
        if hasattr(self, 'roi_method_selector'):
            method = self.roi_method_selector.currentText().lower()
            if method == "h&e":
                method = "hed"  # Convert to method name used in code

            # Get the original image
            original_image = self.results['original_image'].copy()
            if np.max(original_image) > 1.0:
                original_image = original_image / 255.0

            try:
                # Detect ROI with the selected method
                from processing.roi_detection import detect_roi, visualize_roi

                # Show processing status
                self.status_label.setText(f"Detecting ROI using {method} method...")

                # Detect ROI
                roi_mask = detect_roi(original_image, method=method)

                # Update the results with the new ROI mask
                self.results['roi_mask'] = roi_mask

                # Create visualization
                roi_vis = visualize_roi(original_image, roi_mask, alpha=0.5, color='red')

                # Update the canvas
                self.roi_canvas.figure.clear()
                ax = self.roi_canvas.figure.add_subplot(111)
                ax.imshow(roi_vis)
                ax.set_title(f"Region of Interest Detection ({method.upper()})")
                ax.axis('off')
                self.roi_canvas.draw()

                # Update status
                self.status_label.setText(f"ROI detection complete using {method} method")

            except Exception as e:
                # Show error message
                self.status_label.setText(f"Error detecting ROI: {str(e)}")
                QMessageBox.critical(
                    self,
                    "ROI Detection Error",
                    f"Failed to detect ROI using {method} method: {str(e)}"
                )

    def update_cell_view(self):
        """Update the cell detection view."""
        if self.results is None or 'original_image' not in self.results or 'cell_mask' not in self.results:
            return

        try:
            # Get the original image and cell mask
            original_image = self.results['original_image'].copy()
            if np.max(original_image) > 1.0:
                original_image = original_image / 255.0

            cell_mask = self.results['cell_mask']

            # Create cell detection visualization
            from utils.visualization import ResultVisualizer
            visualizer = ResultVisualizer()
            cell_vis = visualizer.overlay_heatmap(original_image, cell_mask)

            # Update the canvas
            self.cell_canvas.figure.clear()
            ax = self.cell_canvas.figure.add_subplot(111)
            ax.imshow(cell_vis)
            ax.set_title("Cell Detection")
            ax.axis('off')
            self.cell_canvas.draw()

        except Exception as e:
            # Show error message
            self.status_label.setText(f"Error updating cell view: {str(e)}")
            QMessageBox.critical(
                self,
                "Cell View Error",
                f"Failed to update cell detection view: {str(e)}"
            )

    def analysis_finished(self, results):
        """Handle the completion of the analysis."""
        # Store results
        self.results = results

        # Update UI
        self.setEnabled(True)
        self.status_label.setText("Analysis complete!")

        # Update class combo box
        self.class_combo.clear()
        self.class_combo.addItems(list(results['counts'].keys()))

        # Automatically enable ROI overlay on original image
        if 'roi_mask' in results and hasattr(self, 'show_roi_on_original'):
            self.show_roi_on_original.setChecked(True)

        # Set the ROI method selector to the method used in the analysis
        if hasattr(self, 'roi_method_selector'):
            # Get the ROI method used in the analysis (from the config)
            roi_method = self.analyzer.config.get('roi_detection_method', 'auto')

            # Convert method name for display
            if roi_method == 'hed':
                roi_method = 'h&e'

            # Find the index of the method in the selector (case-insensitive)
            method_index = self.roi_method_selector.findText(roi_method, Qt.MatchFixedString)
            if method_index >= 0:
                # Set the current index without triggering the currentIndexChanged signal
                self.roi_method_selector.blockSignals(True)
                self.roi_method_selector.setCurrentIndex(method_index)
                self.roi_method_selector.blockSignals(False)

        # Display ROI detection results
        self.roi_canvas.figure.clear()
        ax = self.roi_canvas.figure.add_subplot(111)

        # Create ROI visualization
        # Make sure roi_mask is properly normalized to 0-1 range
        roi_mask = results['roi_mask'].astype(np.float32)
        if np.max(roi_mask) > 1.0:
            roi_mask = roi_mask / 255.0

        # Ensure original_image is in the correct format for visualization
        original_image = results['original_image'].copy()
        if np.max(original_image) > 1.0:
            original_image = original_image / 255.0

        roi_vis = visualize_roi(original_image, roi_mask, alpha=0.5, color='red')
        ax.imshow(roi_vis)
        ax.set_title("Region of Interest Detection")
        ax.axis('off')
        self.roi_canvas.draw()

        # Display cell detection results
        self.cell_canvas.figure.clear()
        ax = self.cell_canvas.figure.add_subplot(111)

        # Create cell detection visualization
        from utils.visualization import ResultVisualizer
        visualizer = ResultVisualizer()
        cell_vis = visualizer.overlay_heatmap(results['original_image'], results['cell_mask'])
        ax.imshow(cell_vis)
        ax.set_title("Cell Detection")
        ax.axis('off')
        self.cell_canvas.draw()

        # Display analysis results
        self.update_results_view()

        # Switch to results tab
        self.view_tabs.setCurrentIndex(3)

        # Show success message
        QMessageBox.information(
            self,
            "Analysis Complete",
            f"Analysis completed successfully!\n\n"
            f"Total cells: {sum(results['counts'].values())}\n"
            f"Cell density: {results['analysis']['cell_density']:.2f} cells/mm²\n\n"
            f"Results saved to: {os.path.abspath(self.analyzer.config['result_dir'])}"
        )

    def analysis_error(self, error_message):
        """Handle errors during analysis."""
        self.setEnabled(True)
        self.status_label.setText("Analysis failed!")

        QMessageBox.critical(
            self,
            "Analysis Error",
            error_message
        )

    def update_results_view(self):
        """Update the results view based on the selected view type."""
        if self.results is None:
            return

        view_type = self.view_combo.currentText()

        if view_type == "Dashboard":
            # Display the dashboard figure
            self.results_canvas.figure = self.results['fig']
            self.results_canvas.draw()

            # Update text results
            counts = self.results['counts']
            analysis = self.results['analysis']

            results_text = f"<h3>Analysis Results</h3>"

            # Display preprocessing steps
            if 'preprocessing_steps' in self.results:
                results_text += f"<p><b>Preprocessing Applied Steps:</b></p>"
                results_text += "<ul>"
                for step in self.results['preprocessing_steps']:
                    results_text += f"<li>{step}</li>"
                results_text += "</ul>"

            results_text += f"<p><b>Total cells:</b> {sum(counts.values())}</p>"
            for cell_type, count in counts.items():
                results_text += f"<p><b>{cell_type}:</b> {count}</p>"

            results_text += f"<p><b>Cell density:</b> {analysis['cell_density']:.2f} cells/mm²</p>"

            # Add anomaly alerts
            if analysis.get('density_anomaly', False):
                results_text += f"<p style='color:red'><b>WARNING:</b> High cell density detected!</p>"
            if analysis.get('size_anomaly', False):
                results_text += f"<p style='color:red'><b>WARNING:</b> Abnormal cell sizes detected!</p>"
            if analysis.get('shape_anomaly', False):
                results_text += f"<p style='color:red'><b>WARNING:</b> Abnormal cell shapes detected!</p>"

            self.results_text.setText(results_text)
            self.results_text.setTextFormat(Qt.RichText)

        elif view_type == "Cell Counts":
            # Create a bar chart of cell counts
            self.results_canvas.figure.clear()
            ax = self.results_canvas.figure.add_subplot(111)

            counts = self.results['counts']
            classes = list(counts.keys())
            values = list(counts.values())

            bars = ax.bar(classes, values, color='skyblue')
            ax.set_title('Cell Counts by Class')
            ax.set_xlabel('Cell Class')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)

            # Add count labels on top of bars
            for i, count in enumerate(values):
                ax.text(i, count + 0.1, str(count), ha='center', va='bottom')

            self.results_canvas.figure.tight_layout()
            self.results_canvas.draw()

        elif view_type == "Density Heatmap":
            # Create a density heatmap
            self.results_canvas.figure.clear()
            ax = self.results_canvas.figure.add_subplot(111)

            # Get selected class
            class_name = self.class_combo.currentText()

            # Create binary mask for the selected class
            cell_details = self.results['cell_details']
            h, w = self.results['cell_mask'].shape
            class_mask = np.zeros((h, w), dtype=np.uint8)

            for cell in cell_details:
                if cell['class'] == class_name:
                    y, x = cell['centroid']
                    y, x = int(y), int(x)
                    if 0 <= y < h and 0 <= x < w:
                        class_mask[y, x] = 1

            # Apply Gaussian blur to create density map
            density_map = cv2.GaussianBlur(class_mask, (51, 51), 0)

            # Display density map
            ax.imshow(self.results['original_image'])
            heatmap = ax.imshow(density_map, cmap='hot', alpha=0.5)
            ax.set_title(f'Cell Density Heatmap ({class_name})')
            ax.axis('off')

            # Add colorbar
            self.results_canvas.figure.colorbar(heatmap, ax=ax, label='Density')

            self.results_canvas.figure.tight_layout()
            self.results_canvas.draw()

        elif view_type == "Morphology":
            # Display morphology histograms
            self.results_canvas.figure.clear()
            gs = self.results_canvas.figure.add_gridspec(2, 2)

            # Extract features
            morphology = self.results['analysis'].get('morphology', [])
            if not morphology:
                ax = self.results_canvas.figure.add_subplot(111)
                ax.text(0.5, 0.5, "No morphology data available", 
                       ha='center', va='center', fontsize=14)
                ax.axis('off')
                self.results_canvas.draw()
                return

            areas = [f['area'] for f in morphology]
            circularities = [f['circularity'] for f in morphology]
            eccentricities = [f['eccentricity'] for f in morphology]
            solidities = [f['solidity'] for f in morphology]

            # Plot histograms
            ax1 = self.results_canvas.figure.add_subplot(gs[0, 0])
            ax1.hist(areas, bins=20)
            ax1.set_title('Cell Area Distribution')
            ax1.set_xlabel('Area (μm²)')

            ax2 = self.results_canvas.figure.add_subplot(gs[0, 1])
            ax2.hist(circularities, bins=20)
            ax2.set_title('Circularity Distribution')
            ax2.set_xlabel('Circularity (0-1)')

            ax3 = self.results_canvas.figure.add_subplot(gs[1, 0])
            ax3.hist(eccentricities, bins=20)
            ax3.set_title('Eccentricity Distribution')
            ax3.set_xlabel('Eccentricity (0-1)')

            ax4 = self.results_canvas.figure.add_subplot(gs[1, 1])
            ax4.hist(solidities, bins=20)
            ax4.set_title('Solidity Distribution')
            ax4.set_xlabel('Solidity (0-1)')

            self.results_canvas.figure.tight_layout()
            self.results_canvas.draw()

        elif view_type == "Spatial Analysis":
            # Display spatial analysis
            self.results_canvas.figure.clear()
            ax = self.results_canvas.figure.add_subplot(111)

            # Display the image with cell centroids
            ax.imshow(self.results['original_image'])

            # Plot cell centroids
            cell_details = self.results['cell_details']
            for cell in cell_details:
                y, x = cell['centroid']
                class_name = cell['class']
                color = 'red'  # Default color

                # Use different colors for different classes
                if class_name == "Neoplastic":
                    color = 'red'
                elif class_name == "Inflammatory":
                    color = 'green'
                elif class_name == "Connective":
                    color = 'blue'
                elif class_name == "Dead":
                    color = 'yellow'
                elif class_name == "Epithelial":
                    color = 'magenta'

                ax.plot(x, y, 'o', color=color, markersize=5)

            # Add spatial analysis info
            analysis = self.results['analysis']
            info_text = f"Nearest Neighbor Distance: {analysis.get('nearest_neighbor_distance', 0):.2f} μm\n"
            info_text += f"Clustering Index: {analysis.get('clustering_index', 0):.2f}\n"
            info_text += f"Clustered: {'Yes' if analysis.get('is_clustered', False) else 'No'}"

            ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8))

            ax.set_title("Spatial Distribution Analysis")
            ax.axis('off')
            self.results_canvas.draw()

            # Update text results
            results_text = f"<h3>Spatial Analysis Results</h3>"
            results_text += f"<p><b>Nearest neighbor distance:</b> {analysis.get('nearest_neighbor_distance', 0):.2f} μm</p>"
            results_text += f"<p><b>Clustering index:</b> {analysis.get('clustering_index', 0):.2f}</p>"
            if analysis.get('is_clustered', False):
                results_text += f"<p><b>Distribution:</b> Cells show significant clustering</p>"
            else:
                results_text += f"<p><b>Distribution:</b> Cells are distributed randomly</p>"

            self.results_text.setText(results_text)
            self.results_text.setTextFormat(Qt.RichText)
