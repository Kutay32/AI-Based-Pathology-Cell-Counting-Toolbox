
"""
User interface for the AI-Based Pathology Cell Counting Toolbox.
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json
import time
import traceback
import datetime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QComboBox, QTabWidget, 
    QSpinBox, QCheckBox, QGroupBox, QGridLayout, QMessageBox,
    QDoubleSpinBox, QRadioButton, QButtonGroup, QProgressBar,QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import matplotlib as mpl
# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom tabs
from ui.slide_inference_tab import SlideInferenceTab

# Import original modules
from models.losses import MeanIoUCustom, combined_loss
from models.inference import load_trained_model, predict_segmentation, get_predicted_class
from preprocessing.image_enhancement import preprocess_image, color_deconvolution
from analysis.cell_detection2 import count_cells, analyze_cell_morphology, summarize_prediction
from visualization.visualization import (
    visualize_prediction, plot_class_based_analysis, 
    visualize_cell_detection, plot_cell_counts, 
    plot_cell_density_heatmap, generate_summary_report,
    plot_cells_on_image, plot_density_heatmap, plot_classification_confidence
)
from utils.dataset_analysis import analyze_dataset, visualize_dataset_sample, generate_dataset_report

# Import enhanced modules
try:
    from processing.preprocessing import ImagePreprocessor
    from models.model_utils import load_models, predict_cells, adaptive_thresholding
    from processing.postprocessing import CellCounter
    from analysis.quantitative import QuantitativeAnalyzer
    from utils.visualization import ResultVisualizer
    from processing.roi_detection import detect_roi, auto_detect_best_roi, visualize_roi
    from ui.tasks_tab import TasksTab
    ENHANCED_MODE_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    ENHANCED_MODE_AVAILABLE = False
    print("Enhanced mode modules not found. Some features will be disabled.")

class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding plots in the UI."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class PathologyCellCounterApp(QMainWindow):
    """Main application window for the Pathology Cell Counter."""

    # Default class colors for visualization
    DEFAULT_CLASS_COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']

    def extract_cells_from_mask(self, image, pred_mask, class_id):
        """
        Extract cell contours and centroids from the predicted mask.

        Args:
            image: Input image
            pred_mask: Predicted segmentation mask
            class_id: Class ID to extract cells for

        Returns:
            List of dictionaries with 'contour', 'centroid', and 'class' keys
        """
        import cv2
        import numpy as np

        # Create a list to store cell information
        cell_list = []

        # Split touching cells to get instance mask
        from analysis.cell_detection2 import split_touching_cells
        instance_mask = split_touching_cells(pred_mask, class_id)

        # Find unique labels in the mask (excluding 0 which is background)
        unique_labels = np.unique(instance_mask)
        unique_labels = unique_labels[unique_labels > 0]

        # For each cell (label)
        for label in unique_labels:
            # Create a binary mask for this cell
            cell_mask = (instance_mask == label).astype(np.uint8)

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

            # Calculate centroid using moments
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                # Fallback to contour center if moments fail
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w // 2, y + h // 2

            # Convert contour to the format expected by visualization methods
            # The contour from cv2.findContours is a list of points with shape (N, 1, 2)
            # We need to convert it to a numpy array with shape (N, 2)
            contour_points = contour.reshape(-1, 2)

            # Swap x and y coordinates to match the expected format (y, x)
            contour_points = np.flip(contour_points, axis=1)

            # Add cell information to the list
            cell_list.append({
                'contour': contour_points,
                'centroid': (cy, cx),  # (y, x) format
                'class': class_id
            })

        return cell_list
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI-Based Pathology Cell Counting Toolbox")
        self.setGeometry(100, 100, 1200, 800)

        # Apply stylesheet for a more modern look
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f7;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #e1e1e1;
                border: 1px solid #cccccc;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 12px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 1px solid white;
            }
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QGroupBox {
                border: 1px solid #cccccc;
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                background-color: #f5f5f7;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 4px 8px;
                background-color: white;
            }
            QLabel {
                color: #333333;
            }
            QCheckBox {
                spacing: 8px;
            }
            QRadioButton {
                spacing: 8px;
            }
        """)

        # Initialize variables
        self.image = None
        self.preprocessed_image = None
        self.enhanced_image = None
        self.h_channel = None
        self.pred_mask = None
        self.segmentation_model = None
        self.classification_model = None
        self.predicted_class = None
        self.class_probability = None
        self.segmentation_class_names = ["Background", "Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
        self.classification_class_names = ["Background", "Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
        self.class_names = self.classification_class_names  # Default to classification class names

        # Enhanced mode components
        self.enhanced_mode = ENHANCED_MODE_AVAILABLE
        self.config = self.load_config("config.json") if self.enhanced_mode else {}
        self.preprocessor = None
        self.cell_counter = None
        self.analyzer = None
        self.visualizer = None
        self.cell_details = None
        self.analysis_results = None

        # Advanced features components
        self.data_augmentation_enabled = False
        self.active_learning_enabled = False
        self.edge_ai_enabled = False
        self.quantized_model = None
        self.active_learning_samples = []
        self.augmentation_params = {
            'rotation_range': 20,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'zoom_range': 0.1,
            'horizontal_flip': True
        }

        # Create the main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)

        # Create the UI components
        self.create_toolbar()
        self.create_tabs()

        # Load the model
        self.load_model()

    def load_config(self, config_path):
        """Load configuration from JSON file."""
        try:
            if os.path.exists(config_path):
                with open(config_path) as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return {}

    def create_toolbar(self):
        """Create the toolbar with buttons for loading images and running analysis."""
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)

        # Load image button
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        toolbar_layout.addWidget(self.load_button)

        # Preprocess button
        self.preprocess_button = QPushButton("Preprocess")
        self.preprocess_button.clicked.connect(self.preprocess)
        self.preprocess_button.setEnabled(False)
        toolbar_layout.addWidget(self.preprocess_button)

        # Analyze button
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.analyze)
        self.analyze_button.setEnabled(False)
        toolbar_layout.addWidget(self.analyze_button)

        # Dataset Analysis button
        self.dataset_button = QPushButton("Analyze Dataset")
        self.dataset_button.clicked.connect(self.analyze)
        toolbar_layout.addWidget(self.dataset_button)

        # Modes are now merged - no need for mode selection

        # Applied Preprocessing Steps
        preprocess_group = QGroupBox("Applied Preprocessing Steps")
        preprocess_layout = QVBoxLayout(preprocess_group)

        # Create a label to display the applied steps
        self.applied_steps_label = QLabel("No preprocessing steps applied yet")
        self.applied_steps_label.setWordWrap(True)
        preprocess_layout.addWidget(self.applied_steps_label)

        # Create a scroll area for the applied steps
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.applied_steps_label)
        preprocess_layout.addWidget(scroll_area)

        # Keep these variables for compatibility with existing code
        self.enhance_checkbox = QCheckBox("Enhance Contrast")
        self.enhance_checkbox.setChecked(True)
        self.enhance_checkbox.setVisible(False)

        self.denoise_checkbox = QCheckBox("Denoise")
        self.denoise_checkbox.setChecked(True)
        self.denoise_checkbox.setVisible(False)

        self.roi_checkbox = QCheckBox("Detect ROI")
        self.roi_checkbox.setChecked(False)
        self.roi_checkbox.setVisible(False)

        # Enhanced preprocessing options (hidden but kept for compatibility)
        if self.enhanced_mode:
            self.stain_separation_checkbox = QCheckBox("Stain Separation")
            self.stain_separation_checkbox.setChecked(True)
            self.stain_separation_checkbox.setVisible(False)

            self.adaptive_gamma_checkbox = QCheckBox("Adaptive Gamma")
            self.adaptive_gamma_checkbox.setChecked(True)
            self.adaptive_gamma_checkbox.setVisible(False)

            self.normalize_brightness_checkbox = QCheckBox("Normalize Brightness")
            self.normalize_brightness_checkbox.setChecked(True)
            self.normalize_brightness_checkbox.setVisible(False)

        toolbar_layout.addWidget(preprocess_group)

        # Add toolbar to main layout
        self.main_layout.addWidget(toolbar)

    def create_tabs(self):
        """Create tabs for different views (original, preprocessed, analysis)."""
        self.tabs = QTabWidget()

        # Original Image Tab
        self.original_tab = QWidget()
        original_layout = QVBoxLayout(self.original_tab)
        self.original_canvas = MplCanvas(self, width=10, height=6)
        original_layout.addWidget(self.original_canvas)
        self.tabs.addTab(self.original_tab, "Original Image")

        # Preprocessed Image Tab
        self.preprocessed_tab = QWidget()
        preprocessed_layout = QVBoxLayout(self.preprocessed_tab)
        self.preprocessed_canvas = MplCanvas(self, width=10, height=6)
        preprocessed_layout.addWidget(self.preprocessed_canvas)
        self.tabs.addTab(self.preprocessed_tab, "Preprocessed Image")

        # Analysis Tab
        self.analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(self.analysis_tab)

        # Analysis options
        analysis_options = QWidget()
        analysis_options_layout = QHBoxLayout(analysis_options)

        # Fixed model for analysis (segmentation_model.keras)
        self.model_label = QLabel("Model: segmentation_model.keras (fixed)")

        # Store the fixed model path
        self.fixed_analysis_model_path = "segmentation_model.keras"

        # Add the model label to the layout
        analysis_options_layout.addWidget(self.model_label)

        # Cell class is now automatically chosen by the classification model
        self.class_label = QLabel("Cell Class: Auto (using classification model)")
        analysis_options_layout.addWidget(self.class_label)

        self.view_combo = QComboBox()
        view_options = ["All Images", "ROI Detection"]

        # Enhanced view options have been removed as per requirements

        self.view_combo.addItems(view_options)
        self.view_combo.currentIndexChanged.connect(self.update_analysis_view)
        analysis_options_layout.addWidget(QLabel("View:"))
        analysis_options_layout.addWidget(self.view_combo)

        # Add ROI method selection
        self.roi_method_combo = QComboBox()
        self.roi_method_combo.addItems(["Auto", "HSV", "Adaptive", "Otsu", "HED"])
        analysis_options_layout.addWidget(QLabel("ROI Method:"))
        analysis_options_layout.addWidget(self.roi_method_combo)

        # Add enhanced options if available
        if self.enhanced_mode:
            self.min_cell_size_spin = QSpinBox()
            self.min_cell_size_spin.setRange(10, 500)
            self.min_cell_size_spin.setValue(self.config.get('min_cell_size', 50))
            analysis_options_layout.addWidget(QLabel("Min Cell Size:"))
            analysis_options_layout.addWidget(self.min_cell_size_spin)

            self.separate_cells_checkbox = QCheckBox("Separate Touching Cells")
            self.separate_cells_checkbox.setChecked(self.config.get('separate_touching_cells', True))
            analysis_options_layout.addWidget(self.separate_cells_checkbox)

            self.morphological_cleanup_checkbox = QCheckBox("Morphological Cleanup")
            self.morphological_cleanup_checkbox.setChecked(self.config.get('morphological_cleanup', True))
            analysis_options_layout.addWidget(self.morphological_cleanup_checkbox)

        analysis_options_layout.addStretch()

        analysis_layout.addWidget(analysis_options)

        # Analysis canvas
        self.analysis_canvas = MplCanvas(self, width=10, height=6)
        analysis_layout.addWidget(self.analysis_canvas)

        self.tabs.addTab(self.analysis_tab, "Analysis")

        # Cell Density Heatmap Tab
        self.density_heatmap_tab = QWidget()
        density_heatmap_layout = QVBoxLayout(self.density_heatmap_tab)
        self.density_heatmap_canvas = MplCanvas(self, width=14, height=10)
        density_heatmap_layout.addWidget(self.density_heatmap_canvas)

        # Add the heatmap tab to the main tabs
        self.tabs.addTab(self.density_heatmap_tab, "Heatmap")

        # Dataset Analysis Tab
        self.dataset_tab = QWidget()
        dataset_layout = QVBoxLayout(self.dataset_tab)

        # Dataset analysis options
        dataset_options = QWidget()
        dataset_options_layout = QHBoxLayout(dataset_options)

        self.max_files_spin = QSpinBox()
        self.max_files_spin.setRange(1, 1000)
        self.max_files_spin.setValue(100)
        dataset_options_layout.addWidget(QLabel("Max Files per Folder:"))
        dataset_options_layout.addWidget(self.max_files_spin)

        dataset_options_layout.addStretch()

        dataset_layout.addWidget(dataset_options)

        # Dataset analysis canvas
        self.dataset_canvas = MplCanvas(self, width=10, height=6)
        dataset_layout.addWidget(self.dataset_canvas)

        # Dataset statistics text area
        self.dataset_stats_label = QLabel("Dataset Statistics:")
        dataset_layout.addWidget(self.dataset_stats_label)

        self.tabs.addTab(self.dataset_tab, "Dataset Analysis")

        # Tasks Tab
        self.tasks_tab = TasksTab()
        self.tabs.addTab(self.tasks_tab, "Tasks")

        # Advanced Features Tab
        self.advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_tab)

        # Create a scroll area for the advanced features
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Data Augmentation Group
        data_aug_group = QGroupBox("Data Augmentation")
        data_aug_layout = QVBoxLayout(data_aug_group)

        # Enable/disable data augmentation
        self.data_aug_checkbox = QCheckBox("Enable Data Augmentation")
        self.data_aug_checkbox.setChecked(self.data_augmentation_enabled)
        self.data_aug_checkbox.toggled.connect(self.toggle_data_augmentation)
        data_aug_layout.addWidget(self.data_aug_checkbox)

        # Data augmentation parameters
        aug_params_group = QGroupBox("Augmentation Parameters")
        aug_params_layout = QGridLayout(aug_params_group)

        # Rotation range
        aug_params_layout.addWidget(QLabel("Rotation Range:"), 0, 0)
        self.rotation_range_spin = QSpinBox()
        self.rotation_range_spin.setRange(0, 180)
        self.rotation_range_spin.setValue(self.augmentation_params['rotation_range'])
        self.rotation_range_spin.valueChanged.connect(lambda v: self.update_aug_param('rotation_range', v))
        aug_params_layout.addWidget(self.rotation_range_spin, 0, 1)

        # Width shift range
        aug_params_layout.addWidget(QLabel("Width Shift Range:"), 1, 0)
        self.width_shift_spin = QDoubleSpinBox()
        self.width_shift_spin.setRange(0, 1.0)
        self.width_shift_spin.setSingleStep(0.05)
        self.width_shift_spin.setValue(self.augmentation_params['width_shift_range'])
        self.width_shift_spin.valueChanged.connect(lambda v: self.update_aug_param('width_shift_range', v))
        aug_params_layout.addWidget(self.width_shift_spin, 1, 1)

        # Height shift range
        aug_params_layout.addWidget(QLabel("Height Shift Range:"), 2, 0)
        self.height_shift_spin = QDoubleSpinBox()
        self.height_shift_spin.setRange(0, 1.0)
        self.height_shift_spin.setSingleStep(0.05)
        self.height_shift_spin.setValue(self.augmentation_params['height_shift_range'])
        self.height_shift_spin.valueChanged.connect(lambda v: self.update_aug_param('height_shift_range', v))
        aug_params_layout.addWidget(self.height_shift_spin, 2, 1)

        # Zoom range
        aug_params_layout.addWidget(QLabel("Zoom Range:"), 3, 0)
        self.zoom_range_spin = QDoubleSpinBox()
        self.zoom_range_spin.setRange(0, 1.0)
        self.zoom_range_spin.setSingleStep(0.05)
        self.zoom_range_spin.setValue(self.augmentation_params['zoom_range'])
        self.zoom_range_spin.valueChanged.connect(lambda v: self.update_aug_param('zoom_range', v))
        aug_params_layout.addWidget(self.zoom_range_spin, 3, 1)

        # Horizontal flip
        self.horizontal_flip_check = QCheckBox("Horizontal Flip")
        self.horizontal_flip_check.setChecked(self.augmentation_params['horizontal_flip'])
        self.horizontal_flip_check.toggled.connect(lambda v: self.update_aug_param('horizontal_flip', v))
        aug_params_layout.addWidget(self.horizontal_flip_check, 4, 0, 1, 2)

        # Add augmentation parameters group to data augmentation group
        data_aug_layout.addWidget(aug_params_group)

        # Apply augmentation button
        self.apply_aug_button = QPushButton("Apply Augmentation to Current Image")
        self.apply_aug_button.clicked.connect(self.apply_augmentation)
        data_aug_layout.addWidget(self.apply_aug_button)

        # Add data augmentation group to scroll layout
        scroll_layout.addWidget(data_aug_group)

        # Active Learning Group
        active_learning_group = QGroupBox("Active Learning")
        active_learning_layout = QVBoxLayout(active_learning_group)

        # Enable/disable active learning
        self.active_learning_checkbox = QCheckBox("Enable Active Learning")
        self.active_learning_checkbox.setChecked(self.active_learning_enabled)
        self.active_learning_checkbox.toggled.connect(self.toggle_active_learning)
        active_learning_layout.addWidget(self.active_learning_checkbox)

        # Active learning parameters
        al_params_group = QGroupBox("Active Learning Parameters")
        al_params_layout = QGridLayout(al_params_group)

        # Number of samples per iteration
        al_params_layout.addWidget(QLabel("Samples per Iteration:"), 0, 0)
        self.samples_per_iter_spin = QSpinBox()
        self.samples_per_iter_spin.setRange(1, 50)
        self.samples_per_iter_spin.setValue(10)
        al_params_layout.addWidget(self.samples_per_iter_spin, 0, 1)

        # Number of iterations
        al_params_layout.addWidget(QLabel("Number of Iterations:"), 1, 0)
        self.num_iterations_spin = QSpinBox()
        self.num_iterations_spin.setRange(1, 20)
        self.num_iterations_spin.setValue(3)
        al_params_layout.addWidget(self.num_iterations_spin, 1, 1)

        # Fine-tuning epochs
        al_params_layout.addWidget(QLabel("Fine-tuning Epochs:"), 2, 0)
        self.fine_tune_epochs_spin = QSpinBox()
        self.fine_tune_epochs_spin.setRange(1, 50)
        self.fine_tune_epochs_spin.setValue(5)
        al_params_layout.addWidget(self.fine_tune_epochs_spin, 2, 1)

        # Add active learning parameters group to active learning group
        active_learning_layout.addWidget(al_params_group)

        # Start active learning button
        self.start_al_button = QPushButton("Start Active Learning")
        self.start_al_button.clicked.connect(self.start_active_learning)
        active_learning_layout.addWidget(self.start_al_button)

        # Add active learning group to scroll layout
        scroll_layout.addWidget(active_learning_group)

        # Edge AI Group
        edge_ai_group = QGroupBox("Edge AI Support")
        edge_ai_layout = QVBoxLayout(edge_ai_group)

        # Enable/disable edge AI
        self.edge_ai_checkbox = QCheckBox("Enable Edge AI (Model Quantization)")
        self.edge_ai_checkbox.setChecked(self.edge_ai_enabled)
        self.edge_ai_checkbox.toggled.connect(self.toggle_edge_ai)
        edge_ai_layout.addWidget(self.edge_ai_checkbox)

        # Edge AI parameters
        edge_ai_params_group = QGroupBox("Quantization Parameters")
        edge_ai_params_layout = QGridLayout(edge_ai_params_group)

        # Quantization type
        edge_ai_params_layout.addWidget(QLabel("Quantization Type:"), 0, 0)
        self.quant_type_combo = QComboBox()
        self.quant_type_combo.addItems(["Full Integer", "Float16", "Dynamic Range"])
        edge_ai_params_layout.addWidget(self.quant_type_combo, 0, 1)

        # Add edge AI parameters group to edge AI group
        edge_ai_layout.addWidget(edge_ai_params_group)

        # Quantize model button
        self.quantize_button = QPushButton("Quantize Model")
        self.quantize_button.clicked.connect(self.quantize_model)
        edge_ai_layout.addWidget(self.quantize_button)

        # Add edge AI group to scroll layout
        scroll_layout.addWidget(edge_ai_group)

        # Add stretch to push everything to the top
        scroll_layout.addStretch()

        # Set the scroll content and add to the layout
        scroll_area.setWidget(scroll_content)
        advanced_layout.addWidget(scroll_area)

        # Add the slide inference tab
        self.slide_inference_tab = SlideInferenceTab(self)
        self.tabs.addTab(self.slide_inference_tab, "Slide Inference")

        # Add the advanced features tab
        self.tabs.addTab(self.advanced_tab, "Advanced Features")

        # Add tabs to main layout
        self.main_layout.addWidget(self.tabs)

    def load_model(self):
        """Load the pre-trained segmentation and classification models."""
        try:
            # Initialize enhanced components if available
            if self.enhanced_mode:
                try:
                    self.preprocessor = ImagePreprocessor(
                        stain_matrix=np.array(self.config.get('stain_matrix')) if 'stain_matrix' in self.config else None
                    )

                    # Load models using enhanced loader if available
                    seg_model_path = self.config.get('seg_model_path', "segmentation_model.keras")
                    cls_model_path = self.config.get('cls_model_path', "classification_model.keras")

                    if os.path.exists(seg_model_path) and os.path.exists(cls_model_path):
                        self.segmentation_model, self.classification_model = load_models(
                            seg_model_path,
                            cls_model_path
                        )

                        # Initialize other enhanced components
                        self.cell_counter = CellCounter(
                            class_names=self.config.get('class_names', self.classification_class_names),
                            min_cell_size=self.config.get('min_cell_size', 50),
                            max_cell_size=self.config.get('max_cell_size', 1000)
                        )
                        self.analyzer = QuantitativeAnalyzer(
                            pixel_size=self.config.get('pixel_size', 0.25)
                        )
                        self.visualizer = ResultVisualizer(
                            class_colors=self.config.get('class_colors')
                        )

                        QMessageBox.information(
                            self, 
                            "Enhanced Mode Active", 
                            "Enhanced mode is active with advanced cell detection and analysis capabilities."
                        )
                        return

                except Exception as e:
                    print(f"Error initializing enhanced components: {e}")
                    # Fall back to standard mode
                    self.enhanced_mode = False
                    # Clear enhanced components
                    self.preprocessor = None
                    self.cell_counter = None
                    self.analyzer = None
                    self.visualizer = None
                    self.cell_details = None
                    self.analysis_results = None
                    if hasattr(self, 'enhanced_mode_radio'):
                        self.standard_mode_radio.setChecked(True)

            # Standard model loading (fallback)
            # Custom objects for model loading
            custom_objects = {
                'combined_loss': combined_loss,
                'MeanIoUCustom': MeanIoUCustom
            }

            # Check if segmentation model file exists
            seg_model_path = "segmentation_model.keras"
            if not os.path.exists(seg_model_path):
                # Try fallback to old model name
                seg_model_path = "unet_model_v3.5.keras"
                if not os.path.exists(seg_model_path):
                    QMessageBox.warning(
                        self, 
                        "Segmentation Model Not Found", 
                        f"Segmentation model file not found. Please place the model file in the application directory."
                    )
                    return

            # Load the segmentation model
            self.segmentation_model = load_trained_model(
                seg_model_path,
                custom_objects=custom_objects,
                model_type='segmentation'
            )

            # Load the classification model
            cls_model_path = "classification_model.keras"
            self.classification_model = load_trained_model(
                cls_model_path,
                custom_objects=custom_objects,
                model_type='classification'
            )

            # Show success message
            QMessageBox.information(
                self, 
                "Models Loaded", 
                "Both segmentation and classification models loaded successfully."
            )

        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error Loading Models", 
                f"Failed to load models: {str(e)}"
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
                # Load the image
                self.image = cv2.imread(file_path)
                if self.image is None:
                    raise ValueError("Failed to load image file")

                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                self.image_path = file_path  # Store the path for reference

                # Display the image
                self.display_image(self.original_canvas, self.image, "Original Image")

                # Enable preprocessing button
                self.preprocess_button.setEnabled(True)

                # Switch to original image tab
                self.tabs.setCurrentIndex(0)

            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "Error Loading Image", 
                    f"Failed to load image: {str(e)}"
                )

    def display_image(self, canvas, image, title):
        """Helper method to display images on canvas"""
        canvas.axes.clear()

        # Normalize image to 0-1 range for display
        if image.max() > 1.0:
            image = image / 255.0

        if len(image.shape) == 2:  # Grayscale
            canvas.axes.imshow(image, cmap='gray')
        else:  # Color
            canvas.axes.imshow(image)

        canvas.axes.set_title(title)
        canvas.axes.axis('off')
        canvas.draw()

    def preprocess(self):
        """Preprocess the loaded image."""
        if self.image is None:
            return

        try:
            # Create a task for preprocessing
            task_id = self.tasks_tab.add_task("Image Preprocessing", {
                "image_path": getattr(self, "image_path", "Unknown"),
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            self.current_preprocess_task_id = task_id

            # Update task status
            self.tasks_tab.update_task(task_id, "In Progress", progress=10, 
                                      status_message="Starting preprocessing...")

            # Always use enhanced mode if available
            use_enhanced = self.enhanced_mode

            if use_enhanced and self.preprocessor:
                # Clear standard preprocessing variables
                self.pred_mask = None
                self.cell_details = None
                self.analysis_results = None

                # Get enhanced preprocessing options
                stain_separation = self.stain_separation_checkbox.isChecked()
                adaptive_gamma = self.adaptive_gamma_checkbox.isChecked()
                normalize_brightness = self.normalize_brightness_checkbox.isChecked()

                # Show processing message
                QMessageBox.information(
                    self,
                    "Processing",
                    "Enhanced preprocessing in progress. This may take a moment..."
                )

                # Convert image path to file if needed
                if isinstance(self.image, np.ndarray):
                    # Save temporary file
                    temp_path = "temp_image.png"
                    cv2.imwrite(temp_path, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
                    self.h_channel, self.enhanced_image, self.applied_steps = self.preprocessor.preprocess_pipeline(temp_path)
                    os.remove(temp_path)  # Clean up
                else:
                    self.h_channel, self.enhanced_image, self.applied_steps = self.preprocessor.preprocess_pipeline(self.image)

                # Use enhanced image as preprocessed image
                self.preprocessed_image = self.enhanced_image

                # Update the applied steps label
                steps_html = "<ul>"
                for step in self.applied_steps:
                    steps_html += f"<li>{step}</li>"
                steps_html += "</ul>"
                self.applied_steps_label.setText(steps_html)
                self.applied_steps_label.setTextFormat(Qt.RichText)

                # Display the preprocessed image (show both h-channel and enhanced)
                self.preprocessed_canvas.fig.clf()
                gs = self.preprocessed_canvas.fig.add_gridspec(1, 2)

                ax1 = self.preprocessed_canvas.fig.add_subplot(gs[0, 0])
                ax1.imshow(self.h_channel, cmap='gray')
                ax1.set_title("H-Channel (Nuclei)")
                ax1.axis('off')

                ax2 = self.preprocessed_canvas.fig.add_subplot(gs[0, 1])
                ax2.imshow(self.enhanced_image)
                ax2.set_title("Enhanced Image")
                ax2.axis('off')

                self.preprocessed_canvas.fig.tight_layout()
                self.preprocessed_canvas.draw()

            else:
                # Standard preprocessing
                # Clear enhanced preprocessing variables
                self.h_channel = None
                self.enhanced_image = None
                self.cell_details = None
                self.analysis_results = None

                # Get preprocessing options
                enhance = self.enhance_checkbox.isChecked()
                denoise = self.denoise_checkbox.isChecked()
                detect_regions = self.roi_checkbox.isChecked()

                # Create a list of applied steps for standard preprocessing
                self.applied_steps = ["Standard Preprocessing"]
                if enhance:
                    self.applied_steps.append("Enhance Contrast")
                if denoise:
                    self.applied_steps.append("Denoise")
                if detect_regions:
                    self.applied_steps.append("ROI Detection")

                # Update the applied steps label
                steps_html = "<ul>"
                for step in self.applied_steps:
                    steps_html += f"<li>{step}</li>"
                steps_html += "</ul>"
                self.applied_steps_label.setText(steps_html)
                self.applied_steps_label.setTextFormat(Qt.RichText)

                # Preprocess the image
                self.preprocessed_image, roi_mask = preprocess_image(
                    self.image, 
                    enhance=enhance, 
                    denoise=denoise, 
                    detect_regions=detect_regions
                )

                # Display the preprocessed image
                self.preprocessed_canvas.axes.clear()

                # Normalize image to 0-1 range for display
                normalized_image = self.preprocessed_image.copy()
                if normalized_image.max() > 1.0:
                    normalized_image = normalized_image / 255.0

                self.preprocessed_canvas.axes.imshow(normalized_image)
                self.preprocessed_canvas.axes.set_title("Preprocessed Image")
                self.preprocessed_canvas.axes.axis('off')
                self.preprocessed_canvas.draw()

            # Enable analyze button
            self.analyze_button.setEnabled(True)

            # Switch to preprocessed image tab
            self.tabs.setCurrentIndex(1)

            # Update task status
            self.tasks_tab.update_task(self.current_preprocess_task_id, "Completed", progress=100, 
                                      status_message="Preprocessing completed successfully",
                                      results={
                                          "Image Size": f"{self.preprocessed_image.shape[1]}x{self.preprocessed_image.shape[0]}",
                                          "Preprocessing Applied Steps": getattr(self, 'applied_steps', []) if use_enhanced else [
                                              "Standard Preprocessing",
                                              "Enhance Contrast" if self.enhance_checkbox.isChecked() else None,
                                              "Denoise" if self.denoise_checkbox.isChecked() else None,
                                              "ROI Detection" if self.roi_checkbox.isChecked() else None
                                          ]
                                      })

        except Exception as e:
            # Update task status to failed
            self.tasks_tab.update_task(self.current_preprocess_task_id, "Failed", 
                                      status_message=f"Error: {str(e)}",
                                      results={"Error": str(e), "Traceback": traceback.format_exc()})

            QMessageBox.critical(
                self, 
                "Error Preprocessing", 
                f"Failed to preprocess image: {str(e)}\n{traceback.format_exc()}"
            )

    # Method removed as model selection is disabled

    def analyze(self):
        """Analyze the preprocessed image."""
        if self.preprocessed_image is None or self.segmentation_model is None:
            return

        try:
            # Always use the fixed model for analysis (segmentation_model.keras)
            temp_model = None

            if os.path.exists(self.fixed_analysis_model_path):
                try:
                    # Load the fixed model for this analysis
                    from models.inference import load_trained_model

                    # Custom objects for model loading
                    custom_objects = {
                        'combined_loss': combined_loss,
                        'MeanIoUCustom': MeanIoUCustom
                    }

                    # Try to load the model
                    temp_model = load_trained_model(
                        self.fixed_analysis_model_path,
                        custom_objects=custom_objects
                    )

                    # Show a message that we're using the fixed model for this analysis
                    print(f"Using fixed model {self.fixed_analysis_model_path} for analysis")

                except Exception as model_error:
                    # If there's an error loading the model, show a message and use the default model
                    print(f"Error loading fixed model {self.fixed_analysis_model_path}: {model_error}")
                    QMessageBox.warning(
                        self,
                        "Model Loading Error",
                        f"Failed to load fixed model {self.fixed_analysis_model_path}. Using default model instead.\nError: {str(model_error)}"
                    )

            # Initialize classification result
            classification_result = None

            # Run classification prediction first if classification model is available
            # Use the temporary model if one was loaded, otherwise use the default classification model
            classification_model_to_use = temp_model if temp_model is not None else self.classification_model

            if classification_model_to_use is not None:
                try:
                    # Get classification results
                    classification_result = get_predicted_class(
                        classification_model_to_use,
                        self.preprocessed_image
                    )

                    # Store classification results
                    if isinstance(classification_result, dict):
                        self.predicted_class = classification_result.get('predicted_class', 1)
                        self.class_probability = classification_result.get('probability', 1.0)

                        # Store full class probabilities if available
                        if 'probabilities' in classification_result:
                            self.class_probabilities = classification_result['probabilities']
                        else:
                            # Create an array with the confidence for the predicted class
                            self.class_probabilities = np.zeros(len(self.class_names))
                            self.class_probabilities[self.predicted_class] = self.class_probability
                    else:
                        # Default values if classification_result is not a dictionary
                        self.predicted_class = 1
                        self.class_probability = 1.0
                        self.class_probabilities = np.zeros(len(self.class_names))
                        self.class_probabilities[self.predicted_class] = self.class_probability
                except Exception as classification_error:
                    print(f"Classification error: {classification_error}")
                    # Set default values if classification fails
                    self.predicted_class = 1
                    self.class_probability = 1.0
                    self.class_probabilities = np.zeros(len(self.class_names))
                    self.class_probabilities[self.predicted_class] = self.class_probability
            else:
                # Set default values if no classification model
                self.predicted_class = 1
                self.class_probability = 1.0
                self.class_probabilities = np.zeros(len(self.class_names))
                self.class_probabilities[self.predicted_class] = self.class_probability

            # Run segmentation prediction with classification result
            # Use the temporary model if one was loaded, otherwise use the default segmentation model
            segmentation_model_to_use = temp_model if temp_model is not None else self.segmentation_model

            self.pred_mask = predict_segmentation(
                segmentation_model_to_use, 
                self.preprocessed_image,
                classification_result=classification_result
            )

            # Count cells
            counts = count_cells(self.pred_mask, self.class_names, classification_result)

            # Generate results for the results tab
            self.generate_results()

            # Display results in analysis tab
            self.display_analysis_results()

            # Switch to analysis tab
            self.tabs.setCurrentIndex(2)

        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error Analyzing", 
                f"Failed to analyze image: {str(e)}"
            )

    def display_analysis_results(self):
        """Display all analysis results in the GUI"""
        # Clear previous results
        self.analysis_canvas.fig.clf()

        # Create 2x2 grid for visualizations
        gs = self.analysis_canvas.fig.add_gridspec(2, 2)

        # Original Image
        ax1 = self.analysis_canvas.fig.add_subplot(gs[0, 0])
        self.display_image_on_axis(ax1, self.image, "Original Image")

        # Preprocessed Image
        ax2 = self.analysis_canvas.fig.add_subplot(gs[0, 1])
        self.display_image_on_axis(ax2, self.preprocessed_image, "Preprocessed Image")

        # Segmentation Mask
        ax3 = self.analysis_canvas.fig.add_subplot(gs[1, 0])
        if self.pred_mask is not None:
            cmap = ListedColormap(['black', 'red', 'green', 'blue', 'yellow', 'purple'])
            ax3.imshow(self.pred_mask, cmap=cmap, vmin=0, vmax=5)
            ax3.set_title("Segmentation Mask")
            ax3.axis('off')

        # Cell Detection
        ax4 = self.analysis_canvas.fig.add_subplot(gs[1, 1])
        if self.pred_mask is not None:
            overlay = self.image.copy()
            for class_idx in range(1, len(self.class_names)):
                mask = self.pred_mask == class_idx
                overlay[mask] = [255, 0, 0]  # Red for detected cells

            ax4.imshow(overlay)
            ax4.set_title("Cell Detection")
            ax4.axis('off')

        self.analysis_canvas.fig.tight_layout()
        self.analysis_canvas.draw()

    def display_image_on_axis(self, ax, image, title):
        """Display image on a matplotlib axis"""
        ax.clear()
        if image is not None:
            if image.max() > 1.0:
                image = image / 255.0
            if len(image.shape) == 2:
                ax.imshow(image, cmap='gray')
            else:
                ax.imshow(image)
        else:
            ax.text(0.5, 0.5, "Image not available", ha='center', va='center')
        ax.set_title(title)
        ax.axis('off')

    def generate_results(self):
        """Generate and display results in separate result tabs."""
        if self.preprocessed_image is None or self.pred_mask is None:
            return

        try:
            # Initialize cell type label
            cell_type = "All Classes"

            # ---------- Helper Functions ----------
            def plot_fallback_heatmap(ax, message, fig=None):
                heatmap_data = np.random.rand(10, 10) * 0.56
                im = ax.imshow(heatmap_data, cmap='viridis')
                cbar = fig.colorbar(im, ax=ax) if fig else self.density_heatmap_canvas.fig.colorbar(im, ax=ax)
                cbar.set_label('Cell Density')
                ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=10, color='white')


            def get_detected_cells():
                nonlocal cell_type
                import matplotlib.pyplot as plt

                # Initialize empty list for cells
                all_cells = []

                if self.enhanced_mode and hasattr(self, 'analysis_results') and self.analysis_results is not None:
                    # Use cell details from analysis results if available
                    cell_details = getattr(self, 'cell_details', [])
                    if cell_details:
                        return cell_details

                # If no enhanced analysis results, use standard detection
                # Create a proper classification result with all available information
                classification_result = {
                    'predicted_class': self.predicted_class,
                    'probability': getattr(self, 'class_probability', 1.0),
                    'probabilities': getattr(self, 'class_probabilities', None)
                } if self.predicted_class is not None else None

                # Use summarize_prediction to get cell features with proper classification
                try:
                    from analysis.cell_detection2 import summarize_prediction
                    all_features, class_summary_data = summarize_prediction(
                        self.preprocessed_image, 
                        self.pred_mask, 
                        class_names=self.class_names, 
                        classification_result=classification_result,
                        classification_model=getattr(self, 'classification_model', None)
                    )

                    # Convert cell features to the format expected by plot_density_heatmap
                    for feature in all_features:
                        if 'BBox' in feature:
                            x, y, w, h = feature['BBox']
                            # Create a simple contour from the bounding box
                            contour = np.array([
                                [y, x],
                                [y, x+w],
                                [y+h, x+w],
                                [y+h, x]
                            ])

                            # Use PredictedClass if available, otherwise use the original Class
                            class_id = feature.get('PredictedClass', feature.get('Class', 1))

                            all_cells.append({
                                'contour': contour,
                                'centroid': (y + h//2, x + w//2),
                                'class': class_id,
                                'class_name': feature.get('PredictedClassName', self.class_names[class_id])
                            })

                    cell_type = "All Classes"
                except Exception as e:
                    print(f"Error detecting cells: {e}")

                return all_cells

                if self.enhanced_mode and hasattr(self, 'analysis_results') and self.analysis_results is not None:
                    counts = self.analysis_results.get('counts', {})
                    cell_details = getattr(self, 'cell_details', [])

                    if cell_details:
                        self.class_colors = self.config.get('class_colors', self.DEFAULT_CLASS_COLORS)
                        plot_cells_on_image(self, self.preprocessed_image, cell_details)

                        # Convert figure to image using a more reliable method
                        import io
                        from matplotlib.backends.backend_agg import FigureCanvasAgg
                        current_fig = plt.gcf()
                        canvas = FigureCanvasAgg(current_fig)
                        canvas.draw()
                        buf = io.BytesIO()
                        canvas.print_png(buf)
                        buf.seek(0)
                        import PIL.Image
                        img = np.array(PIL.Image.open(buf))

                        # Display the image on the provided axis
                        ax.imshow(img)
                        plt.close()
                        ax.text(0.02, 0.02, f"Total cells: {total_count}\n{cell_summary}",
                                transform=ax.transAxes, fontsize=10, color='white',
                                bbox=dict(facecolor='black', alpha=0.7))
                    else:
                        ax.text(0.5, 0.5, f"No cell details\n\n{cell_summary}",
                                ha='center', va='center', fontsize=12)
                else:
                    # Create a proper classification result with all available information
                    classification_result = {
                        'predicted_class': self.predicted_class,
                        'probability': getattr(self, 'class_probability', 1.0),
                        'probabilities': getattr(self, 'class_probabilities', None)
                    } if self.predicted_class is not None else None

                    # Use summarize_prediction to get cell features with proper classification
                    # This ensures individual cells are classified correctly
                    from analysis.cell_detection2 import summarize_prediction
                    all_features, class_summary_data = summarize_prediction(
                        self.preprocessed_image, 
                        self.pred_mask, 
                        class_names=self.class_names, 
                        classification_result=classification_result,
                        classification_model=getattr(self, 'classification_model', None)
                    )

                    # Convert cell features to the format expected by plot_cells_on_image
                    all_cells = []
                    for feature in all_features:
                        if 'BBox' in feature:
                            x, y, w, h = feature['BBox']
                            # Create a simple contour from the bounding box
                            contour = np.array([
                                [y, x],
                                [y, x+w],
                                [y+h, x+w],
                                [y+h, x]
                            ])

                            # Use PredictedClass if available, otherwise use the original Class
                            class_id = feature.get('PredictedClass', feature.get('Class', 1))

                            all_cells.append({
                                'contour': contour,
                                'centroid': (y + h//2, x + w//2),
                                'class': class_id,
                                'class_name': feature.get('PredictedClassName', self.class_names[class_id])
                            })

                    # Get counts from class summary
                    counts = {summary['ClassName']: summary['Cell Count'] for summary in class_summary_data}
                    total_count = sum(counts.values())
                    cell_summary = "\n".join([f"{name}: {counts.get(name, 0)}" for name in self.class_names[1:]])

                    if all_cells:
                        self.class_colors = self.DEFAULT_CLASS_COLORS
                        plot_cells_on_image(self, self.preprocessed_image, all_cells)

                        # Convert figure to image using a more reliable method
                        import io
                        from matplotlib.backends.backend_agg import FigureCanvasAgg
                        current_fig = plt.gcf()
                        canvas = FigureCanvasAgg(current_fig)
                        canvas.draw()
                        buf = io.BytesIO()
                        canvas.print_png(buf)
                        buf.seek(0)
                        import PIL.Image
                        img = np.array(PIL.Image.open(buf))

                        # Display the image on the provided axis
                        ax.imshow(img)
                        plt.close()
                        ax.text(0.02, 0.02, f"Total cells: {total_count}\n{cell_summary}",
                                transform=ax.transAxes, fontsize=10, color='white',
                                bbox=dict(facecolor='black', alpha=0.7))
                    else:
                        ax.text(0.5, 0.5, f"No cells detected\n\n{cell_summary}",
                                ha='center', va='center', fontsize=12)
                    cell_type = "All Classes"

                ax.set_title("Cell Detection")
                ax.axis('off')


                return all_cells if 'all_cells' in locals() else []

            def plot_density_heatmap(cells):
                # Clear the canvas
                self.density_heatmap_canvas.fig.clf()
                ax = self.density_heatmap_canvas.fig.add_subplot(111)

                if cells:
                    from visualization.visualization import plot_density_heatmap as viz_plot_density_heatmap
                    viz_plot_density_heatmap(self, self.preprocessed_image, cells)

                    # Convert figure to image using a more reliable method
                    import io
                    from matplotlib.backends.backend_agg import FigureCanvasAgg
                    current_fig = plt.gcf()
                    canvas = FigureCanvasAgg(current_fig)
                    canvas.draw()
                    buf = io.BytesIO()
                    canvas.print_png(buf)
                    buf.seek(0)
                    import PIL.Image
                    img = np.array(PIL.Image.open(buf))

                    # Display the image on the provided axis
                    ax.imshow(img)
                    plt.close()
                else:
                    plot_fallback_heatmap(ax, "No cells for density heatmap", self.density_heatmap_canvas.fig)
                ax.set_title(f"Cell Density Heatmap ({cell_type})")
                ax.axis('off')

                # Update the canvas
                self.density_heatmap_canvas.fig.tight_layout()
                self.density_heatmap_canvas.draw()

            # ---------- Generate Results for Each Tab ----------
            # Get detected cells for the heatmap
            detected_cells = get_detected_cells()

            # Generate density heatmap plot
            plot_density_heatmap(detected_cells)

            # Switch to the Heatmap tab to show the results
            self.tabs.setCurrentIndex(self.tabs.indexOf(self.density_heatmap_tab))

        except Exception as e:
            import traceback
            print(f"Error generating results: {e}")
            traceback.print_exc()

    def toggle_data_augmentation(self, enabled):
        """Enable or disable data augmentation."""
        self.data_augmentation_enabled = enabled

    def update_aug_param(self, param_name, value):
        """Update a data augmentation parameter."""
        self.augmentation_params[param_name] = value

    def apply_augmentation(self):
        """Apply data augmentation to the current image."""
        if self.image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        try:
            # Import required modules
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            import matplotlib.pyplot as plt

            # Create a data generator with the current parameters
            datagen = ImageDataGenerator(
                rotation_range=self.augmentation_params['rotation_range'],
                width_shift_range=self.augmentation_params['width_shift_range'],
                height_shift_range=self.augmentation_params['height_shift_range'],
                zoom_range=self.augmentation_params['zoom_range'],
                horizontal_flip=self.augmentation_params['horizontal_flip'],
                fill_mode='nearest'
            )

            # Prepare the image for augmentation (add batch dimension)
            img = np.expand_dims(self.image, 0)

            # Generate an augmented image
            aug_iter = datagen.flow(img, batch_size=1)
            aug_img = next(aug_iter)[0].astype(np.uint8)

            # Display the augmented image
            self.preprocessed_image = aug_img
            self.display_image(self.preprocessed_canvas, aug_img, "Augmented Image")

            # Switch to the preprocessed tab
            self.tabs.setCurrentIndex(1)

            QMessageBox.information(
                self, 
                "Augmentation Applied", 
                "Data augmentation has been applied to the image. You can now analyze the augmented image."
            )

        except Exception as e:
            QMessageBox.critical(
                self, 
                "Augmentation Error", 
                f"Failed to apply augmentation: {str(e)}"
            )

    def toggle_active_learning(self, enabled):
        """Enable or disable active learning."""
        self.active_learning_enabled = enabled

    def start_active_learning(self):
        """Start the active learning process."""
        if self.segmentation_model is None or self.classification_model is None:
            QMessageBox.warning(
                self, 
                "Models Not Loaded", 
                "Please ensure both segmentation and classification models are loaded."
            )
            return

        try:
            # Import required modules
            from models.active_learning import active_learning_loop
            import tensorflow as tf
            import numpy as np
            import os

            # Create a directory for active learning results
            output_dir = "active_learning_results"
            os.makedirs(output_dir, exist_ok=True)

            # Get parameters from UI
            samples_per_iteration = self.samples_per_iter_spin.value()
            num_iterations = self.num_iterations_spin.value()
            fine_tune_epochs = self.fine_tune_epochs_spin.value()

            # Show a message that this is a simulation
            QMessageBox.information(
                self,
                "Active Learning Simulation",
                "This is a simulation of the active learning process. In a real-world scenario, "
                "you would interact with pathologists to annotate uncertain samples. "
                "For this simulation, we'll generate random annotations."
            )

            # Create a progress dialog
            from PyQt5.QtWidgets import QProgressDialog
            progress = QProgressDialog("Running active learning...", "Cancel", 0, num_iterations, self)
            progress.setWindowTitle("Active Learning Progress")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            # Simulate active learning with a small dataset
            # In a real application, you would use a larger dataset and real expert annotations

            # Create a small labeled dataset (use the current image if available)
            if self.image is not None:
                # Resize image to match model input
                input_shape = self.segmentation_model.input_shape[1:3]
                img = cv2.resize(self.image, input_shape)
                img = img / 255.0  # Normalize

                # Create a small labeled dataset with just this image
                X_labeled = np.expand_dims(img, 0)

                # Create a random mask for demonstration
                num_classes = len(self.class_names)
                mask = np.zeros((input_shape[0], input_shape[1], num_classes))
                random_class = np.random.randint(0, num_classes, size=(input_shape[0], input_shape[1]))
                for c in range(num_classes):
                    mask[:, :, c] = (random_class == c).astype(np.float32)
                y_labeled = np.expand_dims(mask, 0)

                # Create a small unlabeled pool (duplicate the image with slight modifications)
                X_unlabeled = np.zeros((10, input_shape[0], input_shape[1], 3))
                for i in range(10):
                    # Add some noise to create "different" images
                    X_unlabeled[i] = img + np.random.normal(0, 0.1, img.shape)
                    X_unlabeled[i] = np.clip(X_unlabeled[i], 0, 1)  # Ensure values are in [0, 1]

                # Run active learning loop
                for i in range(num_iterations):
                    progress.setValue(i)
                    if progress.wasCanceled():
                        break

                    # In a real implementation, this would be a call to active_learning_loop
                    # For this simulation, we'll just wait a bit to simulate processing
                    import time
                    time.sleep(1)

                    # Update progress
                    progress.setLabelText(f"Iteration {i+1}/{num_iterations}: Selecting uncertain samples...")

                progress.setValue(num_iterations)

                # Show results
                QMessageBox.information(
                    self,
                    "Active Learning Completed",
                    f"Active learning simulation completed with {num_iterations} iterations.\n\n"
                    f"In a real-world scenario, the model would be fine-tuned with expert annotations "
                    f"and performance would improve over iterations."
                )
            else:
                QMessageBox.warning(
                    self,
                    "No Image",
                    "Please load an image first to demonstrate active learning."
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Active Learning Error",
                f"Failed to run active learning: {str(e)}"
            )

    def toggle_edge_ai(self, enabled):
        """Enable or disable edge AI support."""
        self.edge_ai_enabled = enabled

    def quantize_model(self):
        """Quantize the model for edge deployment."""
        if self.segmentation_model is None and self.classification_model is None:
            QMessageBox.warning(
                self,
                "No Models Loaded",
                "Please load models before quantization."
            )
            return

        try:
            # Import required modules
            import tensorflow as tf
            import os

            # Create a directory for quantized models
            output_dir = "quantized_models"
            os.makedirs(output_dir, exist_ok=True)

            # Get quantization type
            quant_type = self.quant_type_combo.currentText()

            # Create a progress dialog
            from PyQt5.QtWidgets import QProgressDialog
            progress = QProgressDialog("Quantizing model...", "Cancel", 0, 100, self)
            progress.setWindowTitle("Quantization Progress")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            # Update progress
            progress.setValue(10)
            progress.setLabelText("Preparing for quantization...")

            # Choose which model to quantize
            model_to_quantize = None
            model_name = ""
            if self.segmentation_model is not None and self.classification_model is not None:
                # Ask user which model to quantize
                from PyQt5.QtWidgets import QMessageBox
                choice = QMessageBox.question(
                    self,
                    "Choose Model",
                    "Which model would you like to quantize?",
                    QMessageBox.Cancel | QMessageBox.No | QMessageBox.Yes,
                    QMessageBox.Yes
                )

                if choice == QMessageBox.Yes:
                    model_to_quantize = self.segmentation_model
                    model_name = "segmentation"
                elif choice == QMessageBox.No:
                    model_to_quantize = self.classification_model
                    model_name = "classification"
                else:
                    progress.cancel()
                    return
            elif self.segmentation_model is not None:
                model_to_quantize = self.segmentation_model
                model_name = "segmentation"
            elif self.classification_model is not None:
                model_to_quantize = self.classification_model
                model_name = "classification"

            if model_to_quantize is None:
                progress.cancel()
                QMessageBox.warning(
                    self,
                    "No Model Selected",
                    "No model was selected for quantization."
                )
                return

            # Update progress
            progress.setValue(20)
            progress.setLabelText(f"Quantizing {model_name} model...")

            # Define a representative dataset generator
            def representative_dataset():
                # Generate random data matching the model's input shape
                input_shape = model_to_quantize.input_shape
                for _ in range(100):
                    data = np.random.random((1,) + input_shape[1:])
                    yield [data.astype(np.float32)]

            # Convert the model to TensorFlow Lite format
            converter = tf.lite.TFLiteConverter.from_keras_model(model_to_quantize)

            # Apply quantization based on selected type
            if quant_type == "Full Integer":
                # Full integer quantization
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            elif quant_type == "Float16":
                # Float16 quantization
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            elif quant_type == "Dynamic Range":
                # Dynamic range quantization
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # Update progress
            progress.setValue(50)
            progress.setLabelText("Converting model...")

            # Convert the model
            quantized_tflite_model = converter.convert()

            # Update progress
            progress.setValue(80)
            progress.setLabelText("Saving quantized model...")

            # Save the quantized model
            quantized_model_path = os.path.join(output_dir, f"quantized_{model_name}_model.tflite")
            with open(quantized_model_path, 'wb') as f:
                f.write(quantized_tflite_model)

            # Calculate model size reduction
            original_size = os.path.getsize(f"{model_name}_model.keras") if os.path.exists(f"{model_name}_model.keras") else 1000000
            quantized_size = os.path.getsize(quantized_model_path)
            size_reduction = (1 - quantized_size / original_size) * 100

            # Update progress
            progress.setValue(100)

            # Store the quantized model
            self.quantized_model = quantized_model_path

            # Show results
            QMessageBox.information(
                self,
                "Quantization Completed",
                f"Model quantized successfully and saved to {quantized_model_path}\n\n"
                f"Original size: {original_size / 1024:.2f} KB\n"
                f"Quantized size: {quantized_size / 1024:.2f} KB\n"
                f"Size reduction: {size_reduction:.2f}%"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Quantization Error",
                f"Failed to quantize model: {str(e)}"
            )

    def update_analysis_view(self):
        """Update the analysis view based on the selected options."""
        import matplotlib as mpl
        if self.preprocessed_image is None:
            return
        import matplotlib.pyplot as plt

        try:
            # Always use enhanced mode if available
            use_enhanced = self.enhanced_mode

            # Get selected class and view
            view_type = self.view_combo.currentText()

            # Always use the predicted class from the classification model if available
            if hasattr(self, 'predicted_class') and self.predicted_class is not None:
                class_idx = self.predicted_class
            else:
                # Default to class 1 if no predicted class is available
                class_idx = 1

            # Ensure class_idx is within valid range
            if class_idx >= len(self.class_names):
                print(f"Warning: Predicted class index {class_idx} is out of range. Using default class index 1.")
                class_idx = 1  # Use a safe default

            # Check if we need to preserve results when switching views
            preserve_results = False
            if hasattr(self, 'current_view_type') and self.current_view_type != view_type:
                # We're switching views, so we should preserve results
                preserve_results = True
                print(f"Switching from {self.current_view_type} to {view_type}")

            # Store the current view type to track changes
            if not hasattr(self, 'current_view_type'):
                self.current_view_type = view_type

            # Clear the canvas
            self.analysis_canvas.fig.clf()

            # Handle All Images view
            if view_type == "All Images":
                # Check if we're switching from ROI Detection view and need to preserve results
                if preserve_results and hasattr(self, 'roi_results') and self.roi_results:
                    # We have results from ROI Detection view that we should preserve
                    print("Preserving ROI Detection results while showing All Images")

                # Check if we already have All Images results that we can reuse
                if hasattr(self, 'all_images_results') and self.all_images_results and not preserve_results:
                    # Reuse existing All Images results
                    print("Reusing existing All Images results")

                    # Create a 2x2 grid for the four images
                    gs = self.analysis_canvas.fig.add_gridspec(2, 2)

                    # Use stored values
                    class_idx = self.all_images_results.get('class_idx', 1)
                    stored_pred_mask = self.all_images_results.get('pred_mask')
                    stored_original_image = self.all_images_results.get('original_image')
                    stored_preprocessed_image = self.all_images_results.get('preprocessed_image')

                    # Original Image
                    ax1 = self.analysis_canvas.fig.add_subplot(gs[0, 0])
                    if stored_original_image is not None:
                        # Normalize image to 0-1 range for display
                        normalized_image = stored_original_image.copy()
                        if normalized_image.max() > 1.0:
                            normalized_image = normalized_image / 255.0

                        if stored_original_image.shape[-1] == 1:
                            ax1.imshow(normalized_image.squeeze(), cmap='gray')
                        else:
                            ax1.imshow(normalized_image)
                    else:
                        ax1.text(0.5, 0.5, "No original image", ha='center', va='center')
                    ax1.set_title("Original Image")
                    ax1.axis('off')

                    # Preprocessed Image
                    ax2 = self.analysis_canvas.fig.add_subplot(gs[0, 1])
                    if stored_preprocessed_image is not None:
                        # Normalize image to 0-1 range for display
                        normalized_image = stored_preprocessed_image.copy()
                        if normalized_image.max() > 1.0:
                            normalized_image = normalized_image / 255.0

                        if stored_preprocessed_image.shape[-1] == 1:
                            ax2.imshow(normalized_image.squeeze(), cmap='gray')
                        else:
                            ax2.imshow(normalized_image)
                    else:
                        ax2.text(0.5, 0.5, "No preprocessed image", ha='center', va='center')
                    ax2.set_title("Preprocessed Image")
                    ax2.axis('off')

                    # Segmentation Mask
                    ax3 = self.analysis_canvas.fig.add_subplot(gs[1, 0])
                    if stored_pred_mask is not None:
                        # Create a colormap for the mask
                        from matplotlib.colors import ListedColormap
                        # Generate colors for each class
                        import matplotlib.pyplot as plt
                        colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))
                        cmap = ListedColormap(colors)

                        # Display the mask
                        im = ax3.imshow(stored_pred_mask, cmap=cmap, vmin=0, vmax=len(self.class_names)-1)

                        # Add a colorbar
                        from mpl_toolkits.axes_grid1 import make_axes_locatable
                        divider = make_axes_locatable(ax3)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        cbar = self.analysis_canvas.fig.colorbar(im, cax=cax)

                        # Set colorbar ticks and labels
                        cbar.set_ticks(np.arange(len(self.class_names)) + 0.5)
                        cbar.set_ticklabels(self.class_names)
                    else:
                        ax3.text(0.5, 0.5, "No segmentation mask", ha='center', va='center')
                    ax3.set_title("Segmentation Mask")
                    ax3.axis('off')

                    # Enhanced Detection (if available)
                    ax4 = self.analysis_canvas.fig.add_subplot(gs[1, 1])
                    if use_enhanced and hasattr(self, 'cell_details') and self.cell_details and hasattr(self, 'enhanced_image') and self.enhanced_image is not None:
                        # Use the enhanced visualizer
                        overlay = self.visualizer.overlay_heatmap(
                            self.enhanced_image,
                            stored_pred_mask
                        )
                        ax4.imshow(overlay)
                        ax4.set_title("Enhanced Detection")
                    elif stored_pred_mask is not None and stored_original_image is not None:
                        # Create a binary mask for the selected class
                        binary_mask = (stored_pred_mask == class_idx).astype(np.uint8)
                        # Create a simple overlay
                        overlay = np.zeros_like(stored_original_image)
                        overlay[binary_mask > 0] = [255, 0, 0]  # Red for detected cells
                        # Blend with original image
                        alpha = 0.3
                        blended = cv2.addWeighted(stored_original_image, 1-alpha, overlay, alpha, 0)
                        ax4.imshow(blended)

                        # Safely get class name
                        class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"Unknown (Class {class_idx})"
                        ax4.set_title(f"Cell Detection - {class_name}")
                    else:
                        ax4.text(0.5, 0.5, "No detection data available", ha='center', va='center')
                        ax4.set_title("Cell Detection")
                    ax4.axis('off')

                else:
                    # Calculate new results
                    # Create a 2x2 grid for the four images
                    gs = self.analysis_canvas.fig.add_gridspec(2, 2)

                    # Original Image
                    ax1 = self.analysis_canvas.fig.add_subplot(gs[0, 0])
                    if hasattr(self, 'original_image'):
                        # Normalize image to 0-1 range for display
                        normalized_image = self.original_image.copy()
                        if normalized_image.max() > 1.0:
                            normalized_image = normalized_image / 255.0

                        if self.original_image.shape[-1] == 1:
                            ax1.imshow(normalized_image.squeeze(), cmap='gray')
                        else:
                            ax1.imshow(normalized_image)
                    else:
                        ax1.text(0.5, 0.5, "No original image", ha='center', va='center')
                    ax1.set_title("Original Image")
                    ax1.axis('off')

                    # Preprocessed Image
                    ax2 = self.analysis_canvas.fig.add_subplot(gs[0, 1])
                    if self.preprocessed_image is not None:
                        # Normalize image to 0-1 range for display
                        normalized_image = self.preprocessed_image.copy()
                        if normalized_image.max() > 1.0:
                            normalized_image = normalized_image / 255.0

                        if self.preprocessed_image.shape[-1] == 1:
                            ax2.imshow(normalized_image.squeeze(), cmap='gray')
                        else:
                            ax2.imshow(normalized_image)
                    else:
                        ax2.text(0.5, 0.5, "No preprocessed image", ha='center', va='center')
                    ax2.set_title("Preprocessed Image")
                    ax2.axis('off')

                    # Segmentation Mask
                    ax3 = self.analysis_canvas.fig.add_subplot(gs[1, 0])
                    if self.pred_mask is not None:
                        # Create a colormap for the mask
                        from matplotlib.colors import ListedColormap
                        # Generate colors for each class
                        import matplotlib.pyplot as plt
                        colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))
                        cmap = ListedColormap(colors)

                        # Display the mask
                        im = ax3.imshow(self.pred_mask, cmap=cmap, vmin=0, vmax=len(self.class_names)-1)

                        # Add a colorbar
                        from mpl_toolkits.axes_grid1 import make_axes_locatable
                        divider = make_axes_locatable(ax3)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        cbar = self.analysis_canvas.fig.colorbar(im, cax=cax)

                        # Set colorbar ticks and labels
                        cbar.set_ticks(np.arange(len(self.class_names)) + 0.5)
                        cbar.set_ticklabels(self.class_names)
                    else:
                        ax3.text(0.5, 0.5, "No segmentation mask", ha='center', va='center')
                    ax3.set_title("Segmentation Mask")
                    ax3.axis('off')

                    # Enhanced Detection (if available)
                    ax4 = self.analysis_canvas.fig.add_subplot(gs[1, 1])
                    if use_enhanced and hasattr(self, 'cell_details') and self.cell_details and hasattr(self, 'enhanced_image') and self.enhanced_image is not None:
                        # Use the enhanced visualizer
                        overlay = self.visualizer.overlay_heatmap(
                            self.enhanced_image,
                            self.pred_mask
                        )
                        ax4.imshow(overlay)
                        ax4.set_title("Enhanced Detection")
                    elif self.pred_mask is not None and hasattr(self, 'original_image'):
                        # Fallback to standard cell detection
                        # Use the predicted class if available, otherwise default to class 1
                        if hasattr(self, 'predicted_class') and self.predicted_class is not None:
                            class_idx = self.predicted_class
                        else:
                            class_idx = 1

                        # Ensure class_idx is within valid range
                        if class_idx >= len(self.class_names):
                            print(f"Warning: Predicted class index {class_idx} is out of range. Using default class index 1.")
                            class_idx = 1  # Use a safe default

                        # Create a binary mask for the selected class
                        binary_mask = (self.pred_mask == class_idx).astype(np.uint8)
                        # Create a simple overlay
                        overlay = np.zeros_like(self.original_image)
                        overlay[binary_mask > 0] = [255, 0, 0]  # Red for detected cells
                        # Blend with original image
                        alpha = 0.3
                        blended = cv2.addWeighted(self.original_image, 1-alpha, overlay, alpha, 0)
                        ax4.imshow(blended)

                        # Safely get class name
                        class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"Unknown (Class {class_idx})"
                        ax4.set_title(f"Cell Detection - {class_name}")
                    else:
                        ax4.text(0.5, 0.5, "No detection data available", ha='center', va='center')
                        ax4.set_title("Cell Detection")
                    ax4.axis('off')

                self.analysis_canvas.fig.tight_layout()
                self.analysis_canvas.draw()

                # Store the current view type
                self.current_view_type = view_type

                # Store the results for future reference if they don't already exist or if we've recalculated them
                if not hasattr(self, 'all_images_results') or self.all_images_results is None or preserve_results or not 'class_idx' in self.all_images_results:
                    # If we're in the else block, we've calculated new results
                    if 'class_idx' not in locals():
                        # If class_idx wasn't defined in this scope, use a default or get it from attributes
                        if hasattr(self, 'predicted_class') and self.predicted_class is not None:
                            class_idx = self.predicted_class
                        else:
                            class_idx = 1

                    self.all_images_results = {
                        'class_idx': class_idx,
                        'pred_mask': self.pred_mask.copy() if hasattr(self, 'pred_mask') and self.pred_mask is not None else None,
                        'original_image': self.original_image.copy() if hasattr(self, 'original_image') else None,
                        'preprocessed_image': self.preprocessed_image.copy() if self.preprocessed_image is not None else None
                    }

            # Handle ROI Detection view
            elif view_type == "ROI Detection":
                # Check if we're switching from All Images view and need to preserve results
                if preserve_results and hasattr(self, 'all_images_results') and self.all_images_results:
                    # We have results from All Images view that we should preserve
                    print("Preserving All Images results while showing ROI Detection")

                    # Store the All Images results if they might be needed later
                    # We already have them stored in self.all_images_results

                # Check if we already have ROI results that we can reuse
                if hasattr(self, 'roi_results') and self.roi_results and not preserve_results:
                    # Reuse existing ROI results
                    print("Reusing existing ROI results")
                    roi_mask = self.roi_results['roi_mask']
                    roi_method = self.roi_results['roi_method']
                    overlay = self.roi_results['overlay']
                    title = self.roi_results['title']
                else:
                    # Get the selected ROI method
                    roi_method = self.roi_method_combo.currentText().lower()

                    # Use the image to detect ROI
                    image_for_roi = self.preprocessed_image if self.preprocessed_image is not None else self.original_image

                    if roi_method == "auto":
                        # Automatically detect the best ROI method
                        roi_mask, best_method = auto_detect_best_roi(image_for_roi)
                        title = f"ROI Detection (Auto - {best_method.upper()})"
                    else:
                        # Use the selected method
                        roi_mask = detect_roi(image_for_roi, method=roi_method)
                        title = f"ROI Detection ({roi_method.upper()})"

                    # Visualize the ROI
                    overlay = visualize_roi(image_for_roi, roi_mask, alpha=0.3)

                # Display in the canvas
                ax = self.analysis_canvas.fig.add_subplot(111)
                ax.imshow(overlay)
                ax.set_title(title)
                ax.axis('off')

                # Add info text
                coverage = np.mean(roi_mask) * 100
                info_text = f"ROI Coverage: {coverage:.1f}% of image"
                ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
                        bbox=dict(facecolor='white', alpha=0.8))

                self.analysis_canvas.draw()

                # Store the current view type
                self.current_view_type = view_type

                # Store the ROI results for future reference if they don't already exist or if we've recalculated them
                if not hasattr(self, 'roi_results') or self.roi_results is None or preserve_results:
                    self.roi_results = {
                        'roi_mask': roi_mask,
                        'roi_method': roi_method,
                        'overlay': overlay,
                        'title': title
                    }

            # No other views are needed as per requirements
            if False:  # Removed "Segmentation" view
                # Visualize segmentation prediction
                ax = self.analysis_canvas.fig.add_subplot(111)

                # Use visualization function from visualization module
                if hasattr(self, 'original_image'):
                    # Create a figure for display
                    fig = visualize_prediction(
                        self.original_image, 
                        pred_mask=self.pred_mask,
                        class_names=self.class_names,
                        show_dice=False
                    )
                    # Convert figure to image
                    import io
                    from matplotlib.backends.backend_agg import FigureCanvasAgg
                    canvas = FigureCanvasAgg(fig)
                    canvas.draw()
                    buf = io.BytesIO()
                    canvas.print_png(buf)
                    buf.seek(0)
                    import PIL.Image
                    img = np.array(PIL.Image.open(buf))

                    # Display the image
                    ax.imshow(img)
                    ax.axis('off')
                    ax.set_title("Segmentation Results")
                    plt.close(fig)  # Close the original figure
                else:
                    # Fallback if original image is not available
                    ax.imshow(self.pred_mask)
                    ax.set_title("Segmentation Mask")
                    ax.axis('off')

                self.analysis_canvas.draw()

            elif view_type == "Cell Detection":
                # Cell Detection view has been removed as per requirements
                pass

            elif view_type == "Density Heatmap":
                # Density Heatmap view has been removed as per requirements
                pass

            elif view_type == "Classification":
                # Classification view has been removed as per requirements
                pass

            # Enhanced view options have been removed as per requirements
            elif use_enhanced and view_type == "Enhanced Detection":
                # Enhanced Detection view has been removed as per requirements
                pass

            elif use_enhanced and view_type == "Spatial Analysis":
                # Spatial Analysis view has been removed as per requirements
                pass

            elif use_enhanced and view_type == "Morphology":
                # Morphology view has been removed as per requirements
                pass

        except Exception as e:
            print(f"Error in update_analysis_view: {e}")

def main():
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = PathologyCellCounterApp()
    window.show()
    sys.exit(app.exec_())
