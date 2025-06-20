# AI-Based Pathology Cell Counting Toolbox

An advanced toolbox for automated cell counting in pathology images using deep learning. This system uses state-of-the-art AI algorithms to accurately detect and count cells from microscopic images, reducing manual workload and increasing diagnostic efficiency.

## Features

### 1. Cell Detection & Counting Module
- AI-Based Image Processing: Deep learning models (Attention U-Net) for cell identification and counting
- Dual Model Approach: Separate UNet models for segmentation and classification tasks
- Pre-Trained Models: Implementation of pre-trained models for segmentation and classification
- Multi-Cell Type Recognition: Differentiation between multiple cell types (Neoplastic, Inflammatory, Connective, Dead, Epithelial)

### 2. Image Preprocessing & Enhancement
- Noise Reduction & Normalization: Adaptive histogram equalization and Gaussian filtering
- Color Deconvolution: Extraction of specific stain components (H&E, DAB)
- Automated ROI Selection: Highlighting areas with the highest cell density

### 3. AI-Powered Quantitative Analysis
- Cell Density Estimation: Automated measurement of cell density per region
- Morphological Feature Extraction: Shape, size, and distribution analysis
- Anomaly Detection: Identification of abnormal cell growth patterns

### 4. User Interface & Visualization Tools
- Interactive Dashboard: User-friendly UI to upload images, run analysis, and visualize results
- Heatmaps & Segmentation Masks: Highlighting detected cells and providing insights into cell clustering
- Exportable Reports: Generating summary reports for integration into pathology workflow

### 5. Model Training & Optimization
- Advanced Loss Functions: Combined loss function using Dice, Focal, Tversky, and Boundary losses
- Custom Metrics: Mean IoU for accurate evaluation
- Optimized Training: Early stopping, learning rate scheduling, and model checkpointing
- Data Augmentation: Rotation, shift, zoom, and flip transformations to improve model generalization
- Active Learning: Uncertainty-based sample selection for efficient expert annotation
- Edge AI Support: Model quantization for deployment on pathology workstations

## Project Structure

```
AI-Based-Pathology-Cell-Counting-Toolbox/
├── analysis/                  # Cell detection and analysis
│   ├── __init__.py
│   └── cell_detection.py      # Cell detection and feature extraction
├── models/                    # Deep learning models
│   ├── __init__.py
│   ├── unet.py                # Original Attention U-Net architecture
│   ├── unet_models.py         # Segmentation and classification UNet models
│   ├── losses.py              # Loss functions and metrics
│   ├── training.py            # Original model training utilities
│   ├── training_utils.py      # Training utilities for both model types
│   └── inference.py           # Inference utilities for both model types
├── preprocessing/             # Image preprocessing
│   ├── __init__.py
│   └── image_enhancement.py   # Image enhancement and normalization
├── ui/                        # User interface
│   ├── __init__.py
│   └── app.py                 # PyQt5-based GUI
├── utils/                     # Utility functions
│   ├── __init__.py
│   └── helpers.py             # Helper functions
├── visualization/             # Visualization tools
│   ├── __init__.py
│   └── visualization.py       # Visualization and reporting
├── main.py                    # Main entry point
└── README.md                  # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI-Based-Pathology-Cell-Counting-Toolbox.git
cd AI-Based-Pathology-Cell-Counting-Toolbox
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained model:
```bash
# Place the pre-trained model in the project root directory
# The default model name is "unet_model_v3.5.keras"
```

## Usage

### Graphical User Interface (GUI)

To launch the GUI:

```bash
python main.py
```

The GUI provides an intuitive interface for:
- Loading pathology images
- Preprocessing images with various options
- Running cell detection and analysis
- Visualizing results with different views
- Generating and exporting reports
- Analyzing datasets and generating reports

#### Using the GUI Tools

1. **Loading and Analyzing Images**:
   - Click the "Load Image" button to select an image file
   - Use the preprocessing options (Enhance Contrast, Denoise, Detect ROI) as needed
   - Click "Preprocess" to apply the selected preprocessing options
   - Click "Analyze" to run the cell detection and analysis

2. **Viewing Analysis Results**:
   - Use the "Cell Class" dropdown to select the cell type to analyze
   - Use the "View" dropdown to switch between different visualization modes:
     - Segmentation: Shows the segmentation mask overlaid on the image
     - Cell Detection: Shows individual detected cells
     - Density Heatmap: Shows a heatmap of cell density

3. **Dataset Analysis**:
   - Click the "Analyze Dataset" button to select a dataset directory
   - Set the maximum number of files to analyze per folder
   - The analysis will process the dataset and display statistics
   - A report will be generated and saved automatically

### Command Line Interface (CLI)

For batch processing or integration into other workflows:

```bash
python main.py --cli --image <path_to_image> [--model <path_to_model>] [--output <output_directory>]
```

Options:
- `--cli`: Run in command-line mode
- `--image`: Path to the input image (required in CLI mode)
- `--model`: Path to the pre-trained model (default: "unet_model_v3.5.keras")
- `--output`: Directory to save the output results

### Dataset Analysis

For analyzing datasets and generating reports:

```bash
python main.py --analyze-dataset --dataset <path_to_dataset> [--max-files <number>] [--output <output_directory>]
```

Options:
- `--analyze-dataset`: Run in dataset analysis mode
- `--dataset`: Path to the dataset directory (required in dataset analysis mode)
- `--max-files`: Maximum number of files to analyze per folder (default: 100)
- `--output`: Directory to save the analysis results and reports

## Examples

### Using the Main Application

```bash
# Run analysis on a sample image and save results
python main.py --cli --image samples/sample1.jpg --output results/sample1

# Analyze a dataset and generate a report
python main.py --analyze-dataset --dataset dataset/pannuke_processed --max-files 50 --output results/dataset_analysis
```

### Using Both Models for Inference

The toolbox includes an example script for running inference with both segmentation and classification models:

```bash
# Run inference with both models
python inference_example.py --image <path_to_image> --segmentation-model <path_to_segmentation_model> --classification-model <path_to_classification_model> --output <output_directory>
```

### Programmatic Inference

You can also use the models programmatically for inference:

```python
from models.inference import load_trained_model, predict_segmentation, get_predicted_class
from utils.helpers import load_image, get_default_class_names

# Load models
segmentation_model = load_trained_model("segmentation_model.keras", model_type="segmentation")
classification_model = load_trained_model("classification_model.keras", model_type="classification")

# Load and preprocess image
image = load_image("sample.jpg")

# Run segmentation
segmentation_mask = predict_segmentation(segmentation_model, image)

# Run classification
predicted_class, probability = get_predicted_class(classification_model, image)
class_names = get_default_class_names()
predicted_class_name = class_names[predicted_class]

print(f"Predicted class: {predicted_class_name}, Probability: {probability:.2f}")
```

## Training Your Own Models

The toolbox includes utilities for training your own segmentation and classification models on custom datasets. For advanced features like data augmentation, active learning, and edge AI support, see [docs/advanced_features.md](docs/advanced_features.md).

### Using the Example Script

The easiest way to train models is using the provided example script:

```bash
# Train both segmentation and classification models
python train_models_example.py --dataset <path_to_dataset> --output <output_directory> --model-type both

# Train only segmentation model
python train_models_example.py --dataset <path_to_dataset> --output <output_directory> --model-type segmentation

# Train only classification model
python train_models_example.py --dataset <path_to_dataset> --output <output_directory> --model-type classification
```

### Training Models Programmatically

You can also train models programmatically:

```python
# For segmentation model
from models.unet_models import build_unet_segmentation
from models.training_utils import train_segmentation_model
from models.losses import combined_loss, MeanIoUCustom

# Build segmentation model
seg_model = build_unet_segmentation(input_shape=(256, 256, 3), num_classes=6)

# Train segmentation model
seg_history = train_segmentation_model(
    seg_model, 
    train_images, 
    train_masks, 
    val_images, 
    val_masks, 
    batch_size=16, 
    epochs=50,
    model_save_path="segmentation_model.keras"
)

# For classification model
from models.unet_models import build_unet_classification
from models.training_utils import train_classification_model

# Build classification model
cls_model = build_unet_classification(input_shape=(256, 256, 3), num_classes=6)

# Train classification model
cls_history = train_classification_model(
    cls_model, 
    train_images, 
    train_labels, 
    val_images, 
    val_labels, 
    batch_size=16, 
    epochs=50,
    model_save_path="classification_model.keras"
)
```

## Requirements

- Python 3.7+
- TensorFlow 2.5+
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-image
- PyQt5 (for GUI)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was developed to assist pathologists in their diagnostic workflow
- Special thanks to the open-source community for their contributions to the field of medical image analysis
