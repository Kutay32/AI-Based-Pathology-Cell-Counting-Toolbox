# Training Models in the AI-Based Pathology Cell Counting Toolbox

This guide explains how to train custom models on your own datasets using the AI-Based Pathology Cell Counting Toolbox.

## Prerequisites

Before training a model, ensure you have:

1. Installed all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepared a dataset with the following structure:
   ```
   dataset/
   ├── images/          # Contains input images
   │   ├── image1.png
   │   ├── image2.png
   │   └── ...
   └── labels/          # Contains corresponding segmentation masks
       ├── image1.png   # Pixel values represent class IDs (0, 1, 2, etc.)
       ├── image2.png
       └── ...
   ```

   Alternatively, you can use the provided PanNuke dataset:
   ```
   dataset/pannuke_processed/
   ├── fold1/
   │   ├── images/
   │   └── labels/
   ├── fold2/
   │   ├── images/
   │   └── labels/
   └── fold3/
   │   ├── images/
   │   └── labels/
   ```

## Training a Model

We provide a script `train_model_example.py` that demonstrates how to train a model on a dataset:

```bash
python train_model_example.py --dataset path/to/dataset --output path/to/output_dir
```

### Command-line Arguments

- `--dataset`: Path to the dataset directory (required)
- `--output`: Directory to save the trained model and results (required)
- `--batch-size`: Batch size for training (default: 16)
- `--epochs`: Number of epochs to train (default: 50)
- `--img-size`: Size to resize images to (default: 256)
- `--val-split`: Validation split ratio (default: 0.2)

### Example Usage

Train on the PanNuke dataset fold1:

```bash
python train_model_example.py --dataset dataset/pannuke_processed/fold1 --output models/trained_fold1 --epochs 100
```

Train on a custom dataset with different parameters:

```bash
python train_model_example.py --dataset path/to/custom_dataset --output models/custom_model --batch-size 8 --img-size 512 --epochs 200
```

## Training Process

The training process includes the following steps:

1. **Data Loading**: The script loads images and their corresponding segmentation masks from the dataset directory.

2. **Preprocessing**:
   - Images are resized to the specified size (default: 256×256)
   - Pixel values are normalized to the range [0, 1]
   - Masks are resized using nearest-neighbor interpolation to preserve class IDs
   - Masks are converted to categorical format (one-hot encoding)

3. **Data Splitting**: The dataset is split into training and validation sets according to the specified ratio.

4. **Model Building**: An Attention U-Net model is built with the appropriate input shape and number of output classes.

5. **Training**: The model is trained using:
   - Categorical cross-entropy loss function
   - Adam optimizer
   - Early stopping to prevent overfitting
   - Learning rate reduction on plateau
   - Model checkpointing to save the best model

6. **Visualization**: Training and validation accuracy/loss are plotted and saved.

## Using a Trained Model

After training, you can use the trained model for inference:

```python
from models.training import load_trained_model
from models.losses import combined_loss, MeanIoUCustom

# Load the trained model
model = load_trained_model(
    "path/to/model_weights.keras",
    custom_objects={
        'combined_loss': combined_loss,
        'MeanIoUCustom': MeanIoUCustom
    }
)

# Use the model for prediction
prediction = model.predict(input_image)
```

You can also use the trained model with the main application:

```bash
python main.py --cli --image path/to/image.png --model path/to/model_weights.keras --output path/to/results
```

## Advanced Training Options

### Using Custom Loss Functions

The default training script uses categorical cross-entropy loss, but you can modify it to use the combined loss function:

```python
from models.losses import combined_loss, MeanIoUCustom

# Compile the model with combined loss
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=combined_loss,
    metrics=[MeanIoUCustom]
)
```

### Data Augmentation

To improve model generalization, you can add data augmentation:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create data generators with augmentation
data_gen_args = dict(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Fit the generators
image_datagen.fit(X_train)
mask_datagen.fit(y_train)

# Create generators
image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=42)
mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=42)

# Combine generators
train_generator = zip(image_generator, mask_generator)

# Train with generators
model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, earlystop, lr_scheduler]
)
```

## Troubleshooting

### Out of Memory Errors

If you encounter out of memory errors during training:

1. Reduce the batch size: `--batch-size 4` or `--batch-size 2`
2. Reduce the image size: `--img-size 128`
3. Use data generators to load batches of data instead of loading the entire dataset into memory

### Class Imbalance

If your dataset has class imbalance (some classes appear much more frequently than others):

1. Use the weighted loss function in models/losses.py
2. Adjust class weights based on class frequency in your dataset

### Overfitting

If your model is overfitting (high training accuracy but low validation accuracy):

1. Increase dropout rate in the model
2. Add data augmentation
3. Reduce model complexity
4. Use early stopping (already implemented in the training script)

## Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [Keras Documentation](https://keras.io/api/)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Attention U-Net Paper](https://arxiv.org/abs/1804.03999)