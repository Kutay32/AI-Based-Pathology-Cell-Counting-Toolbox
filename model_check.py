import tensorflow as tf
import os

def check_model(model_path):
    print(f"\nChecking model: {model_path}")
    try:
        # Load the model
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Print model summary
        print("Model summary:")
        model.summary()
        
        # Print model input and output shapes
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        
        # Print model size
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")
        
        # Check if it's a classification or segmentation model
        if len(model.output_shape) == 2:
            print("This appears to be a classification model")
            print(f"Number of classes: {model.output_shape[-1]}")
        elif len(model.output_shape) == 4:
            print("This appears to be a segmentation model")
            print(f"Number of classes: {model.output_shape[-1]}")
        else:
            print("Model type is unclear based on output shape")
        
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

# Check all .h5 models in the current directory
models = ["ef4.h5", "Efficent_pet_203_clf-end.h5", "model.h5"]
for model_path in models:
    check_model(model_path)