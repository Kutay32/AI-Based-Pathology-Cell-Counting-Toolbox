import h5py
import os

def check_h5_file(file_path):
    print(f"\nChecking model: {file_path}")
    try:
        # Print file size
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")
        
        # Open the h5 file
        with h5py.File(file_path, 'r') as f:
            # Print the keys at the root level
            print("Root level keys:", list(f.keys()))
            
            # Check if it's a Keras model
            if 'model_weights' in f:
                print("This appears to be a Keras model")
                
                # Try to get the model config
                if 'model_config' in f.attrs:
                    print("Model config found")
                else:
                    print("No model config found")
                
                # Print the model architecture by exploring the weights
                print("\nModel architecture based on weights:")
                model_weights = f['model_weights']
                
                def print_group(name, obj):
                    if isinstance(obj, h5py.Group):
                        print(f"Layer: {name}")
                        if 'kernel:0' in obj:
                            shape = obj['kernel:0'].shape
                            print(f"  Kernel shape: {shape}")
                    
                model_weights.visititems(print_group)
            
            # If it's not a standard Keras model, try to explore the structure
            else:
                print("This doesn't appear to be a standard Keras model")
                print("\nFile structure:")
                
                def print_attrs(name, obj):
                    print(f"Object: {name}")
                    if isinstance(obj, h5py.Dataset):
                        print(f"  Dataset shape: {obj.shape}, dtype: {obj.dtype}")
                
                f.visititems(print_attrs)
        
        return True
    except Exception as e:
        print(f"Error checking h5 file: {str(e)}")
        return False

# Check all .h5 models in the current directory
models = ["ef4.h5", "Efficent_pet_203_clf-end.h5", "model.h5"]
for model_path in models:
    check_h5_file(model_path)