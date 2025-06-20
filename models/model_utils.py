import tensorflow as tf
import numpy as np
import cv2

def dice_loss(y_true, y_pred):
    smooth = 1e-5
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def tversky_loss(y_true, y_pred, alpha=0.7):
    smooth = 1e-5
    y_true_pos = tf.keras.backend.flatten(y_true)
    y_pred_pos = tf.keras.backend.flatten(y_pred)
    true_pos = tf.keras.backend.sum(y_true_pos * y_pred_pos)
    false_neg = tf.keras.backend.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.keras.backend.sum((1 - y_true_pos) * y_pred_pos)
    return 1 - (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

class AttentionGate(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)
        self.conv_g = tf.keras.layers.Conv2D(filters, 1)
        self.conv_x = tf.keras.layers.Conv2D(filters, 1)
        self.psi = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        g, x = inputs
        g1 = self.conv_g(g)
        x1 = self.conv_x(x)
        psi = self.relu(tf.keras.layers.add([g1, x1]))
        psi = self.psi(psi)
        return x * psi

    def get_config(self):
        config = super(AttentionGate, self).get_config()
        return config

def unet_model(input_size=(256, 256, 1)):
    inputs = tf.keras.Input(input_size)

    # Encoder
    c1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    # Bridge
    b = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    b = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(b)

    # Decoder with attention gates
    u1 = tf.keras.layers.UpSampling2D((2, 2))(b)
    att1 = AttentionGate(128)([c2, u1])
    u1 = tf.keras.layers.concatenate([u1, att1])
    c3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(u1)
    c3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c3)

    u2 = tf.keras.layers.UpSampling2D((2, 2))(c3)
    att2 = AttentionGate(64)([c1, u2])
    u2 = tf.keras.layers.concatenate([u2, att2])
    c4 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
    c4 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c4)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(c4)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

def load_models(seg_path, cls_path):
    seg_model = tf.keras.models.load_model(
        seg_path,
        custom_objects={
            'dice_loss': dice_loss,
            'tversky_loss': tversky_loss,
            'AttentionGate': AttentionGate
        }
    )

    cls_model = tf.keras.models.load_model(
        cls_path,
        custom_objects={
            'dice_loss': dice_loss,
            'tversky_loss': tversky_loss
        }
    )
    return seg_model, cls_model

def adaptive_thresholding(prediction, threshold=0.2):
    """Apply adaptive thresholding to model output"""
    # First try Otsu's method
    prediction_8bit = (prediction * 255).astype(np.uint8)
    otsu_thresh = cv2.threshold(
        prediction_8bit, 0, 255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # If Otsu's method doesn't detect any cells, use a fixed threshold
    if np.sum(otsu_thresh) == 0:
        fixed_thresh = cv2.threshold(
            prediction_8bit, int(threshold * 255), 255, 
            cv2.THRESH_BINARY
        )[1]
        return fixed_thresh / 255.0

    return otsu_thresh / 255.0

def predict_cells(seg_model, image, min_size=50, morphological_cleanup=True, cls_model=None):
    """Enhanced prediction with morphological cleanup and classification"""
    # Ensure image is in the right format for the model
    if len(image.shape) == 2:
        # Convert single channel to 3 channels
        image = np.stack([image, image, image], axis=-1)
    elif image.shape[-1] == 1 and len(image.shape) == 3:
        # Convert single channel to 3 channels
        image = np.concatenate([image, image, image], axis=-1)
    elif image.shape[-1] > 3 and len(image.shape) == 3:
        # Use only the first 3 channels
        image = image[:, :, :3]

    # Predict segmentation
    pred = seg_model.predict(np.expand_dims(image, axis=0))[0, ..., 0]
    binary_mask = adaptive_thresholding(pred)

    # Check if any cells were detected
    if np.sum(binary_mask) == 0:
        # If no cells detected, try with a lower threshold
        binary_mask = adaptive_thresholding(pred, threshold=0.1)

    # Convert to uint8 for OpenCV operations
    binary_mask_uint8 = binary_mask.astype(np.uint8)

    if morphological_cleanup:
        # Morphological cleanup with smaller kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(binary_mask_uint8, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    else:
        cleaned = binary_mask_uint8

    # Find connected components (cells)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned
    )

    # If no cells detected, return the original binary mask
    if num_labels <= 1:
        return binary_mask

    # Create a multi-class mask
    multi_class_mask = np.zeros_like(labels)

    # If classification model is provided, use it to classify each cell
    if cls_model is not None:
        # Get class prediction for the whole image
        if image.ndim == 3 and image.shape[2] == 3:
            # Image is already in the right format for classification
            cls_input = np.expand_dims(image, axis=0)
        else:
            # Convert to 3 channels if needed
            cls_input = np.expand_dims(np.stack([image[:,:,0]]*3, axis=-1), axis=0)

        # Get class probabilities
        class_probs = cls_model.predict(cls_input)[0]
        predicted_class = np.argmax(class_probs) + 1  # +1 because 0 is background

        # Assign the predicted class to all detected cells
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                multi_class_mask[labels == i] = predicted_class
    else:
        # No classification model, assign class 1 to all cells
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                multi_class_mask[labels == i] = 1

    return multi_class_mask
