import cv2
import numpy as np
from skimage import exposure, color

class ImagePreprocessor:
    def __init__(self, stain_matrix=None):
        # H&E stain matrix (Ruifrok & Johnston)
        if stain_matrix is None:
            self.stain_matrix = np.array([
                [0.65, 0.70, 0.29],
                [0.07, 0.99, 0.11],
                [0.27, 0.57, 0.78]
            ])
        else:
            self.stain_matrix = stain_matrix
        self.od_threshold = 0.15  # Optical density threshold

        self.od_threshold = 0.15  # Optical density threshold

    def load_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found at {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def normalize_brightness(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_norm = clahe.apply(l)
        lab_norm = cv2.merge((l_norm, a, b))
        return cv2.cvtColor(lab_norm, cv2.COLOR_LAB2RGB)

    def deconvolve_stains(self, img):
        """Perform color deconvolution using Macenko method"""
        # Convert to optical density space
        od = -np.log((img.astype(np.float32) + 1) / 256)
        od = od.reshape((-1, 3))

        # Remove transparent/saturated pixels
        od_thresh = od[~np.any(od < self.od_threshold, axis=1)]

        # Calculate SVD
        _, eigvecs = np.linalg.eigh(np.cov(od_thresh.T))
        eigvecs = eigvecs[:, np.argsort(-eigvecs[2, :])]

        # Calculate stain vectors
        stain_vectors = eigvecs[:, :2]
        stains = np.dot(od, stain_vectors)
        return stains.reshape(img.shape[:2] + (2,))
    def enhance_contrast(self, img):
        # Adaptive gamma correction
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mid = 0.5
        mean = np.mean(gray)
        gamma = np.log(mid) / np.log(mean/255)
        return exposure.adjust_gamma(img, gamma=gamma)

    def preprocess_pipeline(self, image_path):
        # List to track applied preprocessing steps
        applied_steps = []

        # Step 1: Load image
        img = self.load_image(image_path)
        applied_steps.append("Image Loading")

        # Step 2: Normalize brightness
        img = self.normalize_brightness(img)
        applied_steps.append("Brightness Normalization")

        # Step 3: Deconvolve stains
        stains = self.deconvolve_stains(img)
        applied_steps.append("Stain Deconvolution")

        # Step 4: Enhance contrast
        enhanced = self.enhance_contrast(img)  # Apply to original image
        applied_steps.append("Contrast Enhancement")

        # Extract hematoxylin channel (nuclei)
        h_channel = stains[..., 0]

        # Normalize to float32 range
        h_channel = (h_channel - h_channel.min()) / (h_channel.max() - h_channel.min())

        # Return processed images and the list of applied steps
        return h_channel.astype(np.float32), enhanced, applied_steps
