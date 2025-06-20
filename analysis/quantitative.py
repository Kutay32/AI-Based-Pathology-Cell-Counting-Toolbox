import numpy as np
from scipy import ndimage
from skimage.measure import regionprops, label

class QuantitativeAnalyzer:
    def __init__(self, pixel_size=0.25):  # microns per pixel
        self.pixel_size = pixel_size
        self.mm2_per_pixel = (pixel_size / 1000) ** 2

    def calculate_density(self, binary_mask, roi_mask=None):
        """Compute cell density in cells/mm²"""
        if roi_mask is None:
            roi_mask = np.ones_like(binary_mask)
            
        roi_area = np.sum(roi_mask) * self.mm2_per_pixel
        
        # Count cells (connected components)
        labeled_mask = label(binary_mask)
        cell_count = np.max(labeled_mask)
        
        return cell_count / roi_area if roi_area > 0 else 0

    def morphological_features(self, binary_mask):
        """Extract shape features from cells"""
        labeled = label(binary_mask)
        regions = regionprops(labeled)
        
        features = []
        for region in regions:
            features.append({
                'area': region.area * self.mm2_per_pixel,
                'perimeter': region.perimeter * self.pixel_size,
                'circularity': (4 * np.pi * region.area) / (region.perimeter ** 2) if region.perimeter > 0 else 0,
                'eccentricity': region.eccentricity,
                'solidity': region.solidity
            })
        return features

    def detect_anomalies(self, features, density):
        """Identify abnormal cell patterns"""
        # Calculate statistics
        areas = [f['area'] for f in features]
        mean_area = np.mean(areas) if areas else 0
        std_area = np.std(areas) if areas else 0
        
        # Define anomaly criteria
        size_anomaly = any(a > mean_area + 3 * std_area for a in areas) if areas else False
        density_anomaly = density > 5000  # cells/mm²
        shape_anomaly = any(f['circularity'] < 0.7 for f in features) if features else False
        
        return {
            'size_anomaly': size_anomaly,
            'density_anomaly': density_anomaly,
            'shape_anomaly': shape_anomaly,
            'mean_area': mean_area,
            'cell_density': density
        }
        
    def spatial_analysis(self, binary_mask, class_mask=None):
        """Analyze spatial distribution of cells"""
        labeled = label(binary_mask)
        regions = regionprops(labeled)
        
        if len(regions) < 2:
            return {
                'nearest_neighbor_distance': 0,
                'clustering_index': 0,
                'is_clustered': False
            }
        
        # Extract centroids
        centroids = np.array([r.centroid for r in regions])
        
        # Calculate nearest neighbor distances
        distances = []
        for i, c1 in enumerate(centroids):
            dist_to_others = [np.sqrt(np.sum((c1 - c2)**2)) for j, c2 in enumerate(centroids) if i != j]
            if dist_to_others:
                distances.append(min(dist_to_others))
        
        mean_distance = np.mean(distances) * self.pixel_size if distances else 0
        std_distance = np.std(distances) * self.pixel_size if distances else 0
        
        # Calculate clustering index (coefficient of variation)
        clustering_index = std_distance / mean_distance if mean_distance > 0 else 0
        
        return {
            'nearest_neighbor_distance': mean_distance,
            'clustering_index': clustering_index,
            'is_clustered': clustering_index > 0.5  # Threshold for clustering
        }