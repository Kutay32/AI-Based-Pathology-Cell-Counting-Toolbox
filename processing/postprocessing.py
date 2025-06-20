import numpy as np
import cv2
from skimage.measure import regionprops, label
from sklearn.cluster import DBSCAN

class CellCounter:
    def __init__(self, class_names=None, min_cell_size=50, max_cell_size=1000):
        self.class_names = class_names or [
            "Background", "Neoplastic", "Inflammatory", 
            "Connective", "Dead", "Epithelial"
        ]
        self.min_cell_size = min_cell_size
        self.max_cell_size = max_cell_size

    def process_detections(self, mask, class_probs=None):
        """Identify cells and classify them"""
        # Check if mask is already a multi-class mask
        if np.max(mask) > 1:
            # Multi-class mask - each cell already has a class
            labeled_mask = label(mask > 0)  # Create a labeled mask from any non-zero values
            regions = regionprops(labeled_mask)

            counts = {cls: 0 for cls in self.class_names[1:]}
            cell_details = []

            for region in regions:
                # Size filtering
                if region.area < self.min_cell_size or region.area > self.max_cell_size:
                    continue

                # Get the most common class in this region
                minr, minc, maxr, maxc = region.bbox
                region_mask = mask[minr:maxr, minc:maxc]
                region_mask = region_mask[labeled_mask[minr:maxr, minc:maxc] == region.label]

                # Get the most common class (excluding 0/background)
                classes, counts_per_class = np.unique(region_mask[region_mask > 0], return_counts=True)
                if len(classes) == 0:
                    continue

                class_id = classes[np.argmax(counts_per_class)]

                # Only count if class_id is valid
                if class_id > 0 and class_id < len(self.class_names):
                    cls_name = self.class_names[class_id]
                    counts[cls_name] += 1
                    cell_details.append({
                        'centroid': region.centroid,
                        'bbox': region.bbox,
                        'class': cls_name,
                        'probability': 1.0,  # We don't have probabilities in this case
                        'area': region.area
                    })
        else:
            # Binary mask - use class_probs to classify cells
            labeled_mask = label(mask)
            regions = regionprops(labeled_mask)

            counts = {cls: 0 for cls in self.class_names[1:]}
            cell_details = []

            # If class_probs is a 1D array (whole image classification)
            if class_probs is not None and len(class_probs.shape) == 1:
                # Use the highest probability class for all cells
                class_id = np.argmax(class_probs)

                for region in regions:
                    # Size filtering
                    if region.area < self.min_cell_size or region.area > self.max_cell_size:
                        continue

                    # Only count if probability > 0.3
                    if class_probs[class_id] > 0.3 and class_id > 0:
                        cls_name = self.class_names[class_id]
                        counts[cls_name] += 1
                        cell_details.append({
                            'centroid': region.centroid,
                            'bbox': region.bbox,
                            'class': cls_name,
                            'probability': class_probs[class_id],
                            'area': region.area
                        })
            # If class_probs is a 3D array (per-pixel classification)
            elif class_probs is not None and len(class_probs.shape) == 3:
                for region in regions:
                    # Size filtering
                    if region.area < self.min_cell_size or region.area > self.max_cell_size:
                        continue

                    # Extract region from classification map
                    minr, minc, maxr, maxc = region.bbox
                    region_probs = class_probs[minr:maxr, minc:maxc, :]

                    # Classify by average probability
                    avg_probs = np.mean(region_probs, axis=(0, 1))
                    class_id = np.argmax(avg_probs)

                    # Only count if probability > 0.3
                    if avg_probs[class_id] > 0.3 and class_id > 0:
                        cls_name = self.class_names[class_id]
                        counts[cls_name] += 1
                        cell_details.append({
                            'centroid': region.centroid,
                            'bbox': region.bbox,
                            'class': cls_name,
                            'probability': avg_probs[class_id],
                            'area': region.area
                        })
            else:
                # No class_probs provided, assign all cells to class 1
                for region in regions:
                    # Size filtering
                    if region.area < self.min_cell_size or region.area > self.max_cell_size:
                        continue

                    # Assign to first class (index 1)
                    cls_name = self.class_names[1]
                    counts[cls_name] += 1
                    cell_details.append({
                        'centroid': region.centroid,
                        'bbox': region.bbox,
                        'class': cls_name,
                        'probability': 1.0,
                        'area': region.area
                    })

        return counts, cell_details

    def separate_clumped_cells(self, binary_mask):
        """Use distance transform to separate touching cells"""
        # Compute distance transform
        dist = cv2.distanceTransform(
            binary_mask.astype(np.uint8), 
            cv2.DIST_L2, 5
        )

        # Find local maxima
        kernel = np.ones((5, 5), np.uint8)
        local_max = cv2.dilate(dist, kernel)
        peaks = (dist == local_max) & (dist > 0.7 * dist.max())

        # Marker-controlled watershed
        markers = label(peaks)
        binary_mask_3c = cv2.cvtColor(binary_mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        markers = markers.astype(np.int32)
        cv2.watershed(binary_mask_3c, markers)

        # Convert watershed result to binary mask
        segmented = (markers > 0).astype(np.uint8)
        return segmented

    def cluster_cells(self, cell_details, eps=50):
        """DBSCAN clustering to remove duplicate detections"""
        if not cell_details:
            return {cls: 0 for cls in self.class_names[1:]}, []

        points = np.array([c['centroid'] for c in cell_details])
        if len(points) == 0:
            return {cls: 0 for cls in self.class_names[1:]}, []

        clustering = DBSCAN(eps=eps, min_samples=1).fit(points)
        unique_cells = []
        for cluster_id in np.unique(clustering.labels_):
            cluster_cells = [c for i, c in enumerate(cell_details) 
                            if clustering.labels_[i] == cluster_id]
            # Keep highest probability detection
            best_cell = max(cluster_cells, key=lambda x: x['probability'])
            unique_cells.append(best_cell)

        # Rebuild counts
        counts = {cls: 0 for cls in self.class_names[1:]}
        for cell in unique_cells:
            counts[cell['class']] += 1

        return counts, unique_cells
