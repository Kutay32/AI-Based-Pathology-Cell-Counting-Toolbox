import matplotlib.pyplot as plt
import numpy as np
import cv2

class ResultVisualizer:
    def __init__(self, class_colors=None):
        self.class_colors = class_colors or {
            "Neoplastic": (255, 0, 0),
            "Inflammatory": (0, 255, 0),
            "Connective": (0, 0, 255),
            "Dead": (255, 255, 0),
            "Epithelial": (255, 0, 255)
        }

    def overlay_heatmap(self, original, binary_mask, class_map=None):
        """Create class-specific heatmap overlay"""
        # Create RGB mask
        mask_rgb = np.zeros(original.shape, dtype=np.uint8)

        # Check if binary_mask contains class information (values > 1)
        if class_map is not None:
            # Apply class-specific colors using provided class_map
            for class_name, color in self.class_colors.items():
                class_id = list(self.class_colors.keys()).index(class_name) + 1
                mask_indices = np.where(class_map == class_id)
                if len(mask_indices[0]) > 0:  # If there are any pixels with this class
                    mask_rgb[mask_indices] = color
        elif np.max(binary_mask) > 1:
            # If binary_mask contains values > 1, treat it as a class map
            for class_name, color in self.class_colors.items():
                class_id = list(self.class_colors.keys()).index(class_name) + 1
                mask_indices = np.where(binary_mask == class_id)
                if len(mask_indices[0]) > 0:  # If there are any pixels with this class
                    mask_rgb[mask_indices] = color
        else:
            # If no class map and binary_mask is truly binary, use a default color (white)
            mask_rgb[binary_mask > 0] = (255, 255, 255)

        # Blend with original image
        overlay = cv2.addWeighted(original, 0.7, mask_rgb, 0.3, 0)

        # Add contours - handle both binary and class masks
        if np.max(binary_mask) > 1:
            # For class masks, we need to create a binary version for contour detection
            binary_version = (binary_mask > 0).astype(np.uint8)
            contours, _ = cv2.findContours(
                binary_version, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
        else:
            # For binary masks, use as is
            contours, _ = cv2.findContours(
                binary_mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )

        # Draw contours
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
        return overlay

    def plot_dashboard(self, original, processed, mask, counts, features=None):
        """Generate comprehensive result dashboard"""
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))

        # Original vs Processed
        ax[0, 0].imshow(original)
        ax[0, 0].set_title("Original Image")
        ax[0, 0].axis('off')

        # Show processed image
        if len(processed.shape) == 2 or processed.shape[2] == 1:
            ax[0, 1].imshow(processed, cmap='gray')
        else:
            ax[0, 1].imshow(processed)
        ax[0, 1].set_title("Processed Image")
        ax[0, 1].axis('off')

        # Segmentation mask
        overlay = self.overlay_heatmap(original, mask)
        ax[1, 0].imshow(overlay)
        ax[1, 0].set_title("Cell Detection")
        ax[1, 0].axis('off')

        # Statistics
        stats_text = "\n".join([f"{k}: {v}" for k, v in counts.items()])
        stats_text += f"\n\nTotal Cells: {sum(counts.values())}"

        if features:
            density = features.get('cell_density', 0)
            stats_text += f"\nDensity: {density:.2f} cells/mm²"

            # Add anomaly alerts
            if features.get('density_anomaly', False):
                stats_text += "\n\nALERT: High cell density detected!"
            if features.get('size_anomaly', False):
                stats_text += "\n\nALERT: Abnormal cell sizes detected!"
            if features.get('shape_anomaly', False):
                stats_text += "\n\nALERT: Abnormal cell shapes detected!"

            # Add spatial analysis if available
            if 'clustering_index' in features:
                stats_text += f"\n\nClustering Index: {features['clustering_index']:.2f}"
                if features.get('is_clustered', False):
                    stats_text += "\nCells show significant clustering"

        ax[1, 1].text(0.1, 0.5, stats_text, fontsize=12)
        ax[1, 1].axis('off')

        plt.tight_layout()
        return fig

    def generate_report(self, image_path, counts, features, output_path):
        """Save PDF report with results"""
        try:
            from matplotlib.backends.backend_pdf import PdfPages

            with PdfPages(output_path) as pdf:
                # Summary page
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.text(0.1, 0.8, f"Analysis Report: {image_path}", fontsize=16)
                ax.text(0.1, 0.6, "\n".join(
                    [f"{k}: {v}" for k, v in counts.items()]), fontsize=12)
                ax.text(0.1, 0.4, f"Cell Density: {features['cell_density']:.2f} cells/mm²")

                # Add anomaly alerts
                y_pos = 0.3
                if features.get('density_anomaly', False):
                    ax.text(0.1, y_pos, "ALERT: High cell density detected!", 
                           fontsize=12, color='red')
                    y_pos -= 0.05
                if features.get('size_anomaly', False):
                    ax.text(0.1, y_pos, "ALERT: Abnormal cell sizes detected!", 
                           fontsize=12, color='red')
                    y_pos -= 0.05
                if features.get('shape_anomaly', False):
                    ax.text(0.1, y_pos, "ALERT: Abnormal cell shapes detected!", 
                           fontsize=12, color='red')

                ax.axis('off')
                pdf.savefig(fig)
                plt.close()

                # Feature distributions if available
                if features.get('morphology'):
                    fig = self.plot_morphology(features['morphology'])
                    pdf.savefig(fig)
                    plt.close()

            return True
        except Exception as e:
            print(f"Error generating report: {e}")
            return False

    def plot_morphology(self, morphology_features):
        """Plot morphological feature distributions"""
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        # Extract features
        areas = [f['area'] for f in morphology_features]
        circularities = [f['circularity'] for f in morphology_features]
        eccentricities = [f['eccentricity'] for f in morphology_features]
        solidities = [f['solidity'] for f in morphology_features]

        # Plot histograms
        axs[0, 0].hist(areas, bins=20)
        axs[0, 0].set_title('Cell Area Distribution')
        axs[0, 0].set_xlabel('Area (μm²)')

        axs[0, 1].hist(circularities, bins=20)
        axs[0, 1].set_title('Circularity Distribution')
        axs[0, 1].set_xlabel('Circularity (0-1)')

        axs[1, 0].hist(eccentricities, bins=20)
        axs[1, 0].set_title('Eccentricity Distribution')
        axs[1, 0].set_xlabel('Eccentricity (0-1)')

        axs[1, 1].hist(solidities, bins=20)
        axs[1, 1].set_title('Solidity Distribution')
        axs[1, 1].set_xlabel('Solidity (0-1)')

        plt.tight_layout()
        return fig
