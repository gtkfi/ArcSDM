
import arcpy
import numpy as np
from typing import List

from arcsdm.csi.analysis.cosine_similarity import cosine_similarity


def calculate_pixel_to_label_csi(
    pixel_vectors: np.ndarray,
    labeled_features: np.ndarray,
    csv_nodata: float
) -> List[np.ndarray]:
    """
    Calculate CSI between each pixel vector and each labeled point vector.
    Returns:
        list of 2D arrays, one per labeled point.
    """
    nrows, ncols, n_features = pixel_vectors.shape
    n_labeled = len(labeled_features)

    arcpy.AddMessage(f"Calculating pixel-to-label CSI for {nrows}x{ncols} pixels vs {n_labeled} labeled points")

    # Reshape pixel_vectors to 2D: (n_pixels, n_features)
    n_pixels = nrows * ncols
    pixel_vectors_2d = pixel_vectors.reshape(n_pixels, n_features)

    # Pre-compute valid pixel mask (pixels with all valid features)
    arcpy.AddMessage("Pre-filtering valid pixels...")
    pixel_valid = ~(np.isnan(pixel_vectors_2d) | (pixel_vectors_2d == csv_nodata))
    pixel_all_valid = np.all(pixel_valid, axis=1)  # True if ALL features are valid
    valid_indices = np.where(pixel_all_valid)[0]
    n_valid = len(valid_indices)

    arcpy.AddMessage(f"Found {n_valid} valid pixels out of {n_pixels} ({100*n_valid/n_pixels:.1f}%)")

    if n_valid == 0:
        arcpy.AddWarning("No valid pixels found!")
        return [np.full((nrows, ncols), csv_nodata, dtype=np.float64) for _ in range(n_labeled)]

    # Pre-compute valid feature masks for labeled points
    labeled_valid = []
    for i in range(n_labeled):
        mask = ~(np.isnan(labeled_features[i]) | (labeled_features[i] == csv_nodata))
        labeled_valid.append(mask)

    # Initialize output arrays - one per labeled point
    csi_arrays = []
    for i in range(n_labeled):
        csi_arrays.append(np.full((nrows, ncols), csv_nodata, dtype=np.float64))

    # Process only valid pixels
    arcpy.AddMessage("Calculating CSI for valid pixels...")
    for label_idx in range(n_labeled):

        labeled_vector = labeled_features[label_idx]
        csi_values_1d = np.full(n_pixels, csv_nodata, dtype=np.float64)

        # Process valid pixels in chunks for better memory efficiency
        chunk_size = 10000
        for chunk_start in range(0, n_valid, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_valid)
            chunk_indices = valid_indices[chunk_start:chunk_end]

            for idx in chunk_indices:
                pixel_vector = pixel_vectors_2d[idx]
                csi_value = cosine_similarity(pixel_vector, labeled_vector, csv_nodata)

                if csi_value != csv_nodata and np.isfinite(csi_value):
                    csi_values_1d[idx] = float(csi_value)

        # Reshape back to 2D
        csi_arrays[label_idx] = csi_values_1d.reshape(nrows, ncols)

    arcpy.AddMessage(f"Completed CSI calculation")
    return csi_arrays
