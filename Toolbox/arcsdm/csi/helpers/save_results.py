import re
import os
import arcpy
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

def save_csv_results(
    corner_matrix: np.ndarray,
    evidence_results: Dict[str, np.ndarray],
    centroids_df: pd.DataFrame,
    centroid_csi_matrix: np.ndarray,
    out_labelled_pairwise_csv: Optional[str],
    out_evidence_matrix_csv: Optional[str],
    out_centroids_csv: Optional[str],
    out_centroid_csi_csv: Optional[str]
) -> None:
    """Save results to CSV files"""
    try:
        # Save corner CSI matrix
        if out_labelled_pairwise_csv and corner_matrix is not None and len(corner_matrix) > 0:
            outdir = os.path.dirname(out_labelled_pairwise_csv)
            if outdir:
                os.makedirs(outdir, exist_ok=True)
            corner_df = pd.DataFrame(corner_matrix)
            corner_df.index = [f"Point_{i+1}" for i in range(len(corner_matrix))]
            corner_df.columns = [f"Point_{i+1}" for i in range(len(corner_matrix))]
            corner_df.to_csv(out_labelled_pairwise_csv)
            arcpy.AddMessage(f"Saved corner CSI matrix: {out_labelled_pairwise_csv}")

        # Save evidence matrix
        if out_evidence_matrix_csv and evidence_results and 'evidence_matrix' in evidence_results:
            outdir = os.path.dirname(out_evidence_matrix_csv)
            if outdir:
                os.makedirs(outdir, exist_ok=True)
            evidence_matrix = evidence_results['evidence_matrix']
            evidence_df = pd.DataFrame(evidence_matrix)
            evidence_df.index = [f"Point_{i+1}" for i in range(len(evidence_df))]
            evidence_df.columns = [f"Raster_{i+1}" for i in range(evidence_df.shape[1])]
            evidence_df.to_csv(out_evidence_matrix_csv)
            arcpy.AddMessage(f"Saved evidence matrix ({evidence_df.shape}): {out_evidence_matrix_csv}")

        # Save class centroids
        if out_centroids_csv and centroids_df is not None and len(centroids_df) > 0:
            outdir = os.path.dirname(out_centroids_csv)
            if outdir:
                os.makedirs(outdir, exist_ok=True)
            centroids_df.to_csv(out_centroids_csv, index=False)
            arcpy.AddMessage(f"Saved class centroids: {out_centroids_csv}")

        # Save centroid CSI matrix
        if out_centroid_csi_csv and centroid_csi_matrix is not None and len(centroid_csi_matrix) > 0:
            outdir = os.path.dirname(out_centroid_csi_csv)
            if outdir:
                os.makedirs(outdir, exist_ok=True)
            centroid_csi_df = pd.DataFrame(centroid_csi_matrix)
            if 'class' in centroids_df.columns:
                class_labels = centroids_df['class'].values
                centroid_csi_df.index = [f"Class_{c}" for c in class_labels]
                centroid_csi_df.columns = [f"Class_{c}" for c in class_labels]
            centroid_csi_df.to_csv(out_centroid_csi_csv)
            arcpy.AddMessage(f"Saved centroid-to-centroid CSI matrix: {out_centroid_csi_csv}")

    except Exception as e:
        arcpy.AddError(f"Error saving CSV results: {e}")


def save_csi_rasters(
    csi_arrays: List[np.ndarray],
    labeled_df: pd.DataFrame,
    reference_props: Dict[str, Any],
    out_raster_folder: str,
    label_field_names: List[str],
    csv_nodata: float
) -> None:
    """
    Save CSI arrays as raster files - one per labeled point.
    Direct array to raster conversion.
    """
    # Always create rasters in the output folder, warn if label columns are missing or invalid
    if not out_raster_folder or not isinstance(out_raster_folder, str) or not out_raster_folder.strip():
        arcpy.AddWarning("Output raster folder is not specified or invalid. Attempting to save anyway.")

    if labeled_df.empty or len(label_field_names) == 0 or not all(col in labeled_df.columns for col in label_field_names):
        arcpy.AddWarning("Label columns are missing or invalid. Raster filenames may be generic.")

    os.makedirs(out_raster_folder, exist_ok=True)

    n_labeled = len(csi_arrays)
    arcpy.AddMessage(f"Saving {n_labeled} CSI rasters to: {out_raster_folder}")

    # Set up raster environment
    extent = reference_props['extent']
    cell_width = reference_props['cell_width']
    cell_height = reference_props['cell_height']
    spatial_ref = reference_props['spatial_ref']

    for label_idx, csi_array in enumerate(csi_arrays):
        try:
            # Create label identifier
            label_id = f"label_{label_idx + 1}"

            # Try to get a meaningful label from the data if available
            if label_field_names and len(label_field_names) > 0:
                label_col = label_field_names[0]
                if label_col in labeled_df.columns:
                    label_value = labeled_df.iloc[label_idx][label_col]
                    if pd.notna(label_value) and str(label_value).strip():
                        # Sanitize label value for filename
                        safe_label = re.sub(r"[^A-Za-z0-9_]+", "_", str(label_value))[:20]
                        label_id = f"label_{label_idx + 1}_{safe_label}"

            # Create output filename
            output_filename = f"csi_{label_id}.tif"
            output_path = os.path.join(out_raster_folder, output_filename)

            # Ensure array is the right type and format
            if csi_array.dtype != np.float64:
                csi_array = csi_array.astype(np.float64)

            # Make sure array is contiguous
            if not csi_array.flags['C_CONTIGUOUS']:
                csi_array = np.ascontiguousarray(csi_array)

            # Replace csv_nodata with np.nan for proper NoData handling
            clean_array = np.where(csi_array == csv_nodata, np.nan, csi_array)

            # Create raster from array - NO INTERPOLATION
            lower_left = arcpy.Point(extent.XMin, extent.YMin)
            raster = arcpy.NumPyArrayToRaster(
                clean_array, 
                lower_left, 
                cell_width, 
                cell_height,
                value_to_nodata=np.nan
            )

            # Set spatial reference
            arcpy.management.DefineProjection(raster, spatial_ref)

            # Save raster
            raster.save(output_path)

            # Build statistics and pyramids
            arcpy.management.CalculateStatistics(output_path)
            arcpy.management.BuildPyramids(output_path)

            # Calculate statistics for reporting
            valid_pixels = np.sum(~np.isnan(clean_array))
            if valid_pixels > 0:
                min_csi = np.nanmin(clean_array)
                max_csi = np.nanmax(clean_array)
                mean_csi = np.nanmean(clean_array)
                arcpy.AddMessage(f"Saved {output_filename}: {valid_pixels} valid pixels, "
                               f"CSI range [{min_csi:.4f}, {max_csi:.4f}], mean {mean_csi:.4f}")
            else:
                arcpy.AddWarning(f"Saved {output_filename}: No valid pixels (all NoData)")

        except Exception as e:
            arcpy.AddError(f"Error saving raster for label {label_idx + 1}: {e}")
            import traceback
            arcpy.AddError(traceback.format_exc())
