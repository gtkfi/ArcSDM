import os
import arcpy
import numpy as np
from typing import List, Optional
from arcsdm.csi.core.calculate_evidence_matrix import calculate_evidence_matrix
from arcsdm.csi.core.corner_csi import calculate_corner_csi
from arcsdm.csi.core.centroid_csi import calculate_centroid_to_centroid_csi
from arcsdm.csi.analysis.class_centroids import calculate_class_centroids
from arcsdm.csi.analysis.pixel_to_label_csi import pixel_to_label_csi

from arcsdm.csi.helpers.save_results import save_csv_results
from arcsdm.csi.helpers.rows_with_labels import rows_with_labels
from arcsdm.csi.helpers.load_data import (load_labeled_data, load_raster_data)

def calculation(
    selected_label_field: Optional[str],
    labelled_path: str,
    label_field_names: List[str],
    coordinate_field_names: List[str],
    feature_field_names: List[str],
    rasters_list: List[str],
    evidence_type: Optional[str],
    csv_nodata: float,
    out_labelled_pairwise_csv: str,
    out_evidence_matrix_csv: Optional[str],
    out_raster_folder: Optional[str],
    out_class_centroid: Optional[str] = None,
    out_centroid_csi_csv: Optional[str] = None,
):
    """Perform CSI calculation workflow."""

    if coordinate_field_names is None or len(coordinate_field_names) != 2:
        arcpy.AddError("Please provide exactly two coordinate field names")
        return

    # Load labeled data
    all_df, feature_fields, has_geometry = load_labeled_data(
        labelled_path, label_field_names, feature_field_names, coordinate_field_names
    )

    if all_df is None:
        arcpy.AddError("Failed to load labeled data - cannot proceed with analysis.")
        return

    # Filter to only labeled points
    apply_nodata = os.path.splitext(labelled_path)[1] in [".csv", ".txt"]
    label_mask = rows_with_labels(all_df, label_field_names, apply_nodata, csv_nodata)
    labeled_df = all_df.loc[label_mask].reset_index(drop=True)

    # Further filter by selected label field
    if selected_label_field and selected_label_field in labeled_df.columns:
        selected_mask = labeled_df[selected_label_field].notna()
        labeled_df = labeled_df.loc[selected_mask].reset_index(drop=True)
        arcpy.AddMessage(f"Filtered to {len(labeled_df)} points with valid {selected_label_field}")

    arcpy.AddMessage(f"Using {len(labeled_df)} labeled points for analysis")

    if len(labeled_df) == 0:
        arcpy.AddError("No labeled rows found - cannot proceed with analysis.")
        return

    # Calculate corner CSI matrix
    arcpy.AddMessage("\nCorner CSI Matrix Calculation")
    corner_matrix = calculate_corner_csi(labeled_df, feature_fields, csv_nodata)

    # Calculate class centroids
    centroids_df = calculate_class_centroids(
        labeled_df, feature_fields, label_field_names, csv_nodata
    )

    # Calculate centroid-to-centroid CSI
    centroid_csi_matrix = np.array([])
    if len(centroids_df) > 0:
        centroid_csi_matrix = calculate_centroid_to_centroid_csi(
            centroids_df, feature_fields, csv_nodata
        )

    coord_1, coord_2 = coordinate_field_names
    # PART 2: Pixel-to-Label CSI (if rasters provided and output folder specified)
    if evidence_type == "Raster" and rasters_list and out_raster_folder:
        # Filter out coordinate columns from feature fields
        feature_fields_only = [f for f in feature_fields if f not in (coord_1, coord_2, 'SHAPE@XY')]

        success = pixel_to_label_csi(
            labeled_df, feature_fields_only, rasters_list, out_raster_folder,
            label_field_names, csv_nodata
        )
        if not success:
            arcpy.AddWarning("Workflow failed, continuing with Part 1 results only")

    # Calculate evidence matrix for CSV output (if needed)
    evidence_results = {}
    if evidence_type == "Raster" and rasters_list and out_evidence_matrix_csv:
        arcpy.AddMessage("\nCalculating evidence matrix for CSV output...")
        raster_data = load_raster_data(rasters_list)
        if raster_data:
            evidence_results = calculate_evidence_matrix(
                all_df,
                labeled_df,
                feature_fields,
                raster_data,
                coordinate_field_names,
                has_geometry,
                csv_nodata
            )

    # Save CSV results
    arcpy.AddMessage("\nSaving CSV results...")

    save_csv_results(
        corner_matrix,
        evidence_results,
        centroids_df,
        centroid_csi_matrix,
        out_labelled_pairwise_csv,
        out_evidence_matrix_csv,
        out_class_centroid,
        out_centroid_csi_csv
    )

    arcpy.AddMessage(f"\nCSI Analysis completed successfully!")
    arcpy.AddMessage(f"Corner matrix shape: {corner_matrix.shape}")
    if evidence_type == "Raster" and rasters_list and out_raster_folder:
        arcpy.AddMessage(f"Pixel-to-label CSI rasters created")
