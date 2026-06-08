"""Data I/O, preprocessing, and raster output helpers for MLP regression."""

from typing import Any, Mapping, Optional, Sequence, Tuple

import arcpy
import numpy as np
from torch.utils.data import DataLoader

import arcsdm.machine_learning.general
from arcsdm.machine_learning.mlp.common.data import (
    load_mlp_metadata,
    make_mlp_prediction_loader,
    standardize_from_mlp_metadata,
    validate_mlp_input_rasters,
    warn_if_standardization_setting_differs as warn_if_standardization_setting_differs_common,
)

from arcsdm.machine_learning.mlp.regression.types import MLPRegressorPredictionResult


def validate_regressor_input_rasters(input_rasters: Sequence[str]) -> Sequence[Mapping[str, Any]]:
    """Validate regressor input rasters share a compatible grid."""
    return validate_mlp_input_rasters(input_rasters)


def read_regressor_target_array(
    target_label: str,
    target_labels_attr: Optional[str],
    y_nodata_value: Optional[float],
    ref_raster_path: str
) -> np.ndarray:
    """Read and rasterize regressor targets into a single 2D array."""
    target_desc = arcpy.Describe(target_label).dataType
    if target_desc in ["FeatureLayer", "FeatureClass", "ShapeFile"]:
        return arcsdm.machine_learning.general.rasterize_vector_to_array(
            vector_path=target_label,
            ref_path=ref_raster_path,
            value_field=target_labels_attr,
            const=1,
            classification=False
        )

    if target_desc in ["RasterLayer", "RasterDataset", "RasterBand"]:
        label_bands = arcsdm.machine_learning.general.raster_to_band_arrays(target_label)
        if len(label_bands) != 1:
            msg = "Target label raster must have exactly one band."
            arcpy.AddError(msg)
            raise arcsdm.machine_learning.general.MLPInputError(msg)

        y = label_bands[0]
        if y_nodata_value is not None and not arcsdm.machine_learning.general.is_nan_like(y_nodata_value):
            arcsdm.machine_learning.general.apply_explicit_nodata_inplace(y, y_nodata_value, np.float32)
        return y

    msg = f"Unsupported target label data type: {target_desc}"
    arcpy.AddError(msg)
    raise arcsdm.machine_learning.general.MLPInputError(msg)


def load_regressor_metadata(model_file: str) -> Mapping[str, Any]:
    """Load regressor model metadata from sidecar JSON."""
    return load_mlp_metadata(model_file)


def warn_if_standardization_setting_differs(
    requested_standardize: bool,
    metadata: Mapping[str, Any],
    mode_label: str
) -> None:
    """Warn if prediction-time standardization setting differs from training."""
    warn_if_standardization_setting_differs_common(requested_standardize, metadata, mode_label)


def standardize_from_regressor_metadata(X: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray:
    """Standardize regressor features using scaler stats stored in metadata."""
    return standardize_from_mlp_metadata(X, metadata)


def prepare_regressor_prediction_features(
    input_rasters: Sequence[str],
    X_nodata_value: Optional[float],
    metadata: Mapping[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare flattened features and nodata mask for regression prediction."""
    raster_arrays = arcsdm.machine_learning.general.read_raster_bands(
        raster_files=input_rasters,
        nodata_value=X_nodata_value
    )

    mask_2D = arcsdm.machine_learning.general.get_nodata_mask(raster_arrays)
    valid = ~mask_2D.ravel()

    X = np.column_stack([arr.ravel() for arr in raster_arrays])
    X = X[valid]
    X = standardize_from_regressor_metadata(X, metadata)

    input_dims = int(metadata["input_dims"])
    if X.shape[1] != input_dims:
        msg = f"Input feature count mismatch. Model expects {input_dims}, got {X.shape[1]}."
        arcpy.AddError(msg)
        raise arcsdm.machine_learning.general.MLPInputError(msg)

    return X, mask_2D


def make_regressor_prediction_loader(X: np.ndarray, metadata: Mapping[str, Any]) -> DataLoader:
    """Create regressor prediction loader with metadata-driven batch size."""
    return make_mlp_prediction_loader(X, metadata)


def regressor_prediction_raster(
    predicted_values: np.ndarray,
    height: int,
    width: int,
    nodata_mask: np.ndarray
) -> np.ndarray:
    """Reconstruct flat regression predictions into raster-shaped output."""
    return arcsdm.machine_learning.general.reshape_predictions(
        predictions=predicted_values,
        height=height,
        width=width,
        nodata_mask=nodata_mask
    )


def save_regressor_output_raster(
    prediction_raster_array: np.ndarray,
    ref_raster_path: str,
    output_raster: str
) -> None:
    """Write predicted regression values to an output raster dataset."""
    desc = arcpy.Describe(ref_raster_path)
    lower_left = arcpy.Point(desc.extent.XMin, desc.extent.YMin)
    x_cell_size = desc.meanCellWidth
    y_cell_size = desc.meanCellHeight

    out_ras = arcpy.NumPyArrayToRaster(prediction_raster_array, lower_left, x_cell_size, y_cell_size)
    out_ras.save(output_raster)
    arcpy.AddMessage(f"Saved predicted values raster to {output_raster}")


def save_regressor_prediction_result(
    prediction_result: MLPRegressorPredictionResult,
    output_raster: str
) -> None:
    """Persist regression prediction outputs to disk."""
    save_regressor_output_raster(
        prediction_raster_array=prediction_result["prediction_raster_array"],
        ref_raster_path=prediction_result["ref_raster_path"],
        output_raster=output_raster
    )
