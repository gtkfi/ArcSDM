"""Data I/O, preprocessing, and raster output helpers for MLP classification."""

from typing import Any, Mapping, Optional, Sequence, Tuple

import arcpy
import numpy as np
from torch.utils.data import DataLoader

import arcsdm.machine_learning.general
from Toolbox.arcsdm.machine_learning.mlp.common.data import (
    load_mlp_metadata,
    make_mlp_prediction_loader,
    standardize_from_mlp_metadata,
    validate_mlp_input_rasters,
    warn_if_standardization_setting_differs as warn_if_standardization_setting_differs_common,
)

from Toolbox.arcsdm.machine_learning.mlp.classification.types import MLPClassifierPredictionResult


def read_classifier_target_array(
    target_labels: Sequence[str],
    target_labels_attr: Optional[str],
    y_nodata_value: Optional[float],
    ref_raster_path: str
) -> np.ndarray:
    """Read and rasterize classifier targets into a single 2D label array."""
    if len(target_labels) > 1:
        label_arrays = []
        for i in range(len(target_labels)):
            label_arrays.append(
                arcsdm.machine_learning.general.rasterize_vector_to_array(
                    vector_path=target_labels[i],
                    ref_path=ref_raster_path,
                    value_field=None,
                    const=i + 1
                )
            )
        return arcsdm.machine_learning.general.pick_value(label_arrays, prefer="first")

    target_desc = arcpy.Describe(target_labels[0]).dataType
    if target_desc in ["FeatureLayer", "FeatureClass", "ShapeFile"]:
        return arcsdm.machine_learning.general.rasterize_vector_to_array(
            vector_path=target_labels[0],
            ref_path=ref_raster_path,
            value_field=target_labels_attr,
            const=1
        )
    if target_desc in ["RasterLayer", "RasterDataset", "RasterBand"]:
        label_bands = arcsdm.machine_learning.general.raster_to_band_arrays(target_labels[0])
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


def save_classifier_output_rasters(
    prob_raster_array: np.ndarray,
    class_raster_array: np.ndarray,
    ref_raster_path: str,
    output_raster_prob: Optional[str],
    output_raster_classified: Optional[str]
) -> None:
    """Write probability and classified output rasters using a reference raster grid."""
    desc = arcpy.Describe(ref_raster_path)
    lower_left = arcpy.Point(desc.extent.XMin, desc.extent.YMin)
    x_cell_size = desc.meanCellWidth
    y_cell_size = desc.meanCellHeight

    if output_raster_prob:
        out_prob = arcpy.NumPyArrayToRaster(prob_raster_array, lower_left, x_cell_size, y_cell_size)
        out_prob.save(output_raster_prob)
        arcpy.AddMessage(f"Saved probability raster to {output_raster_prob}")

    if output_raster_classified:
        out_cls = arcpy.NumPyArrayToRaster(class_raster_array, lower_left, x_cell_size, y_cell_size)
        out_cls.save(output_raster_classified)
        arcpy.AddMessage(f"Saved classified raster to {output_raster_classified}")


def validate_classifier_input_rasters(input_rasters: Sequence[str]) -> Sequence[Mapping[str, Any]]:
    """Validate classifier input rasters share a compatible grid."""
    return validate_mlp_input_rasters(input_rasters)


def load_classifier_metadata(model_file: str) -> Mapping[str, Any]:
    """Load classifier model metadata from sidecar JSON."""
    return load_mlp_metadata(model_file)


def warn_if_standardization_setting_differs(
    requested_standardize: bool,
    metadata: Mapping[str, Any],
    mode_label: str
) -> None:
    """Warn if prediction-time standardization setting differs from training."""
    warn_if_standardization_setting_differs_common(requested_standardize, metadata, mode_label)


def standardize_from_classifier_metadata(X: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray:
    """Standardize classifier features using scaler stats stored in metadata."""
    return standardize_from_mlp_metadata(X, metadata)


def get_classifier_targets(
    target_array: Optional[np.ndarray],
    valid_mask: np.ndarray,
    metadata: Mapping[str, Any]
) -> Optional[np.ndarray]:
    """Map raw target values to model class indices using metadata labels."""
    if target_array is None:
        return None

    y_raw = target_array.ravel()[valid_mask]
    known_labels = [float(value) for value in metadata.get("unique_labels", [])]
    label_to_index = {label: idx for idx, label in enumerate(known_labels)}
    try:
        return np.asarray([label_to_index[float(label)] for label in y_raw], dtype=np.int64)
    except KeyError as exc:
        msg = f"Target labels contain unseen class: {exc}."
        arcpy.AddError(msg)
        raise arcsdm.machine_learning.general.MLPInputError(msg)


def make_classifier_prediction_loader(X: np.ndarray, metadata: Mapping[str, Any]) -> DataLoader:
    """Create classifier prediction loader with metadata-driven batch size."""
    return make_mlp_prediction_loader(X, metadata)


def prepare_classifier_prediction_features(
    input_rasters: Sequence[str],
    X_nodata_value: Optional[float],
    target_array: Optional[np.ndarray],
    metadata: Mapping[str, Any]
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Prepare flattened features, optional mapped targets, and nodata mask for prediction."""
    raster_arrays = arcsdm.machine_learning.general.read_raster_bands(
        raster_files=input_rasters,
        nodata_value=X_nodata_value
    )

    mask_inputs = raster_arrays + ([target_array] if target_array is not None else [])
    mask_2D = arcsdm.machine_learning.general.get_nodata_mask(mask_inputs)
    valid = ~mask_2D.ravel()

    X = np.column_stack([arr.ravel() for arr in raster_arrays])
    X = X[valid]
    X = standardize_from_classifier_metadata(X, metadata)

    input_dims = int(metadata["input_dims"])
    if X.shape[1] != input_dims:
        msg = f"Input feature count mismatch. Model expects {input_dims}, got {X.shape[1]}."
        arcpy.AddError(msg)
        raise arcsdm.machine_learning.general.MLPInputError(msg)

    y_true = get_classifier_targets(target_array, valid, metadata)
    return X, y_true, mask_2D


def classifier_prediction_rasters(
    predicted_probabilities: np.ndarray,
    y_pred: np.ndarray,
    height: int,
    width: int,
    nodata_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct probability and class outputs back to raster-shaped arrays."""
    if predicted_probabilities.ndim == 1:
        prob_raster_array = arcsdm.machine_learning.general.reshape_predictions(
            predictions=predicted_probabilities,
            height=height,
            width=width,
            nodata_mask=nodata_mask
        )
    elif predicted_probabilities.ndim == 2:
        class_count = predicted_probabilities.shape[1]
        full_predictions = np.full((class_count, width * height), np.nan, dtype=predicted_probabilities.dtype)
        full_predictions[:, ~nodata_mask.ravel()] = predicted_probabilities.T
        prob_raster_array = full_predictions.reshape((class_count, height, width))
    else:
        msg = f"Unexpected prediction probability shape: {predicted_probabilities.shape}"
        arcpy.AddError(msg)
        raise arcsdm.machine_learning.general.MLPInputError(msg)

    class_raster_array = arcsdm.machine_learning.general.reshape_predictions(
        predictions=y_pred.astype(np.float32),
        height=height,
        width=width,
        nodata_mask=nodata_mask
    )
    return prob_raster_array, class_raster_array


def save_classifier_prediction_result(
    prediction_result: MLPClassifierPredictionResult,
    output_raster_prob: Optional[str],
    output_raster_classified: Optional[str]
) -> None:
    """Persist classifier prediction outputs to raster datasets."""
    save_classifier_output_rasters(
        prob_raster_array=prediction_result["prob_raster_array"],
        class_raster_array=prediction_result["class_raster_array"],
        ref_raster_path=prediction_result["ref_raster_path"],
        output_raster_prob=output_raster_prob,
        output_raster_classified=output_raster_classified
    )
