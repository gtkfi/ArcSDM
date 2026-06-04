import json
import os
from typing import Any, Mapping, Optional, Sequence, Tuple

import arcpy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import arcsdm.machine_learning.general

from arcsdm.mlp.regression.types import MLPRegressorPredictionResult


def validate_regressor_input_rasters(input_rasters: Sequence[str]) -> Sequence[Mapping[str, Any]]:
    grids = [arcsdm.machine_learning.general.describe_raster_grid(path) for path in input_rasters]
    if not arcsdm.machine_learning.general.check_raster_grids(grids, same_extent=True):
        msg = "Input feature rasters should have same grid properties."
        arcpy.AddError(msg)
        raise arcsdm.machine_learning.general.MLPInputError(msg)

    return grids


def read_regressor_target_array(
    target_label: str,
    target_labels_attr: Optional[str],
    y_nodata_value: Optional[float],
    ref_raster_path: str
) -> np.ndarray:
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
    metadata_file = f"{os.path.splitext(model_file)[0]}.meta.json"
    if not os.path.exists(metadata_file):
        msg = f"Model metadata file not found: {metadata_file}"
        arcpy.AddError(msg)
        raise arcsdm.machine_learning.general.MLPInputError(msg)

    with open(metadata_file, "r", encoding="utf-8") as metadata_stream:
        return json.load(metadata_stream)


def warn_if_standardization_setting_differs(
    requested_standardize: bool,
    metadata: Mapping[str, Any],
    mode_label: str
) -> None:
    if bool(requested_standardize) != bool(metadata.get("standardize", False)):
        arcpy.AddWarning(
            f"{mode_label} standardize parameter differs from training metadata; using training metadata settings."
        )


def standardize_from_regressor_metadata(X: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray:
    if not bool(metadata.get("standardize", False)):
        return X

    scaler_mean = metadata.get("scaler_mean")
    scaler_scale = metadata.get("scaler_scale")

    if (scaler_mean is None) or (scaler_scale is None):
        msg = "Model metadata indicates standardization, but scaler statistics are missing."
        arcpy.AddError(msg)
        raise arcsdm.machine_learning.general.MLPInputError(msg)

    scaler_mean = np.asarray(scaler_mean, dtype=np.float32)
    scaler_scale = np.asarray(scaler_scale, dtype=np.float32)

    if X.shape[1] != scaler_mean.shape[0] or X.shape[1] != scaler_scale.shape[0]:
        msg = "Input feature count does not match scaler statistics in model metadata."
        arcpy.AddError(msg)
        raise arcsdm.machine_learning.general.MLPInputError(msg)

    arcpy.AddMessage("Data was standardized using saved training scaler metadata.")
    return (X - scaler_mean) / scaler_scale


def prepare_regressor_prediction_features(
    input_rasters: Sequence[str],
    X_nodata_value: Optional[float],
    metadata: Mapping[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
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
    dummy_labels = torch.zeros(X.shape[0], 1)
    prediction_dataset = TensorDataset(torch.from_numpy(X), dummy_labels)
    prediction_batch_size = int(metadata.get("batch_size", 1024))
    if prediction_batch_size < 1:
        prediction_batch_size = 1024

    return DataLoader(prediction_dataset, batch_size=prediction_batch_size)


def regressor_prediction_raster(
    predicted_values: np.ndarray,
    height: int,
    width: int,
    nodata_mask: np.ndarray
) -> np.ndarray:
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
    save_regressor_output_raster(
        prediction_raster_array=prediction_result["prediction_raster_array"],
        ref_raster_path=prediction_result["ref_raster_path"],
        output_raster=output_raster
    )
