"""Shared data and metadata helpers for MLP prediction workflows."""

import json
import os
from typing import Any, Mapping, Sequence

import arcpy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import arcsdm.machine_learning.general


def validate_mlp_input_rasters(input_rasters: Sequence[str]) -> Sequence[Mapping[str, Any]]:
    """Validate that all input rasters share the same grid and extent."""
    grids = [arcsdm.machine_learning.general.describe_raster_grid(path) for path in input_rasters]
    if not arcsdm.machine_learning.general.check_raster_grids(grids, same_extent=True):
        msg = "Input feature rasters should have same grid properties."
        arcpy.AddError(msg)
        raise arcsdm.machine_learning.general.MLPInputError(msg)

    return grids


def load_mlp_metadata(model_file: str) -> Mapping[str, Any]:
    """Load JSON metadata stored alongside a trained MLP model file."""
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
    """Warn when runtime standardization choice differs from training metadata."""
    if bool(requested_standardize) != bool(metadata.get("standardize", False)):
        arcpy.AddWarning(
            f"{mode_label} standardize parameter differs from training metadata; using training metadata settings."
        )


def standardize_from_mlp_metadata(X: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray:
    """Apply saved scaler statistics from metadata to feature matrix X."""
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


def make_mlp_prediction_loader(X: np.ndarray, metadata: Mapping[str, Any]) -> DataLoader:
    """Build a DataLoader for batch prediction using metadata batch size."""
    dummy_labels = torch.zeros(X.shape[0], 1)
    prediction_dataset = TensorDataset(torch.from_numpy(X), dummy_labels)
    prediction_batch_size = int(metadata.get("batch_size", 1024))
    if prediction_batch_size < 1:
        prediction_batch_size = 1024

    return DataLoader(prediction_dataset, batch_size=prediction_batch_size)
