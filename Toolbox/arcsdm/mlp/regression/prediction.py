from typing import Any, Mapping, Optional, Sequence

import arcpy
import numpy as np
import torch

import arcsdm.common
import arcsdm.machine_learning.pytorch_utils

from arcsdm.mlp.regression.data import (
    load_regressor_metadata,
    make_regressor_prediction_loader,
    prepare_regressor_prediction_features,
    read_regressor_target_array,
    regressor_prediction_raster,
    save_regressor_prediction_result,
    validate_regressor_input_rasters,
    warn_if_standardization_setting_differs,
)
from arcsdm.mlp.regression.metrics import log_regression_test_metrics
from arcsdm.mlp.regression.model import MLPRegressorModel
from arcsdm.mlp.regression.types import MLPRegressorPredictionResult


def load_regressor_model(
    model_file: str,
    metadata: Mapping[str, Any],
    device: torch.device
) -> MLPRegressorModel:
    model = MLPRegressorModel(
        input_dims=int(metadata["input_dims"]),
        hidden_layers=metadata["hidden_layers"],
        last_layer=metadata["last_layer"]
    )

    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _predict_MLP_regressor(
    input_rasters: Sequence[str],
    X_nodata_value: Optional[float],
    standardize: bool,
    model_file: str,
    mode_label: str = "Prediction",
    grids: Optional[Sequence[Mapping[str, Any]]] = None
) -> MLPRegressorPredictionResult:
    device = arcsdm.machine_learning.pytorch_utils.get_device()
    arcpy.AddMessage(f"Device is: {device}")

    if grids is None:
        grids = validate_regressor_input_rasters(input_rasters)

    ref_raster_path = grids[0]["path"]
    metadata = load_regressor_metadata(model_file)
    warn_if_standardization_setting_differs(standardize, metadata, mode_label)
    model = load_regressor_model(model_file, metadata, device)

    X, mask_2D = prepare_regressor_prediction_features(
        input_rasters=input_rasters,
        X_nodata_value=X_nodata_value,
        metadata=metadata
    )
    prediction_loader = make_regressor_prediction_loader(X, metadata)
    predicted = arcsdm.machine_learning.pytorch_utils.predict(device, prediction_loader, model)
    predicted_values = torch.cat(predicted).reshape(-1).cpu().numpy()

    height = int(grids[0]["rows"])
    width = int(grids[0]["cols"])
    prediction_raster_array = regressor_prediction_raster(
        predicted_values=predicted_values,
        height=height,
        width=width,
        nodata_mask=mask_2D
    )

    return {
        "predictions": predicted_values,
        "prediction_raster_array": prediction_raster_array,
        "ref_raster_path": ref_raster_path,
    }


@arcsdm.common.gp_tool
def test_MLP_regressor(
    input_rasters,
    X_nodata_value,
    standardize,
    target_labels,
    target_labels_attr,
    y_nodata_value,
    model_file,
    output_raster,
    test_metrics
) -> None:
    """Run MLP regressor test by comparing predictions to target labels and logging metrics.
    Saves a raster of predictions to output_raster.
    Parameters:
        input_rasters: list of paths to rasters to use as input features for prediction
        X_nodata_value: nodata value to use for input features during prediction (if not specified, will use nodata value from input rasters)
        standardize: whether to standardize input features using mean and std from training (must match whether standardization was used during training)
        target_labels: list of target label names to use for testing (only the first one will be used)
        target_labels_attr: attribute in the raster attribute table that contains the target labels (if not specified, will use the raster cell values as target labels)
        y_nodata_value: nodata value to use for target labels during testing (if not specified, will use nodata value from target raster)
        model_file: path to the trained MLP regressor model file
        output_raster: path to save the raster of predictions
        test_metrics: comma-separated list of regression metrics to calculate and log (e.g. "MSE,R2")
    """
    arcpy.AddMessage("Starting MLP regressor test...")
    grids = validate_regressor_input_rasters(input_rasters)

    target_array = read_regressor_target_array(
        target_label=target_labels[0],
        target_labels_attr=target_labels_attr,
        y_nodata_value=y_nodata_value,
        ref_raster_path=grids[0]["path"]
    )
    prediction_result = _predict_MLP_regressor(
        input_rasters=input_rasters,
        X_nodata_value=X_nodata_value,
        standardize=standardize,
        model_file=model_file,
        mode_label="Test",
        grids=grids
    )

    prediction_raster = prediction_result["prediction_raster_array"]
    valid = ~(np.isnan(prediction_raster) | np.isnan(target_array))
    if not valid.any():
        msg = "No overlapping valid target and prediction cells were available for regression testing."
        arcpy.AddError(msg)
        raise arcsdm.machine_learning.general.MLPInputError(msg)

    y_true = target_array[valid].reshape(-1)
    y_pred = prediction_raster[valid].reshape(-1)
    log_regression_test_metrics(test_metrics, y_true, y_pred)
    save_regressor_prediction_result(
        prediction_result=prediction_result,
        output_raster=output_raster
    )
    return None


@arcsdm.common.gp_tool
def predict_with_MLP_regressor(
    input_rasters,
    X_nodata_value,
    standardize,
    model_file,
    output_raster
):
    arcpy.AddMessage("Starting prediction with regressor...")
    prediction_result = _predict_MLP_regressor(
        input_rasters=input_rasters,
        X_nodata_value=X_nodata_value,
        standardize=standardize,
        model_file=model_file,
        mode_label="Prediction"
    )
    save_regressor_prediction_result(
        prediction_result=prediction_result,
        output_raster=output_raster
    )

    return None
