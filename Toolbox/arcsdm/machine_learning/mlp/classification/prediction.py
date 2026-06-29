"""Prediction and testing entry points for MLP classification."""

from typing import Any, Mapping, Optional, Sequence, Tuple

import arcpy
import numpy as np
import torch

import arcsdm.common
import arcsdm.machine_learning.general
import arcsdm.machine_learning.mlp.pytorch_utils

from arcsdm.machine_learning.mlp.classification.data import (
    classifier_prediction_rasters,
    load_classifier_metadata,
    make_classifier_prediction_loader,
    prepare_classifier_prediction_features,
    read_classifier_target_array,
    save_classifier_prediction_result,
    validate_classifier_input_rasters,
    warn_if_standardization_setting_differs,
)
from arcsdm.machine_learning.mlp.classification.metrics import log_classifier_test_metrics
from arcsdm.machine_learning.mlp.classification.model import MLPClassifierModel
from arcsdm.machine_learning.mlp.classification.types import MLPClassifierPredictionResult


def load_classifier_model(
    model_file: str,
    metadata: Mapping[str, Any],
    device: torch.device
) -> MLPClassifierModel:
    model = MLPClassifierModel(
        input_dims=int(metadata["input_dims"]),
        hidden_layers=metadata["hidden_layers"],
        last_layer=metadata["last_layer"]
    )

    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def classification_predictions_from_raw_output(
    predicted_raw: torch.Tensor,
    classification_threshold: Optional[float]
) -> Tuple[np.ndarray, np.ndarray]:
    threshold = 0.5 if classification_threshold is None else classification_threshold
    predicted_probabilities = torch.sigmoid(predicted_raw).reshape(-1).cpu().numpy()
    y_pred = (predicted_probabilities >= threshold).astype(np.int64)

    return y_pred, predicted_probabilities


def _predict_MLP_classifier(
    input_rasters: Sequence[str],
    X_nodata_value: Optional[float],
    standardize: bool,
    model_file: str,
    classification_threshold: Optional[float],
    target_array: Optional[np.ndarray] = None,
    mode_label: str = "Prediction",
    grids: Optional[Sequence[Mapping[str, Any]]] = None
) -> MLPClassifierPredictionResult:
    device = arcsdm.machine_learning.mlp.pytorch_utils.get_device()
    arcpy.AddMessage(f"Device is: {device}")

    if grids is None:
        grids = validate_classifier_input_rasters(input_rasters)

    ref_raster_path = grids[0]["path"]
    metadata = load_classifier_metadata(model_file)

    target_label_count = int(metadata.get("target_label_count", 1))
    if target_label_count != 1:
        msg = "Loaded model metadata indicates multiclass classifier. This MLP implementation supports binary classification only."
        arcpy.AddError(msg)
        raise arcsdm.machine_learning.general.MLPInputError(msg)

    warn_if_standardization_setting_differs(standardize, metadata, mode_label)
    model = load_classifier_model(model_file, metadata, device)

    X, y_true, mask_2D = prepare_classifier_prediction_features(
        input_rasters=input_rasters,
        X_nodata_value=X_nodata_value,
        target_array=target_array,
        metadata=metadata
    )
    pred_loader = make_classifier_prediction_loader(X, metadata)
    predicted = arcsdm.machine_learning.mlp.pytorch_utils.predict(device, pred_loader, model)
    predicted_raw = torch.cat(predicted)
    y_pred, predicted_probs = classification_predictions_from_raw_output(
        predicted_raw=predicted_raw,
        classification_threshold=classification_threshold
    )

    height = int(grids[0]["rows"])
    width = int(grids[0]["cols"])
    prob_raster_array, class_raster_array = classifier_prediction_rasters(
        predicted_probabilities=predicted_probs,
        y_pred=y_pred,
        height=height,
        width=width,
        nodata_mask=mask_2D
    )

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "predicted_probabilities": predicted_probs,
        "prob_raster_array": prob_raster_array,
        "class_raster_array": class_raster_array,
        "ref_raster_path": ref_raster_path,
    }


@arcsdm.common.gp_tool
def test_MLP_classifier(
    input_rasters: Sequence[str],
    X_nodata_value: Optional[float],
    standardize: bool,
    target_labels: Sequence[str],
    target_labels_attr: Optional[str],
    y_nodata_value: Optional[float],
    model_file: str,
    classification_threshold: Optional[float],
    output_raster_prob: Optional[str],
    output_raster_classified: Optional[str],
    test_metrics: Optional[str]
) -> None:
    """Test MLP classifier on input rasters with known target labels, and optionally save prediction rasters and log test metrics.
    
    Parameters:
        input_rasters: List of input raster paths to be used as features for prediction.
        X_nodata_value: NoData value to use for input features. If not provided, it will be inferred from the rasters.
        standardize: Whether to standardize input features using the same settings as during training. Must match the standardization setting used during training.
        target_labels: List of target label values corresponding to each pixel, used for testing. Must be provided if test_metrics is specified.
        target_labels_attr: Optional attribute name to read target labels from, if target_labels are stored in an attribute table of a raster. If not provided, target_labels are expected to be provided as a separate array.
        y_nodata_value: NoData value to use for target labels. If not provided, it will be inferred from the target_labels or target_labels_attr.
        model_file: Path to the trained MLP classifier model file.
        classification_threshold: Threshold to use for converting predicted probabilities to class labels. Only used for binary classification.
        output_raster_prob: Optional path to save the predicted probabilities raster. If not provided, the probabilities raster will not be saved.
        output_raster_classified: Optional path to save the classified raster. If not provided, the classified raster will not be saved.
        test_metrics: Optional string specifying which test metrics to log. If not provided, no metrics will be logged. If specified, target_labels must also be provided. Supported metrics include "accuracy", "precision", "recall", "f1", and "confusion_matrix".
    """
    arcpy.AddMessage("Starting MLP classifier test...")
    grids = validate_classifier_input_rasters(input_rasters)

    target_array = read_classifier_target_array(
        target_labels=target_labels,
        target_labels_attr=target_labels_attr,
        y_nodata_value=y_nodata_value,
        ref_raster_path=grids[0]["path"]
    )
    prediction_ret = _predict_MLP_classifier(
        input_rasters=input_rasters,
        X_nodata_value=X_nodata_value,
        standardize=standardize,
        model_file=model_file,
        classification_threshold=classification_threshold,
        target_array=target_array,
        mode_label="Test",
        grids=grids
    )

    log_classifier_test_metrics(test_metrics, prediction_ret["y_true"], prediction_ret["y_pred"])
    save_classifier_prediction_result(
        prediction_result=prediction_ret,
        output_raster_prob=output_raster_prob,
        output_raster_classified=output_raster_classified
    )

    return None


@arcsdm.common.gp_tool
def predict_MLP_classifier(
    input_rasters: Sequence[str],
    X_nodata_value: Optional[float],
    standardize: bool,
    model_file: str,
    classification_threshold: Optional[float],
    output_raster_prob: Optional[str],
    output_raster_classified: Optional[str]
) -> None:
    """Predict with MLP classifier on input rasters, and optionally save prediction rasters.
    
    Parameters:
        input_rasters: List of input raster paths to be used as features for prediction.
        X_nodata_value: NoData value to use for input features. If not provided, it will be inferred from the rasters.
        standardize: Whether to standardize input features using the same settings as during training. Must match the standardization setting used during training.
        model_file: Path to the trained MLP classifier model file.
        classification_threshold: Threshold to use for converting predicted probabilities to class labels. Only used for binary classification.
        output_raster_prob: Optional path to save the predicted probabilities raster. If not provided, the probabilities raster will not be saved.
        output_raster_classified: Optional path to save the classified raster. If not provided, the classified raster will not be saved."""
    arcpy.AddMessage("Starting prediction with classifier...")
    prediction_ret = _predict_MLP_classifier(
        input_rasters=input_rasters,
        X_nodata_value=X_nodata_value,
        standardize=standardize,
        model_file=model_file,
        classification_threshold=classification_threshold,
        mode_label="Prediction"
    )
    save_classifier_prediction_result(
        prediction_result=prediction_ret,
        output_raster_prob=output_raster_prob,
        output_raster_classified=output_raster_classified
    )

    return None


predict_with_MLP_classifier = predict_MLP_classifier
