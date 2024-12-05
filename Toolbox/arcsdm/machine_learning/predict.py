import arcpy
import numpy as np
import pandas as pd
from typing import Tuple, Union
from sklearn.base import BaseEstimator, is_classifier
from tensorflow import keras


def predict_classifier(
    data: Union[np.ndarray, pd.DataFrame],
    model: Union[BaseEstimator, keras.Model],
    classification_threshold: float = 0.5,
    include_probabilities: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Predict with a trained classifier model.

    Args:
        data: Data used to make predictions.
        model: Trained classifier or regressor. Can be any machine learning model trained with
            EIS Toolkit (Sklearn and Keras models).
        classification_threshold: Threshold for classifying based on probabilities. Only used for
            binary classification. Defaults to 0.5.
        include_probabilities: If the probability array should be returned too. Defaults to True.

    Returns:
        Predicted labels and optionally predicted probabilities as one-dimensional arrays by a classifier model.

    Raises:
        arcpy.AddError: Input model is not a classifier model.
    """
    if isinstance(model, keras.Model):
        probabilities = model.predict(data).astype(np.float32)
        if probabilities.shape[1] == 1:  # Binary classification
            probabilities = probabilities.squeeze()
            labels = (probabilities >= classification_threshold).astype(np.float32)
        else:  # Multiclass classification
            labels = probabilities.argmax(axis=-1).astype(np.float32)
        if include_probabilities:
            return labels, probabilities
        else:
            return labels
    elif isinstance(model, BaseEstimator):
        if not is_classifier(model):
            arcpy.AddError(f"Expected a classifier model: {type(model)}.")
            raise arcpy.ExecuteError
        probabilities = model.predict_proba(data).astype(np.float32)
        if probabilities.shape[1] == 2:  # Binary classification
            probabilities = probabilities[:, 1]
            labels = (probabilities >= classification_threshold).astype(np.float32)
        else:  # Multiclass classification
            labels = probabilities.argmax(axis=-1).astype(np.float32)
        if include_probabilities:
            return labels, probabilities
        else:
            return labels
    else:
        arcpy.AddError(f"Model type not recognized: {type(model)}.")
        raise arcpy.ExecuteError


def predict_regressor(
    data: Union[np.ndarray, pd.DataFrame],
    model: Union[BaseEstimator, keras.Model],
) -> np.ndarray:
    """
    Predict with a trained regressor model.

    Args:
        data: Data used to make predictions.
        model: Trained classifier or regressor. Can be any machine learning model trained with
            EIS Toolkit (Sklearn and Keras models).

    Returns:
        Regression model prediction array.

    Raises:
        arcpy.AddError: Input model is not a regressor model or is not recognized.
        arcpy.AddError: Input models does not have single output unit.
    """
    if isinstance(model, BaseEstimator):
        if is_classifier(model):
            arcpy.AddError(f"Expected a regressor model: {type(model)}.")
            raise arcpy.ExecuteError
    elif isinstance(model, keras.Model):
        if not model.output_shape[-1] == 1:
            arcpy.AddError(f"Expected a single output unit for a regressor model: {type(model)}.")
            raise arcpy.ExecuteError
    else:
        arcpy.AddError(f"Model type not recognized: {type(model)}.")
        raise arcpy.ExecuteError

    result = model.predict(data)
    if result.ndim == 2:
        result = result.squeeze()
    return result