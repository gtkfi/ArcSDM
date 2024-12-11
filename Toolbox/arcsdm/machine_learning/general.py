"""
Original source code from https://github.com/GispoCoding/eis_toolkit
"""

import os
import arcpy
import arcpy.ia
import joblib
import pandas as pd
from numbers import Number
from pathlib import Path
import numpy as np
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, train_test_split
from tensorflow import keras

from arcsdm.evaluation.scoring import score_predictions
from utils.rasterize import rasterize_vector

SPLIT = "split"
KFOLD_CV = "kfold_cv"
SKFOLD_CV = "skfold_cv"
LOO_CV = "loo_cv"
NO_VALIDATION = "none"


def save_model(model: Union[BaseEstimator, keras.Model], path: Path) -> None:
    """
    Save a trained Sklearn model to a .joblib file.

    Args:
        model: Trained model.
        path: Path where the model should be saved. Include the .joblib file extension.
    """
    joblib.dump(model, path + ".joblib")


def load_model(path: Path) -> Union[BaseEstimator, keras.Model]:
    """
    Load a Sklearn model from a .joblib file.

    Args:
        path: Path from where the model should be loaded. Include the .joblib file extension.

    Returns:
        Loaded model.
    """
    return joblib.load(path)


def split_data(
    *data: Union[np.ndarray, pd.DataFrame, sparse.csr.csr_matrix, List[Number]],
    split_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True,
) -> List[Union[np.ndarray, pd.DataFrame, sparse.csr.csr_matrix, List[Number]]]:
    """
    Split data into two parts. Can be used for train-test or train-validation splits.

    For more guidance, read documentation of sklearn.model_selection.train_test_split:
    (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).

    Args:
        *data: Data to be split. Multiple datasets can be given as input (for example X and y),
            but they need to have the same length. All datasets are split into two and the parts returned
            (for example X_train, X_test, y_train, y_test).
        split_size: The proportion of the second part of the split. Typically this is the size of test/validation
            part. The first part will be complemental proportion. For example, if split_size = 0.2, the first part
            will have 80% of the data and the second part 20% of the data. Defaults to 0.2.
        random_state: Seed for random number generation. Defaults to None.
        shuffle: If data is shuffled before splitting. Defaults to True.

    Returns:
        List containing splits of inputs (two outputs per input).
    """

    if not (0 < split_size < 1):
        arcpy.AddError("Split size must be more than 0 and less than 1.")
        raise arcpy.ExecuteError

    split_data = train_test_split(*data, test_size=split_size, random_state=random_state, shuffle=shuffle)

    return split_data


def reshape_predictions(
    predictions: np.ndarray, height: int, width: int, nodata_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Reshape 1D prediction ouputs into 2D Numpy array.

    The output is ready to be visualized and saved as a raster.

    Args:
        predictions: A 1D Numpy array with raw prediction data from `predict` function.
        height: Height of the output array
        width: Width of the output array
        nodata_mask: Nodata mask used to reconstruct original shape of data. This is the same mask
            applied to data before predicting to remove nodata. If any nodata was removed
            before predicting, this mask is required to reconstruct the original shape of data.
            Defaults to None.

    Returns:
        Predictions as a 2D Numpy array in the original array shape.
    """
    full_predictions_array = np.full(width * height, np.nan, dtype=predictions.dtype)
    if nodata_mask is not None:
        full_predictions_array[~nodata_mask.ravel()] = predictions
    predictions_reshaped = full_predictions_array.reshape((height, width))
    return predictions_reshaped


def _check_grid_properties(raster_files):
    """
    Check that all input rasters have the same grid properties.

    Args:
        raster_files: List of filepaths to raster files.

    Returns:
        bool: True if all rasters have the same grid properties, False otherwise.
    """
    if not raster_files:
        raise ValueError("No raster files provided.")

    # Get properties of the first raster as reference
    ref_desc = arcpy.Describe(raster_files[0])
    ref_rows = ref_desc.height
    ref_cols = ref_desc.width
    ref_cell_size_x = ref_desc.meanCellWidth
    ref_cell_size_y = ref_desc.meanCellHeight
    ref_extent = ref_desc.extent

    for raster_file in raster_files[1:]:
        desc = arcpy.Describe(raster_file)
        rows = desc.height
        cols = desc.width
        cell_size_x = desc.meanCellWidth
        cell_size_y = desc.meanCellHeight
        extent = desc.extent

        if (rows != ref_rows or cols != ref_cols or
            cell_size_x != ref_cell_size_x or cell_size_y != ref_cell_size_y or
            extent != ref_extent):
            return False

    return True

def _resample_raster(input_raster, reference_raster, output_path):
    """
    Resample the input raster to match the grid properties of the reference raster.

    Args:
        input_raster: Path to the input raster to be resampled.
        reference_raster: Path to the reference raster with the desired grid properties.
        output_path: Path to save the resampled raster.

    Returns:
        Path to the resampled raster.
    """

    ref_raster = arcpy.Raster(reference_raster)
    resampled_raster = arcpy.management.Resample(
        in_raster=input_raster,
        out_raster=output_path,
        cell_size=ref_raster.meanCellWidth,
        resampling_type="NEAREST"
    )
    return resampled_raster

def prepare_data_for_ml(
    feature_raster_files,
    label_file: Optional[Union[str, os.PathLike]] = None,
    nodata_value: Optional[Number] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Any]:
    """
    Prepare data ready for machine learning model training.

    Performs the following steps:
    - Read all bands of all feature/evidence rasters into a stacked Numpy array
    - Read label data (and rasterize if a vector file is given)
    - Create a nodata mask using all feature rasters and labels, and mask nodata cells out

    Args:
        feature_raster_files: List of filepaths of feature/evidence rasters. Files should only include
            raster that have the same grid properties and extent.
        label_file: Filepath to label (deposits) data. File can be either a vector file or raster file.
            If a vector file is provided, it will be rasterized into similar grid as feature rasters. If
            a raster file is provided, it needs to have same grid properties and extent as feature rasters.
            Optional parameter and can be omitted if preparing data for predicting. Defaults to None.

    Returns:
        Feature data (X) in prepared shape.
        Target labels (y) in prepared shape (if `label_file` was given).
        Refrence raster metadata .
        Nodata mask applied to X and y.

    Raises:
    Error: Input feature rasters contains only one path.
    Error: Input feature rasters, and optionally rasterized label file,
            don't have same grid properties.
    """

    def _read_and_stack_feature_raster(filepath: Union[str, os.PathLike]) -> Tuple[np.ndarray, dict]:
        """Read all bands of raster file with feature/evidence data in a stack."""
        desc = arcpy.Describe(filepath)
        path_to_raster = desc.catalogPath
        out_bands_raster = [arcpy.ia.ExtractBand(path_to_raster, band_ids=i) for i in range(1, desc.bandCount + 1)] 
        dataset = [arcpy.RasterToNumPyArray(band) for band in out_bands_raster] 
        raster_data = np.stack(dataset)
        return raster_data

    if len(feature_raster_files) < 2:
        arcpy.AddError(f"Expected more than one feature raster file: {len(feature_raster_files)}.")
        raise arcpy.ExecuteError
    
    rasters_to_check = feature_raster_files.copy()

    grid_check = _check_grid_properties(rasters_to_check)
    
    if not grid_check:
        arcpy.AddWarning("Resampling feature rasters to match grid properties of the first raster.")
        reference_raster = feature_raster_files[0]
        resampled_feature_raster_files = []
        for i, raster_file in enumerate(feature_raster_files):
            
            resampled_raster = _resample_raster(raster_file,
                                                reference_raster,
                                                os.path.join(arcpy.env.scratchFolder,
                                                f"resampled_feature_raster_{i}.tif"))
            
            resampled_feature_raster_files.append(resampled_raster)
        
        feature_raster_files = resampled_feature_raster_files
        feature_data = [_read_and_stack_feature_raster(file) for file in resampled_feature_raster_files]
    else:
        # Read and stack feature rasters
        feature_data = [_read_and_stack_feature_raster(file) for file in feature_raster_files]

    # Verify that all feature rasters have the same shape after resampling
    shapes = [raster.shape for raster in feature_data]
    if len(set(shapes)) > 1:
        arcpy.AddWarning(f"Feature rasters do not have the same shape after resampling: {shapes}")
        arcpy.AddWarning("Cropping feature rasters to the smallest common shape.")
        min_shape = np.min([raster.shape for raster in feature_data], axis=0)
        feature_data = [raster[:, :min_shape[1], :min_shape[2]] for raster in feature_data]

    # Reshape feature rasters for ML and create mask
    reshaped_data = []
    nodata_mask = None

    for raster in feature_data:

        # Reshape each raster to 2D array where each row is a pixel and each column is a band
        raster_reshaped = raster.reshape(raster.shape[0], -1).T
        reshaped_data.append(raster_reshaped)

        # Create a mask for NaN values
        nan_mask = (raster_reshaped == np.nan).any(axis=1)
        combined_mask = nan_mask if nodata_mask is None else nodata_mask | nan_mask

        # Create a mask for nodata values if nodata_value is provided
        if nodata_value is not None:
            raster_mask = (raster_reshaped == np.nan).any(axis=1)
            combined_mask = combined_mask | raster_mask

        # Combine NaN and nodata masks
        nodata_mask = combined_mask

    X = np.concatenate(reshaped_data, axis=1)

    if label_file is not None:
        
        desc = arcpy.Describe(label_file)
        if desc.dataType == "FeatureClass" or desc.dataType == "FeatureLayer":

            # Rasterize vector file
            rasterized_vector = rasterize_vector(rasters_to_check[0], label_file)

            # Convert raster to numpy array
            y = arcpy.RasterToNumPyArray(rasterized_vector)
        else:
            label_resampled = _resample_raster(label_file, feature_raster_files[0], os.path.join(arcpy.env.scratchFolder, "y_resampled"))
            desc_label_resampled = arcpy.Describe(label_resampled)
            y = arcpy.RasterToNumPyArray(desc_label_resampled.catalogPath)
            
            label_nodata_mask = y == nodata_value
            
            # Truncate the larger array to match the smaller one
            min_size = min(nodata_mask.size, label_nodata_mask.size)
            nodata_mask = nodata_mask[:min_size]
            label_nodata_mask = label_nodata_mask.ravel()[:min_size]

            # Combine masks and apply to label data
            nodata_mask = nodata_mask | label_nodata_mask

        # Flatten label data and make sure it has the same size as the mask
        y = y.ravel()[:nodata_mask.size]
        y = y[~nodata_mask]
        

    else:
        y = None

    X = X[~nodata_mask]

    return X, y, nodata_mask


def read_data_for_evaluation(
    rasters: Sequence[Union[str, os.PathLike]]
) -> Tuple[Sequence[np.ndarray], List[Any], Any]:
    """
    Prepare data ready for evaluating modeling outputs.

    Reads all rasters (only first band), reshapes them (flattens) and masks out all NaN
    and nodata pixels by creating a combined mask from all input rasters.

    Args:
        rasters: List of filepaths of input rasters. Files should only include raster that have
            the same grid properties and extent.

    Returns:
        List of reshaped and masked raster data.
        Refrence raster profile.
        Nodata mask applied to raster data.

    Raises:
    Error: Input rasters contains only one path.
    Error: Input rasters don't have same grid properties.
    """
    if len(rasters) < 2:
        arcpy.AddError(f"Expected more than one raster file: {len(rasters)}.")
        raise arcpy.ExecuteError

    profiles = []
    raster_data = []
    nodata_values = []

    for raster in rasters:
        desc = arcpy.Describe(raster)
        data = arcpy.RasterToNumPyArray(raster)
        bands = [arcpy.ia.ExtractBand(raster, band_ids=i) for i in range(1, desc.bandCount + 1)]
        profile = {
            "width": desc.width,
            "height": desc.height,
            "nodata": desc.noDataValue,
            "extent": desc.extent,
            "meanCellWidth": desc.meanCellWidth,
            "meanCellHeight": desc.meanCellHeight,
        }
        profiles.append(profile)
        raster_data.append(bands)
        nodata_values.append(desc.noDataValue)

    reference_profile = profiles[0]
    nodata_mask = None

    for data, nodata in zip(raster_data, nodata_values):
        nan_mask = np.isnan(data)
        combined_mask = nan_mask if nodata_mask is None else nodata_mask | nan_mask

        if nodata is not None:
            raster_mask = data == nodata
            combined_mask = combined_mask | raster_mask

        nodata_mask = combined_mask
    nodata_mask = nodata_mask.flatten()

    masked_data = []
    for data in raster_data:
        flattened_data = data.flatten()
        masked_data.append(flattened_data[~nodata_mask])

    return masked_data, reference_profile, nodata_mask


def _train_and_validate_sklearn_model(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    model: BaseEstimator,
    validation_method: Literal["split", "kfold_cv", "skfold_cv", "loo_cv", "none"],
    metrics: Sequence[Literal["mse", "rmse", "mae", "r2", "accuracy", "precision", "recall", "f1"]],
    split_size: float = 0.2,
    cv_folds: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> Tuple[BaseEstimator, dict]:
    """
    Train and validate Sklearn model.

    Serves as a common private/inner function for Random Forest, Logistic Regression and Gradient Boosting
    public functions.

    Args:
        X: Training data.
        y: Target labels.
        model: Initialized, to-be-trained Sklearn model.
        validation_method: Validation method to use.
        metrics: Metrics to use for scoring the model.
        split_size: Fraction of the dataset to be used as validation data (for validation method "split").
            Defaults to 0.2.
        cv_folds: Number of folds used in cross-validation. Defaults to 5.
        shuffle: If data is shuffled before splitting. Defaults to True.
        random_state: Seed for random number generation. Defaults to None.

    Returns:
        Trained Sklearn model and metric scores as a dictionary.

    Raises:
    Error: X and y have mismatching sizes.
    Error: Validation method was chosen without any metric or `cv_folds` is too small.
    """
    # Perform checks
    x_size = X.index.size if isinstance(X, pd.DataFrame) else X.shape[0]
    if x_size != y.size:
        arcpy.AddError(f"X and y must have the length {x_size} != {y.size}.")
        raise arcpy.ExecuteError
    if len(metrics) == 0 and validation_method != NO_VALIDATION:
        arcpy.AddError("Metrics must have at least one chosen metric to validate model.")
        raise arcpy.ExecuteError
    if cv_folds < 2:
        arcpy.AddError("Number of cross-validation folds must be at least 2.")
        raise arcpy.ExecuteError

    # Validation approach 1: No validation
    if validation_method == NO_VALIDATION:
        model.fit(X, y)
        metrics = {}

        return model, metrics

    # Validation approach 2: Validation with splitting data once
    elif validation_method == SPLIT:
        X_train, X_valid, y_train, y_valid = split_data(
            X, y, split_size=split_size, random_state=random_state, shuffle=shuffle
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)

        out_metrics = {}
        for metric in metrics:
            score = score_predictions(y_valid, y_pred, metric, decimals=3)
            out_metrics[metric] = score

    # Validation approach 3: Cross-validation
    elif validation_method in [KFOLD_CV, SKFOLD_CV, LOO_CV]:
        cv = _get_cross_validator(validation_method, cv_folds, shuffle, random_state)

        # Initialize output metrics dictionary
        out_metrics = {}
        for metric in metrics:
            out_metrics[metric] = {}
            out_metrics[metric][f"{metric}_all"] = []

        # Loop over cross-validation folds and save metric scores
        for train_index, valid_index in cv.split(X, y):
            model.fit(X[train_index], y[train_index])
            y_pred = model.predict(X[valid_index])

            for metric in metrics:
                score = score_predictions(y[valid_index], y_pred, metric, decimals=3)
                all_scores = out_metrics[metric][f"{metric}_all"]
                all_scores.append(score)

        # Calculate mean and standard deviation for all metrics
        for metric in metrics:
            scores = out_metrics[metric][f"{metric}_all"]
            out_metrics[metric][f"{metric}_mean"] = np.mean(scores)
            out_metrics[metric][f"{metric}_std"] = np.std(scores)

        # Fit on entire dataset after cross-validation
        model.fit(X, y)

        # If we calculated only 1 metric, remove the outer dictionary layer from output
        if len(out_metrics) == 1:
            out_metrics = out_metrics[metrics[0]]

    else:
        arcpy.AddError(f"Unrecognized validation method: {validation_method}")
        raise arcpy.ExecuteError

    return model, out_metrics


def _get_cross_validator(
    cv: Literal["kfold_cv", "skfold_cv", "loo_cv"], folds: int, shuffle: bool, random_state: Optional[int]
) -> Union[KFold, StratifiedKFold, LeaveOneOut]:
    """
    Create a Sklearn cross-validator.

    Args:
        cv: Name/identifier of the cross-validator.
        folds: Number of folds to use (for Kfold and StratifiedKFold).
        shuffle: If data is shuffled before splitting.
        random_state: Seed for random number generation.

    Returns:
        Sklearn cross-validator instance.

    Raises:
    Error: Invalid input for `cv`.
    """
    if cv == KFOLD_CV:
        cross_validator = KFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
    elif cv == SKFOLD_CV:
        cross_validator = StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
    elif cv == LOO_CV:
        cross_validator = LeaveOneOut()
    else:
        arcpy.AddError(f"CV method was not recognized: {cv}")
        raise arcpy.ExecuteError

    return cross_validator