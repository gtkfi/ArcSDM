"""
Original source code from https://github.com/GispoCoding/eis_toolkit
"""

import os
import arcpy
import arcpy.ia
from arcpy import env
from arcpy.sa import IsNull
import joblib
import pandas as pd
from numbers import Number
import numpy as np
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, train_test_split

from arcsdm.evaluation.scoring import score_predictions

SPLIT = "split"
KFOLD_CV = "kfold_cv"
SKFOLD_CV = "skfold_cv"
LOO_CV = "loo_cv"
NO_VALIDATION = "none"

def _is_keras_model(obj) -> bool:
    try:
        import tensorflow as tf
        return isinstance(obj, tf.keras.Model)
    except Exception:
        return False

def save_model(model, path: str) -> str:
    """
    Saves scikit-learn with joblib; saves Keras with model.save().
    If a Keras model is given a .pkl/.joblib path, switch to .keras.
    Returns the actual path written.
    """
    ext = os.path.splitext(path)[1].lower()

    if _is_keras_model(model):
        # Prefer single-file .keras format
        if ext in (".pkl", ".joblib", ""):
            path = os.path.splitext(path)[0] + ".keras"
        # Keras will create either a single .keras file or a SavedModel dir (if ext is missing)
        model.save(path)
        return path
    else:
        joblib.dump(model, path)

def load_model(path: str):
    """
    Loads models saved by save_model().
    - .keras or SavedModel dir -> keras.models.load_model
    - otherwise -> joblib.load
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".keras" or (ext == "" and os.path.isdir(path)):
        from tensorflow.keras import models
        return models.load_model(path)
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


def _describe_raster_grid(r_path: str) -> dict:
    """Collect grid metadata for compatibility checks."""
    r = arcpy.Raster(r_path)
    d = arcpy.Describe(r)

    return {
        "rows": r.height,
        "cols": r.width,
        "cellsize_x": r.meanCellWidth,
        "cellsize_y": r.meanCellHeight,
        "spatial_ref": d.spatialReference.exportToString() if d.spatialReference else None,
        "extent": (d.extent.XMin, d.extent.YMin, d.extent.XMax, d.extent.YMax),
        "nodata": d.noDataValue,
        "path": r_path,
    }

def _check_raster_grids(grids: List[dict], same_extent: bool = True) -> bool:
    """Verify all rasters share cell size, spatial ref, and (optionally) extent."""
    if not grids:
        return False
    g0 = grids[0]
    for g in grids[1:]:
        if not np.isclose(g["cellsize_x"], g0["cellsize_x"]) or not np.isclose(g["cellsize_y"], g0["cellsize_y"]):
            return False
        if g["spatial_ref"] != g0["spatial_ref"]:
            return False
        if same_extent and g["extent"] != g0["extent"]:
            return False
        if g["rows"] != g0["rows"] or g["cols"] != g0["cols"]:
            return False
    return True


def _raster_to_band_arrays(r_path: str) -> list[np.ndarray]:
    """
    Read a raster into NumPy and return a list of 2D arrays (one per band).
    Uses IsNull to get the NoData mask, then maps those cells to np.nan.
    """
    # Read data (no nodata_to_value here!)
    data = arcpy.RasterToNumPyArray(r_path)

    # Build NoData mask using Spatial Analyst
    nodata_mask = arcpy.RasterToNumPyArray(IsNull(r_path)).astype(bool)

    # Ensure float so we can store NaN
    data = data.astype(float, copy=False)

    # data/nodata_mask can be 2D or 3D (bands, rows, cols)
    if data.ndim == 2:
        data[nodata_mask] = np.nan
        return [data]

    if data.ndim == 3:
        # Apply mask band-wise and split to 2D bands
        bands = []
        for i in range(data.shape[0]):
            band = data[i, :, :]
            band_mask = nodata_mask[i, :, :] if nodata_mask.ndim == 3 else nodata_mask
            band[band_mask] = np.nan
            bands.append(band)
        return bands

    raise RuntimeError(f"Unexpected array shape from {r_path}: {data.shape}")

def _read_feature_bands(feature_raster_files: Sequence[str], rows: int, cols: int) -> list[np.ndarray]:
    """
    Read all bands from all feature rasters as a flat list of 2D arrays.
    Validates shape against (rows, cols).
    """
    band_arrays: list[np.ndarray] = []
    for rpath in feature_raster_files:
        for band_arr in _raster_to_band_arrays(rpath):
            if band_arr.shape != (rows, cols):
                raise arcpy.AddError(f"Band grid mismatch: {rpath} -> {band_arr.shape} != {(rows, cols)}")
            band_arrays.append(band_arr)
    return band_arrays

def _rasterize_vector_to_array(
    vector_path: str,
    reference_grid: dict,
    value_field: str = None,
) -> np.ndarray:
    """
    Rasterize a vector to the reference grid using FeatureToRaster.
    If value_field is None, burn constant 1. Returns a 2D float array with NoData as np.nan.
    """
    ref_path = reference_grid["path"]
    ref_r = arcpy.Raster(ref_path)

    old_extent, old_cell, old_snap = env.extent, env.cellSize, env.snapRaster
    must_drop = False
    field_to_use = value_field

    try:
        # Match the reference raster grid
        env.extent = arcpy.Describe(ref_path).extent
        env.cellSize = ref_r.meanCellWidth
        env.snapRaster = ref_path

        # If no label field, add a temp constant
        if field_to_use is None:
            field_to_use = "_ML_CONST_"
            if field_to_use not in [f.name for f in arcpy.ListFields(vector_path)]:
                arcpy.management.AddField(vector_path, field_to_use, "SHORT")
                arcpy.management.CalculateField(vector_path, field_to_use, 1, "PYTHON3")
                must_drop = True

        # Create output name in scratch gdb and run FeatureToRaster
        out_name = arcpy.CreateUniqueName("lbl_ras_", arcpy.env.scratchWorkspace)
        res = arcpy.conversion.FeatureToRaster(
            in_features=vector_path,
            field=field_to_use,
            out_raster=out_name,
            cell_size=ref_r.meanCellWidth,
        )
        out_path = res.getOutput(0)
        out_ras = arcpy.Raster(out_path)

        # Read data and mask NoData via IsNull
        lbl_data = arcpy.RasterToNumPyArray(out_ras)
        lbl_mask = arcpy.RasterToNumPyArray(IsNull(out_ras)).astype(bool)

        lbl_data = lbl_data.astype(float, copy=False)
        lbl_data[lbl_mask] = np.nan
        return lbl_data

    finally:
        # Clean up temp field if we added it
        if must_drop:
            try:
                arcpy.management.DeleteField(vector_path, field_to_use)
            except Exception:
                pass
        # Restore env
        arcpy.env.extent, arcpy.env.cellSize, arcpy.env.snapRaster = old_extent, old_cell, old_snap


def _is_nan_like(v) -> bool:
    return isinstance(v, float) and np.isnan(v)


def _apply_explicit_nodata_inplace(arr: np.ndarray, nodata_value: Number):
    """Mark explicit nodata_value in arr as NaN (casts to float if needed)."""
    if _is_nan_like(nodata_value):
        return  # nothing to do
    if not np.issubdtype(arr.dtype, np.floating):
        arr[...] = arr.astype(float, copy=False)
    # equality is fine for ints; use isclose for floats
    if isinstance(nodata_value, float):
        mask_val = np.isclose(arr, nodata_value)
    else:
        mask_val = (arr == nodata_value)
    arr[mask_val] = np.nan


def prepare_data_for_ml(
    feature_raster_files: Sequence[str],
    label_file: Optional[str] = None,
    label_field: Optional[str] = None,
    feature_raster_nodata_value: Number = np.nan,
    label_nodata_value: Number = np.nan,
) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
    """
    ArcPy version using explicit NoData values for features and labels.
    Returns (X, y, reference_profile) with 'nodata_mask' inside reference_profile.
    """
    if len(feature_raster_files) < 2:
        msg = f"Expected more than one feature raster file: {len(feature_raster_files)}."
        arcpy.AddError(msg)
        raise ValueError(msg)

    # Grid checks
    grids = [_describe_raster_grid(p) for p in feature_raster_files]
    if not _check_raster_grids(grids, same_extent=True):
        msg = "Input feature rasters should have same grid properties."
        arcpy.AddError(msg)
        raise ValueError(msg)

    reference = grids[0]
    rows, cols = reference["rows"], reference["cols"]

    # --- Features ---
    # band_arrays must be a list of 2D float arrays where native NoData is already np.nan
    band_arrays = _read_feature_bands(feature_raster_files, rows, cols)

    # Apply the user-provided feature NoData value (if not NaN)
    if not _is_nan_like(feature_raster_nodata_value):
        for _, band_array in enumerate(band_arrays):
            _apply_explicit_nodata_inplace(band_array, feature_raster_nodata_value)

    # Stack -> (n_pixels, n_features)
    reshaped = [a.reshape(-1) for a in band_arrays]
    X_full = np.stack(reshaped, axis=1)

    # Mask where any feature band is NaN
    nodata_mask = np.isnan(X_full).any(axis=1)

    # --- Labels ---
    y = None
    if label_file is not None and arcpy.Exists(label_file):
        _, ext = os.path.splitext(label_file)
        ext = ext.lower()

        # Vector labels -> rasterize
        if (ext in {".shp", ".geojson", ".json", ".gpkg", ".gdb"} or
            arcpy.Describe(label_file).dataType in {"FeatureClass", "ShapeFile", "FeatureLayer"}):
            y_arr = _rasterize_vector_to_array(label_file, reference, value_field=label_field)
        else:
            # Raster labels -> must match grid
            lbl_grid = _describe_raster_grid(label_file)
            if not _check_raster_grids([reference, lbl_grid], same_extent=True):
                msg = "Label raster should have the same grid properties as features."
                arcpy.AddError(msg)
                raise ValueError(msg)
            # Read label raster; map native NoData -> NaN via IsNull inside helper or here:
            lbl_data = arcpy.RasterToNumPyArray(label_file).astype(float, copy=False)
            lbl_mask = arcpy.RasterToNumPyArray(arcpy.sa.IsNull(label_file)).astype(bool)
            lbl_data[lbl_mask] = np.nan
            y_arr = lbl_data

        if y_arr.shape != (rows, cols):
            msg = "Rasterized/label grid shape mismatch."
            arcpy.AddError(msg)
            raise ValueError(msg)

        # Apply the user-provided label NoData value (if not NaN)
        if not _is_nan_like(label_nodata_value):
            _apply_explicit_nodata_inplace(y_arr, label_nodata_value)

        # Any NaN in labels is excluded from training/eval
        label_mask = np.isnan(y_arr).reshape(-1)
        nodata_mask = nodata_mask | label_mask

        # Flatten labels after masking
        y = y_arr.reshape(-1)[~nodata_mask]

    # Apply final mask to features
    X = X_full[~nodata_mask, :]

    reference_profile = {
        "height": rows,
        "width": cols,
        "transform": None,
        "crs_wkt": reference["spatial_ref"],
        "cellsize_x": reference["cellsize_x"],
        "cellsize_y": reference["cellsize_y"],
        "extent": reference["extent"],
        "path": reference["path"],
        "nodata_mask": nodata_mask,  # (n_pixels,) boolean
    }

    return X, y, reference_profile

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


def train_and_validate_sklearn_model(
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