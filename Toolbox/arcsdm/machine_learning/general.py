import arcpy
import json
import numpy as np
import torch

from numbers import Number
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import List, Sequence


class MLPInputError(Exception):
    """Exception for issues with MLP input parameters."""


class MLPError(Exception):
    """Exception for issues with MLP."""


def describe_raster_grid(r_path: str) -> dict:
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


def check_raster_grids(grids: List[dict], same_extent: bool = True) -> bool:
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


def get_nodata_mask(arrays):
    try:
        stacked = np.stack(arrays)
        nodata_mask = np.isnan(stacked).any(axis=0)
        return nodata_mask
    except ValueError:
        raise


def is_nan_like(v) -> bool:
    return isinstance(v, float) and np.isnan(v)


def apply_explicit_nodata_inplace(arr: np.ndarray, nodata_value: Number, arr_type=float):
    """Mark explicit nodata_value in arr as NaN (casts to float if needed)."""
    if is_nan_like(nodata_value):
        return  # nothing to do
    if not np.issubdtype(arr.dtype, np.floating):
        arr[...] = arr.astype(arr_type, copy=False)
    # equality is fine for ints; use isclose for floats
    if isinstance(nodata_value, float):
        mask_val = np.isclose(arr, nodata_value)
    else:
        mask_val = (arr == nodata_value)
    arr[mask_val] = np.nan


def read_raster_bands(
    raster_files: Sequence[str],
    nodata_value: Number
) -> list[np.ndarray]:
    """
    Read all bands from all feature rasters as a flat list of 2D arrays.
    """
    band_arrays = []
    for rpath in raster_files:
        for band_arr in raster_to_band_arrays(rpath):
            if not is_nan_like(nodata_value):
                apply_explicit_nodata_inplace(band_arr, nodata_value, np.float32)

            band_arrays.append(band_arr)
    return band_arrays


def raster_to_band_arrays(r_path: str) -> list[np.ndarray]:
    """
    Read a raster into NumPy and return a list of 2D arrays (one per band).
    Uses IsNull to get the NoData mask, then maps those cells to np.nan.
    """
    # Read data (no nodata_to_value here!)
    data = arcpy.RasterToNumPyArray(r_path)

    # Build NoData mask using Spatial Analyst
    nodata_mask = arcpy.RasterToNumPyArray(arcpy.sa.IsNull(r_path)).astype(bool)

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


def rasterize_vector_to_array(
    vector_path: str,
    ref_path: str,
    value_field: str = None,
    const: int = 1,
    classification = True
):
    ref_r = arcpy.Raster(ref_path)
    old_extent, old_cell, old_snap = arcpy.env.extent, arcpy.env.cellSize, arcpy.env.snapRaster

    must_drop = False
    field_to_use = value_field

    tmp_files = []

    try:
        # Match the reference raster grid
        arcpy.env.extent = arcpy.Describe(ref_path).extent
        arcpy.env.cellSize = ref_r.meanCellWidth
        arcpy.env.snapRaster = ref_path

        # If no label field, add a temp constant
        if field_to_use is None:
            field_to_use = "_ML_CONST_"
            if field_to_use not in [f.name for f in arcpy.ListFields(vector_path)]:
                arcpy.management.AddField(vector_path, field_to_use, "SHORT")
                arcpy.management.CalculateField(vector_path, field_to_use, const, "PYTHON3")
                must_drop = True

        # If label field is non-numeric, create encoded temp column and use that instead,
        # because arcpy FeatureToRaster does not support non-numeric fields
        fields = arcpy.ListFields(vector_path, value_field)
        if (not must_drop) and fields and fields[0].type not in ["Integer", "SmallInteger", "Float", "Double"]:
            if not classification:
                arcpy.AddError("Non-numeric label fields are only supported for classification tasks.")
                raise MLPInputError("Non-numeric label fields are only supported for classification tasks.")

            values = [row[0] for row in arcpy.da.SearchCursor(vector_path, [value_field])]
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(values)
            field_to_use = "_ML_ENCODED_"
            if field_to_use not in [f.name for f in arcpy.ListFields(vector_path)]:
                arcpy.management.AddField(vector_path, field_to_use, "SHORT")
                with arcpy.da.UpdateCursor(vector_path, [value_field, field_to_use]) as cursor:
                    for i, row in enumerate(cursor):
                        row[1] = int(encoded[i])
                        cursor.updateRow(row)
                must_drop = True

            # Save mapping for later use
            unique_json = arcpy.CreateUniqueName("mapping.json", arcpy.env.scratchFolder)
            mapping = {
                str(int(code)): str(label)
                for code, label in zip(encoder.transform(encoder.classes_), encoder.classes_)
            }
            with open(unique_json, "w") as f:
                json.dump(mapping, f, indent=2)
                json_str = json.dumps(mapping, indent=2)

                arcpy.AddMessage(f"Non-numeric label field detected. Created encoded numeric field for rasterization and saved mapping to {unique_json}. Mapping: {json_str}")

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
        tmp_files.append(out_path)

        # Read data and mask NoData via IsNull
        lbl_data = arcpy.RasterToNumPyArray(out_ras)
        lbl_mask = arcpy.RasterToNumPyArray(arcpy.sa.IsNull(out_ras)).astype(bool)

        lbl_data = lbl_data.astype(np.float32, copy=False)
        lbl_data[lbl_mask] = np.nan

        # Delete tmp file(s)
        for tmp_file in tmp_files:
            arcpy.management.Delete(tmp_file)

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


def pick_value(arrays, prefer="first"):
    """
    arrays: list of same-shape 2D arrays with values or np.nan
    prefer: 'first' or 'last' non-NaN in stack order
    """
    stacked = np.stack(arrays, axis=0)        # (k, rows, cols)
    valid = ~np.isnan(stacked)                # True where array has a value
    any_valid = valid.any(axis=0)             # True where at least one array has value

    if prefer == "first":
        idx = valid.argmax(axis=0)            # first True index (or 0 if none)
    elif prefer == "last":
        idx = stacked.shape[0] - 1 - valid[::-1].argmax(axis=0)
    else:
        raise ValueError("prefer must be 'first' or 'last'")

    rows, cols = np.indices(any_valid.shape)
    out = stacked[idx, rows, cols].astype(float)
    # Keep nan where all were nan
    out[~any_valid] = np.nan
    return out


def standardize(feature, scaler=None):
    fit_transform = False
    if scaler is None:
        scaler = StandardScaler(with_mean=True, with_std=True)
        fit_transform = True

    if fit_transform:
        scaled_feature = scaler.fit_transform(feature)
    else:
        scaled_feature = scaler.transform(feature)

    return scaled_feature, scaler


def plot_loss_curves(
    ax,
    epochs,
    train_loss_dict,
    val_loss_dict
):
    epoch_count = range(1, epochs + 1)
    ax.plot(epoch_count, train_loss_dict.values(), label="Training loss")
    ax.plot(epoch_count, val_loss_dict.values(), label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()


def reshape_predictions(
    predictions: np.ndarray, height: int, width: int, nodata_mask = None
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


def prepare_data_for_prediction(raster_arrays, raster_nodata):
    raster_arrays_clean = []
    for arr in raster_arrays:
        arr_nodata = (arr == raster_nodata)
        arr[arr_nodata] = np.nan
        raster_arrays_clean.append(arr)

    nodata_mask = get_nodata_mask(raster_arrays_clean)

    raster_row_vectors_clean = []
    for arr in raster_arrays:
        arr[nodata_mask] = np.nan
        arr = arr[~np.isnan(arr)]
        raster_row_vectors_clean.append(arr)

    # Stack rows vertically
    X_arr = np.stack(raster_row_vectors_clean, axis=0)
    # Transpose to represent data as column vectors
    X_arr = X_arr.T

    data_tensor = torch.from_numpy(X_arr)
    dummy_labels = torch.zeros(X_arr.shape[0], 1) # Not used, but required by DataLoader
    return data_tensor, dummy_labels

