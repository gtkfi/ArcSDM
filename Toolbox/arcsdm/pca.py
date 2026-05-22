
""" ArcSDM 6 ToolBox for ArcGIS Pro

Conversion and tool development for ArcGIS Pro by Geological Survey of Finland (GTK), 2024.

Compute defined number of principal components for numeric input data and transform the data.

Before computation, data is scaled according to the specified scaler and NaN values are removed or replaced.
Optionally, a nodata value can be given to handle similarly as NaN values.

This tool is based on the PCA implementation in the scikit-learn library originally developed by University of Turku.
Original implementation is included in EIS Toolkit (https://github.com/GispoCoding/eis_toolkit).

Loadings Table in ArcSDM message output:
----------------
The loadings table displays the contribution of each original variable to each principal component.
Loadings are calculated as the eigenvector (component) values multiplied by the square root of the corresponding
eigenvalue (variance explained by the component), i.e., loading = eigenvector * sqrt(eigenvalue)

The data is first scaled using the selected scaler ("standard", "min_max", or "robust") before PCA is performed. 
The loadings table is sorted so that variables are grouped by the principal component on which they have the largest absolute loading.
This helps to highlight which variables contribute most strongly to each principal component.
"""

import sys
from pathlib import Path

import arcpy
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from utils.input_to_numpy_array import read_and_stack_rasters

from arcsdm.exceptions import SDMError

SCALERS = {"standard": StandardScaler, "min_max": MinMaxScaler, "robust": RobustScaler}

def Execute(self, parameters, messages):
    """
    Main entry point for the ArcSDM PCA tool.

    This function handles both vector and raster input data, prepares the data, applies PCA transformation,
    and writes the transformed data to the specified output. Handles nodata values and scaling as specified.

    Args:
        self: Reference to the toolbox class instance (required by ArcPy toolbox interface).
        parameters: List of tool parameters as provided by ArcGIS Pro.
        messages: Message object for ArcPy messaging.

    Returns:
        None. Results are written to ArcGIS outputs and messages.

    Raises:
        arcpy.ExecuteError: If ArcPy encounters an error during processing.
        SDMError: For custom errors related to unsupported vector types.
    """
    try:
        input_data = parameters[0].valueAsText.split(';')
        input_dataType = arcpy.Describe(input_data[0]).dataType
        is_vector = input_dataType in ("FeatureClass", "ShapeFile", "FeatureLayer")

        if (is_vector):
            input_fields = parameters[1].valueAsText.split(';')
            nodata_param = parameters[2].value
            nodata_value = nodata_param if nodata_param is not None else -99
            number_of_components = parameters[3].value
            scaler_type = parameters[4].valueAsText
            nodata_handling = parameters[5].valueAsText
            transformed_data_output = parameters[6].valueAsText

            rasterizedInputs = []
            input_vector = input_data[0]
            desc_input = arcpy.Describe(input_vector)
            input_shape = desc_input.shapeType

            for field in input_fields:
                output_raster = f"in_memory\\raster__{input_vector}__{field}"

                if input_shape == "Point":
                    inputRaster = arcpy.conversion.PointToRaster(
                        in_features=input_vector,
                        value_field=field,
                        out_rasterdataset=output_raster,
                        cellsize=500
                    )
                elif input_shape == "Polyline":
                    inputRaster = arcpy.conversion.PolylineToRaster(
                        in_features=input_vector,
                        value_field=field,
                        out_rasterdataset=output_raster,
                        cellsize=500
                    )
                elif input_shape == "Polygon":
                    inputRaster = arcpy.conversion.PolygonToRaster(
                        in_features=input_vector,
                        value_field=field,
                        out_rasterdataset=output_raster,
                        cellsize=500
                    )
                else:
                    raise SDMError(f"Unsupported vector type: {input_shape} in {input_vector}")

                rasterizedInputs.append(arcpy.Raster(inputRaster))
                
            input_data = rasterizedInputs

        else:
            number_of_components = parameters[1].value
            scaler_type = parameters[2].valueAsText
            nodata_handling = parameters[3].valueAsText
            transformed_data_output = parameters[4].valueAsText
            raster_nodata = arcpy.Raster(input_data[0]).noDataValue
            nodata_value = raster_nodata if raster_nodata is not None else -99

        desc_input = arcpy.Describe(input_data[0])
        is_multiband = hasattr(desc_input, "bandCount") and desc_input.bandCount > 1
        stacked_arrays = read_and_stack_rasters(input_data, nodata_value, nodata_handling = "convert_to_nan")

        # Collect variable names for loadings table, derived after stacking to match actual band count
        if is_vector:
            variable_names = input_fields
        else:
            original_input_paths = parameters[0].valueAsText.split(';')
            band_counts = [arcpy.Raster(p).bandCount for p in original_input_paths]
            if sum(band_counts) != stacked_arrays.shape[0]:
                arcpy.AddWarning("Variable name count does not match stacked band count. Using generic names.")
                variable_names = [f"band_{i + 1}" for i in range(stacked_arrays.shape[0])]
            else:
                variable_names = []
                for p, band_count in zip(original_input_paths, band_counts):
                    base = Path(p).stem
                    if band_count > 1:
                        for b in range(1, band_count + 1):
                            variable_names.append(f"{base}_band{b}")
                    else:
                        variable_names.append(base)
        
        if len(stacked_arrays) == 1:
            arcpy.AddError("Only one band found in input data. PCA requires at least two bands.")
            raise arcpy.ExecuteError
        
        # Perform PCA
        transformed_data, principal_components, explained_variances, explained_variance_ratios = compute_pca(
            stacked_arrays, number_of_components, scaler_type, nodata_handling, nodata_value
        )

        arcpy.AddMessage('='*5 + ' PCA results ' + '='*5)
        
        # Save output data
        if is_vector:
            arcpy.management.CopyFeatures(input_vector, transformed_data_output)
            
            if transformed_data.ndim > 2:
                rows, cols = transformed_data.shape[1], transformed_data.shape[2]
                transformed_data = transformed_data.transpose(1, 2, 0).reshape(rows * cols, -1)
            
            for i in range(principal_components.shape[0]):
                field_name = f"PCA_component_{i+1}"
                arcpy.management.AddField(transformed_data_output, field_name, "DOUBLE")
                
                with arcpy.da.UpdateCursor(transformed_data_output, field_name) as cursor:
                    for j, row in enumerate(cursor):
                        row[0] = float(transformed_data[j, i])
                        cursor.updateRow(row)

        elif transformed_data.ndim == 2 or transformed_data.ndim == 3:
            desc_input = arcpy.Describe(input_data[0])
            if is_multiband:
                first_band = arcpy.ia.ExtractBand(input_data[0], band_ids=1)
                desc_input = arcpy.Describe(first_band)

            transformed_data_raster = arcpy.NumPyArrayToRaster(transformed_data,
                                                            lower_left_corner=desc_input.extent.lowerLeft, 
                                                            x_cell_size=desc_input.meanCellWidth,
                                                            y_cell_size=desc_input.meanCellHeight,
                                                            value_to_nodata=np.nan)
            transformed_data_raster.save(transformed_data_output)
            
            arcpy.AddMessage(f'Transformed data is saved in {transformed_data_output}')
        else:
            arcpy.da.NumPyArrayToTable(transformed_data, transformed_data_output)
            arcpy.AddMessage(f'Transformed data is saved as a table in {transformed_data_output}')

        # Compute scaled loadings: eigenvector * sqrt(eigenvalue), shape: n_features x n_components
        loadings = principal_components.T * np.sqrt(explained_variances)
        n_features, n_components = loadings.shape
        dominant_pc = np.argmax(np.abs(loadings), axis=1)
        sort_order = np.argsort(dominant_pc)

        arcpy.AddMessage("\nVariance explained:")
        pc_w, ev_w, vp_w, cp_w = 6, 14, 12, 14
        var_header = f"{'PC':<{pc_w}}{'Eigenvalue':>{ev_w}}{'Variance %':>{vp_w}}{'Cumulative %':>{cp_w}}"
        var_sep = "-" * (pc_w + ev_w + vp_w + cp_w)
        arcpy.AddMessage(var_header)
        arcpy.AddMessage(var_sep)
        cumulative = 0.0
        for i in range(n_components):
            cumulative += explained_variance_ratios[i]
            arcpy.AddMessage(f"{'PC' + str(i + 1):<{pc_w}}{explained_variances[i]:>{ev_w}.4f}{explained_variance_ratios[i] * 100:>{vp_w}.1f}{cumulative * 100:>{cp_w}.1f}")
        arcpy.AddMessage(var_sep)

        col_w = 10
        var_w = max(12, max(len(n) for n in variable_names) + 2)
        header = f"{'Variable':<{var_w}}" + "".join(f"{'PC' + str(i + 1):>{col_w}}" for i in range(n_components))
        separator = "-" * (var_w + col_w * n_components)
        arcpy.AddMessage("\nLoadings table:")
        arcpy.AddMessage(header)
        arcpy.AddMessage(separator)
        for idx in sort_order:
            name = variable_names[idx]
            row_str = f"{name:<{var_w}}"
            for j in range(n_components):
                val = loadings[idx, j]
                row_str += f"{f'{val:+.4f}':>{col_w}}"
            arcpy.AddMessage(row_str)
        arcpy.AddMessage(separator)
        return

    except arcpy.ExecuteError:
        arcpy.AddError(arcpy.GetMessages(2)) 
    
    except:
        e = sys.exc_info()[1]
        arcpy.AddError(e.args[0])

def _prepare_array_data(
    feature_matrix: np.ndarray, nodata_handling: str, nodata_value = None, reshape = True
):
    """
    Prepare feature matrix for PCA by reshaping and handling missing/nodata values.

    Args:
        feature_matrix: Input data as a numpy ndarray (3D).
        nodata_handling: Strategy for handling missing/nodata values ('remove' or 'replace').
        nodata_value: Value to treat as nodata (converted to NaN).
        reshape: Whether to reshape 3D arrays to 2D (default True).

    Returns:
        Tuple of (processed feature_matrix, nan_mask if rows removed else None).

    Raises:
        arcpy.ExecuteError: If input data is empty.
    """
    if reshape:
        bands, rows, cols = feature_matrix.shape
        feature_matrix = feature_matrix.transpose(1, 2, 0).reshape(rows * cols, bands)

    if feature_matrix.size == 0:
        arcpy.AddError("Input data is empty.")
        raise arcpy.ExecuteError

    return _handle_missing_values(feature_matrix, nodata_handling, nodata_value)

def _handle_missing_values(
    feature_matrix, nodata_handling, nodata_value = None
):
    """
    Handle missing/nodata values in the feature matrix according to the specified strategy.

    Args:
        feature_matrix: Input numpy array.
        nodata_handling: 'remove' to drop rows with NaN, 'replace' to fill with column mean.
        nodata_value: Value to treat as nodata (converted to NaN).

    Returns:
        Tuple of (processed feature_matrix, nan_mask if rows removed else None).

    Raises:
        arcpy.ExecuteError: If nodata_handling is invalid.
    """
    nodata_mask = None

    if nodata_value is not None:
        nodata_mask = feature_matrix == nodata_value
        feature_matrix[nodata_mask] = np.nan

    if nodata_handling == "remove":
        nan_rows_mask = np.isnan(feature_matrix).any(axis=1)
        feature_matrix = feature_matrix[~nan_rows_mask]
        return feature_matrix, nan_rows_mask

    elif nodata_handling == "replace":
        for i in range(feature_matrix.shape[1]):
            column_mask = np.isnan(feature_matrix[:, i])
            column_mean = np.nanmean(feature_matrix[:, i])
            feature_matrix[column_mask, i] = column_mean
        return feature_matrix, None

    else:
        arcpy.AddError(f"Invalid nodata_handling value: {nodata_handling}. Choose 'remove' or 'replace'.")
        raise arcpy.ExecuteError


def _compute_pca(
    feature_matrix, number_of_components, scaler_type
):
    """
    Perform PCA on the feature matrix after scaling.

    Args:
        feature_matrix: 2D numpy array of input data.
        number_of_components: Number of principal components to compute.
        scaler_type: String specifying scaler ('standard', 'min_max', 'robust').

    Returns:
        Tuple of (transformed_data, principal_components, explained_variances, explained_variance_ratios).
    """
    scaler = SCALERS[scaler_type]()
    scaled_data = scaler.fit_transform(feature_matrix)

    pca = PCA(n_components=number_of_components)
    transformed_data = pca.fit_transform(scaled_data)
    principal_components = pca.components_
    explained_variances = pca.explained_variance_
    explained_variance_ratios = pca.explained_variance_ratio_

    return transformed_data, principal_components, explained_variances, explained_variance_ratios


def compute_pca(
    data,
    number_of_components = None,
    scaler_type = "standard",
    nodata_handling = "remove",
    nodata = None
):
    """
    Compute principal components for input data and transform it using PCA.

    Handles both 2D and 3D (multiband raster) numpy arrays, with options for scaling and nodata handling.

    Args:
        data: Input numpy array (2D or 3D).
        number_of_components: Number of principal components to compute. If None, uses all features.
        scaler_type: Type of scaler to use. One of:
            - 'standard': StandardScaler (removes mean, scales to unit variance)
            - 'min_max': MinMaxScaler (scales features to [0, 1] range)
            - 'robust': RobustScaler (scales using median and IQR, robust to outliers)
        nodata_handling: Strategy for missing/nodata values ('remove' or 'replace').
        nodata: Value to treat as nodata (converted to NaN).

    Returns:
        Tuple of (transformed_data, principal_components, explained_variances, explained_variance_ratios).

    Raises:
        arcpy.ExecuteError: If input data is invalid or parameters are out of range.
    """
    if number_of_components is not None and number_of_components < 1:
        arcpy.AddError("The number of principal components should be >= 1.")
        raise arcpy.ExecuteError

    # Get feature matrix (Numpy array) from various input types
    #if isinstance(data, np.ndarray):
    feature_matrix = data
    feature_matrix = feature_matrix.astype(float)
    if feature_matrix.ndim == 2:  # Table-like data (assumme it is a DataFrame transformed to Numpy array)
        feature_matrix, nan_mask = _prepare_array_data(
            feature_matrix, nodata_handling=nodata_handling, nodata_value=nodata, reshape=False
        )
    elif feature_matrix.ndim == 3:  # Assume data represents multiband raster data
        rows, cols = feature_matrix.shape[1], feature_matrix.shape[2]
        feature_matrix, nan_mask = _prepare_array_data(
            feature_matrix, nodata_handling=nodata_handling, nodata_value=nodata, reshape=True
        )
    else:
        arcpy.AddError(f"Unsupported input data format. {feature_matrix.ndim} dimensions detected for given array.")
        raise arcpy.ExecuteError

    # Default number of components to number of features in data if not defined
    if number_of_components is None:
        number_of_components = feature_matrix.shape[1]

    if number_of_components > feature_matrix.shape[1]:
        arcpy.AddError("The number of principal components is too high for the given input data "
            + f"({number_of_components} > {feature_matrix.shape[1]}).")
        raise arcpy.ExecuteError

    # Core PCA computation
    transformed_data, principal_components, explained_variances, explained_variance_ratios = _compute_pca(
        feature_matrix, number_of_components, scaler_type
    )

    if nodata_handling == "remove" and nan_mask is not None:
        transformed_data_with_nans = np.full((nan_mask.size, transformed_data.shape[1]), np.nan)
        transformed_data_with_nans[~nan_mask, :] = transformed_data
        transformed_data = transformed_data_with_nans

    # Convert PCA output to proper format
    if isinstance(data, np.ndarray):
        if data.ndim == 3:
            transformed_data_out = transformed_data.reshape(rows, cols, -1).transpose(2, 0, 1)
        else:
            transformed_data_out = transformed_data

    return transformed_data_out, principal_components, explained_variances, explained_variance_ratios

