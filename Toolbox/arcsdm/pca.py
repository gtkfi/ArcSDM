""" ArcSDM 6 ToolBox for ArcGIS Pro

Conversion and tool development for ArcGIS Pro by Geological Survey of Finland (GTK), 2024.

Compute defined number of principal components for numeric input data and transform the data.

Before computation, data is scaled according to specified scaler and NaN values removed or replaced.
Optionally, a nodata value can be given to handle similarly as NaN values.

This tool is based on the PCA implementation in the scikit-learn library originally developed by University of Turku.
"""

import sys
import arcpy
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

SCALERS = {"standard": StandardScaler, "min_max": MinMaxScaler, "robust": RobustScaler}

def Execute(self, parameters, messages):
    """The source code of the tool."""
    input_data = parameters[0].valueAsText.split(';')
    nodata_value = parameters[1].value
    number_of_components = parameters[2].value
    scaler_type = parameters[3].valueAsText
    nodata_handling = parameters[4].valueAsText
    transformed_data_output = parameters[5].valueAsText
    
    try:
        datasets = []
        for data in input_data:
            desc = arcpy.Describe(data)
            # Load input data
            if desc.dataType == "RasterDataset" or desc.dataType == "RasterLayer":
                data = arcpy.RasterToNumPyArray(data)
            datasets.append(data)
        
            # Check if all datasets have the same shape
            if len(set([dataset.shape for dataset in datasets])) > 1:
                arcpy.AddError("Input datasets have different shapes.")
                raise arcpy.ExecuteError
            
        
        stacked_array = np.stack(datasets, axis=0)

        # Perform PCA
        transformed_data, principal_components, explained_variances, explained_variance_ratios = compute_pca(
            stacked_array, number_of_components, scaler_type, nodata_handling, nodata_value
        )

        arcpy.AddMessage('='*5 + ' PCA results ' + '='*5)
        # Save output data
        if transformed_data.ndim is 2 or transformed_data.ndim is 3:
            desc_input = arcpy.Describe(input_data[0])
            transformed_data_raster = arcpy.NumPyArrayToRaster(transformed_data,
                                                            lower_left_corner=desc_input.extent.lowerLeft, 
                                                            x_cell_size=desc_input.meanCellWidth,
                                                            y_cell_size=desc_input.meanCellHeight)
            transformed_data_raster.save(transformed_data_output)
            
            arcpy.AddMessage(f'Transformed data is saved in {transformed_data_output}')
        else:
            arcpy.da.NumPyArrayToTable(transformed_data, transformed_data_output)
            arcpy.AddMessage(f'Transformed data is saved as a table in {transformed_data_output}')

        arcpy.AddMessage(f'Principal components {principal_components}')
            
        arcpy.AddMessage(f'Explained variances {explained_variances}')
        
        arcpy.AddMessage(f'Explained variance ratio {explained_variance_ratios}')
        return

    except arcpy.ExecuteError:
        arcpy.AddError(arcpy.GetMessages(2)) 
    
    except:
        e = sys.exc_info()[1]
        arcpy.AddError(e.args[0])

def _prepare_array_data(
    feature_matrix: np.ndarray, nodata_handling: str, nodata_value = None, reshape = True
):
    if reshape:
        bands, rows, cols = feature_matrix.shape
        feature_matrix = feature_matrix.transpose(1, 2, 0).reshape(rows * cols, bands)

    if feature_matrix.size == 0:
        raise arcpy.AddError("Input data is empty.")

    return _handle_missing_values(feature_matrix, nodata_handling, nodata_value)

def _handle_missing_values(
    feature_matrix, nodata_handling, nodata_value = None
):
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
        raise arcpy.AddError(f"Invalid nodata_handling value: {nodata_handling}. Choose 'remove' or 'replace'.")


def _compute_pca(
    feature_matrix, number_of_components, scaler_type
):
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

