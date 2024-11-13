""" ArcSDM 6 ToolBox for ArcGIS Pro

Conversion and tool development for ArcGIS Pro by Geological Survey of Finland (GTK), 2024.

Compute defined number of principal components for numeric input data and transform the data.

Before computation, data is scaled according to specified scaler and NaN values removed or replaced.
Optionally, a nodata value can be given to handle similarly as NaN values.
"""

import arcpy
import numpy as np
import pandas as pd

def Execute(self, parameters, messages):
    """The source code of the tool."""
    data = parameters[0].valueAsText
    number_of_components = parameters[1].value
    columns = parameters[2].valueAsText.split(';') if parameters[2].valueAsText else None
    scaler_type = parameters[3].valueAsText
    nodata_handling = parameters[4].valueAsText
    nodata_value = parameters[5].value
    transformed_data_output = parameters[6].valueAsText
    principal_components_output = parameters[7].valueAsText
    explained_variances_output = parameters[8].valueAsText
    explained_variance_ratios_output = parameters[9].valueAsText    

    # Load input data
    if arcpy.Describe(data).dataType == "RasterDataset" or arcpy.Describe(data).dataType == "RasterLayer":
        data = arcpy.RasterToNumPyArray(data)
    #elif arcpy.Describe(input_data).dataType == "Table":
        #data = pd.DataFrame(arcpy.da.TableToNumPyArray(data, "*"))
    elif arcpy.Describe(data).dataType == "FeatureLayer":
        data = arcpy.da.FeatureClassToNumPyArray(data, "*")

    # Perform PCA
    transformed_data, principal_components, explained_variances, explained_variance_ratios = compute_pca(
        data, number_of_components, columns, scaler_type, nodata_handling, nodata_value
    )

    # Save output data
    if isinstance(transformed_data, np.ndarray):
        if transformed_data.ndim is 2 or transformed_data.ndim is 3:
            arcpy.NumPyArrayToRaster(transformed_data).save(transformed_data_output)
        else:
            arcpy.da.NumPyArrayToTable(transformed_data, transformed_data_output)
            
        if principal_components.ndim is 2 or principal_components.ndim is 3:
            arcpy.NumPyArrayToRaster(principal_components).save(principal_components_output)  
        else:
            arcpy.da.NumPyArrayToTable(principal_components, principal_components_output)
            
        '''if explained_variances.ndim is 2 or explained_variances.ndim is 3:
            arcpy.NumPyArrayToRaster(explained_variances).save(explained_variances_output)
        else:
            arcpy.da.NumPyArrayToTable(explained_variances, explained_variances_output)
            
        if explained_variance_ratios.ndim is 2 or explained_variance_ratios.ndim is 3:
            arcpy.NumPyArrayToRaster(explained_variance_ratios).save(explained_variance_ratios_output)
        else:
            arcpy.da.NumPyArrayToTable(explained_variance_ratios, explained_variance_ratios_output)'''
            
    elif isinstance(transformed_data, pd.DataFrame):
        arcpy.da.NumPyArrayToTable(transformed_data.to_numpy(), transformed_data_output)

    return

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
        raise arcpy.AddError("Invalid nodata_handling value. Choose 'remove' or 'replace'.")

def _compute_pca(
    feature_matrix, number_of_components, scaler_type
):
    # Standardize the data
    if scaler_type == "standard":
        mean = np.mean(feature_matrix, axis=0)
        std = np.std(feature_matrix, axis=0)
        scaled_data = (feature_matrix - mean) / std
    elif scaler_type == "min_max":
        min_val = np.min(feature_matrix, axis=0)
        max_val = np.max(feature_matrix, axis=0)
        scaled_data = (feature_matrix - min_val) / (max_val - min_val)
    elif scaler_type == "robust":
        median = np.median(feature_matrix, axis=0)
        q75, q25 = np.percentile(feature_matrix, [75 ,25], axis=0)
        iqr = q75 - q25
        scaled_data = (feature_matrix - median) / iqr
    else:
        raise arcpy.AddError(f"Invalid scaler. Choose from: 'standard', 'min_max', 'robust'")

    # Compute PCA using numpy and scipy
    cov_matrix = np.cov(scaled_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_index]
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    
    # Select the top n_components
    principal_components = sorted_eigenvectors[:, :number_of_components]
    explained_variances = sorted_eigenvalues[:number_of_components]
    explained_variance_ratios = explained_variances / np.sum(sorted_eigenvalues)
    
    # Transform the data
    transformed_data = np.dot(scaled_data, principal_components)

    return transformed_data, principal_components, explained_variances, explained_variance_ratios

def compute_pca(
    data,
    number_of_components = None,
    columns = None,
    scaler_type = "standard", # : Literal["standard", "min_max", "robust"]
    nodata_handling = "remove", # Literal["remove", "replace"]
    nodata = None
):
    """
    Compute defined number of principal components for numeric input data and transform the data.

    Before computation, data is scaled according to specified scaler and NaN values removed or replaced.
    Optionally, a nodata value can be given to handle similarly as NaN values.

    If input data is a Numpy array, interpretation of the data depends on its dimensions.
    If array is 3D, it is interpreted as a multiband raster/stacked rasters format (bands, rows, columns).
    If array is 2D, it is interpreted as table-like data, where each column represents a variable/raster band
    and each row a data point (similar to a Dataframe).

    Args:
        data: Input data for PCA.
        number_of_components: The number of principal components to compute. Should be >= 1 and at most
            the number of features found in input data. If not defined, will be the same as number of
            features in data. Defaults to None.
        columns: Select columns used for the PCA. Other columns are excluded from PCA, but added back
            to the result Dataframe intact. Only relevant if input is (Geo)Dataframe. Defaults to None.
        scaler_type: Transform data according to a specified Sklearn scaler.
            Options are "standard", "min_max" and "robust". Defaults to "standard".
        nodata_handling: If observations with nodata (NaN and given `nodata`) should be removed for the time
            of PCA computation or replaced with column/band mean. Defaults to "remove".
        nodata: Define a nodata value to remove. Defaults to None.

    Returns:
        The transformed data in same format as input data, computed principal components, explained variances
        and explained variance ratios for each component.

    Raises:
        EmptyDataException: The input is empty.
        InvalidColumnException: Selected columns are not found in the input Dataframe.
        InvalidNumberOfPrincipalComponents: The number of principal components is less than 1 or more than
            number of columns if input was (Geo)DataFrame.
        InvalidParameterValueException: If value for `number_of_components` is invalid.
    """

    if number_of_components is not None and number_of_components < 1:
        raise arcpy.AddError("The number of principal components should be >= 1.")

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
        raise arcpy.AddError(
            f"Unsupported input data format. {feature_matrix.ndim} dimensions detected for given array."
        )

    '''elif isinstance(data, pd.DataFrame):
        df = data.copy()
        if df.empty:
            raise arcpy.AddError("Input DataFrame is empty.")
        if isinstance(data, gpd.GeoDataFrame):
            geometries = data.geometry
            crs = data.crs
            df = df.drop(columns=["geometry"])
        if columns is not None and columns != []:
            df = df[columns]

        df = df.convert_dtypes()
        df = df.apply(pd.to_numeric, errors="ignore")
        df = df.select_dtypes(include=np.number)
        df = df.astype(dtype=np.number)
        feature_matrix = df.to_numpy()
        feature_matrix = feature_matrix.astype(float)
        feature_matrix, nan_mask = _handle_missing_values(feature_matrix, nodata_handling, nodata)'''

    # Default number of components to number of features in data if not defined
    if number_of_components is None:
        number_of_components = feature_matrix.shape[1]

    if number_of_components > feature_matrix.shape[1]:
        arcpy.AddError(
            "The number of principal components is too high for the given input data "
            + f"({number_of_components} > {feature_matrix.shape[1]})."
        )

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