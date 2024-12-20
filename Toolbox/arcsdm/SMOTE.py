import numpy as np
import pandas as pd
import arcpy
import sys
from typing import Any, List, Optional, Union
from imblearn.combine import SMOTETomek
from utils.input_to_numpy_array import read_and_stack_rasters



"""@beartype"""

def execute_smote(self, parameters, messages):
    #try:
    input_file = parameters[0].valueAsText.split(';')
    y_data = parameters[1].valueAsText.split(';')
    sampling_strategy = parameters[2].valueAsText
    random_state = parameters[3].value
    out_x = parameters[4].valueAsText

    stacked_arrays = []
    """#for input_file in x_data:
    desc = arcpy.Describe(input_file)
    if desc.dataType == "RasterDataset" or desc.dataType == "RasterLayer":
        stacked_arrays.append(read_and_stack_rasters(input_file, nodata_handling = "convert_to_nan"))
    else:
        muuttuja = arcpy.da.FeatureClassToNumPyArray(input_file)
        muuttuja_stacked = np.stack(muuttuja, axis=0)
        stacked_arrays.append(muuttuja_stacked)
"""
    arcpy.AddWarning(f"Input file: {input_file}")
    desc = arcpy.Describe(input_file[0])
    band_data = arcpy.RasterToNumPyArray(desc.catalogPath)
    #band_data = band_data.astype(float)
    arcpy.AddWarning(f"Band data: {band_data}")
    target_as_array = read_and_stack_rasters(y_data)  

    balance_SMOTETomek(band_data, target_as_array, sampling_strategy, random_state)

    """except arcpy.ExecuteError:
        arcpy.AddError(arcpy.GetMessages(2)) 

    except:
        e = sys.exc_info()[1]
        arcpy.AddError(e.args[0])
"""

def balance_SMOTETomek(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    sampling_strategy: Union[float, str, dict] = "auto",
    random_state: Optional[int] = None,
) -> tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
    """Balances the classes of input dataset using SMOTETomek resampling method.

    Args:
        X: The feature matrix (input data as a DataFrame).
        y: The target labels corresponding to the feature matrix.
        sampling_strategy: Parameter controlling how to perform the resampling.
            If float, specifies the ratio of samples in minority class to samples of majority class,
            if str, specifies classes to be resampled ("minority", "not minority", "not majority", "all", "auto"),
            if dict, the keys should be targeted classes and values the desired number of samples for the class.
            Defaults to "auto", which will resample all classes except the majority class.
        random_state: Parameter controlling randomization of the algorithm. Can be given a seed (number).
            Defaults to None, which randomizes the seed.

    Returns:
        Resampled feature matrix and target labels.

    Raises:
        NonMatchingParameterLengthsException: If X and y have different length.
    """
  

    """if len(X) != len(y):
        raise NonMatchingParameterLengthsException("Feature matrix X and target labels y must have the same length.")"""
    
    x_size = X.index.size if isinstance(X, pd.DataFrame) else X.shape[0]
    arcpy.AddWarning(f"X size: {x_size}")
    arcpy.AddWarning(f"y size: {y.size}")
    arcpy.AddWarning(f"y.shape: {y.shape[0]}")
    arcpy.AddWarning(y)
    arcpy.AddWarning(type(y[0]))
    arcpy.AddWarning(y.size)
    #y=y.astype(int)
    y[~np.isnan(y)] = 1
    y[np.isnan(y)] = 0
    arcpy.AddWarning(y)
    """if x_size != y.size:
        arcpy.AddError(f"X and y must have the length {x_size} != {y.size}.")
        raise arcpy.ExecuteError
"""
    X_res, y_res = SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_state).fit_resample(X, y)
    return X_res, y_res

