import sys
import arcpy
import numpy as np
import pandas as pd
import sklearn.neural_network
from typing import Optional, Sequence
from sklearn.inspection import permutation_importance

from utils.input_to_numpy_array import input_to_numpy_arrays

def Execute(self, parameters, messages):
    """The source code of the tool."""

    #try:
        
    model = parameters[0].valueAsText
    input_data = parameters[1].valueAsText.split(';')
    target_data = parameters[2].valueAsText.split(';')
    feature_names = parameters[3].valueAsText.split(';')
    n_repeats = parameters[4].value
    random_state = parameters[5].value
    output_table = parameters[6].valueAsText

    input_as_arrays = input_to_numpy_arrays(input_data)
    target_as_array = input_to_numpy_arrays(target_data)   
    
    stacked_array = np.stack(input_as_arrays, axis=0)

    target_as_np_array = np.array(target_as_array)
    
    feature_importance, result = _evaluate_feature_importance(
                                                            model, 
                                                            stacked_array, 
                                                            target_as_np_array, 
                                                            feature_names, 
                                                            n_repeats, 
                                                            random_state)
    
    arcpy.da.NumPyArrayToTable(np.array(feature_importance.to_records(index=False)), output_table)
    
    arcpy.AddMessage('='*5 + ' Feature Importance finished' + '='*5)
    
    arcpy.AddMessage(f'Output table saved to: {output_table}')
    
    arcpy.AddMessage(f'Permutation importance:')
    arcpy.AddMessage(result)
        
    '''except arcpy.ExecuteError:
        arcpy.AddError(arcpy.GetMessages(2)) 
    
    except:
        e = sys.exc_info()[1]
        arcpy.AddError(e.args[0])'''

def _evaluate_feature_importance(
    model: sklearn.base.BaseEstimator,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    n_repeats: Optional[int] = 50,
    random_state: Optional[int] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate the feature importance of a sklearn classifier or regressor.

    Args:
        model: A trained and fitted Sklearn model.
        x_test: Testing feature data (X data need to be normalized / standardized).
        y_test: Testing label data.
        feature_names: Names of the feature columns.
        n_repeats: Number of iteration used when calculate feature importance. Defaults to 50.
        random_state: random state for repeatability of results. Optional parameter.

    Returns:
        A dataframe containing features and their importance.
        A dictionary containing importance mean, importance std, and overall importance.

    Raises:
        InvalidDatasetException: Either array is empty.
        InvalidParameterValueException: Value for 'n_repeats' is not at least one.
    """
    if x_test.size == 0:
        arcpy.AddError("Array 'x_test' is empty.")
        raise arcpy.ExecuteError

    if y_test.size == 0:
        arcpy.AddError("Array 'y_test' is empty.")
        raise arcpy.ExecuteError

    if n_repeats and n_repeats < 1:
        arcpy.AddError("Value for 'n_repeats' is less than one.")
        raise arcpy.ExecuteError

    result = permutation_importance(model, x_test, y_test.ravel(), n_repeats=n_repeats, random_state=random_state)

    feature_importance = pd.DataFrame({"Feature": feature_names, "Importance": result.importances_mean * 100})

    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

    return feature_importance, result