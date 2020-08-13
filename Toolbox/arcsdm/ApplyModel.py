"""
    Apply Model tool

    This module contains the execution code for the tool Apply Model
    - Just the execute function should be called from the exterior, all other functions are called by this.
    - This module makes use of the non-standard modules:
        arcpy: for GIS operations and communication with ArcGIS. Is distributed along with ArcGIS desktop software
        sklearn: for models training, for more information visit http://scikit-learn.org/stable/index.html


    Authors: Irving Cabrera <irvcaza@gmail.com>
"""


import arcpy
import numpy as np
import sys
from sklearn.externals import joblib
import os

# Global Name for the messages object and be called from any function
MESSAGES = None

# Verbosity switch, True to print more detailed information
VERBOSE = True
if VERBOSE:
    def _verbose_print(text):
        global MESSAGES
        MESSAGES.AddMessage("Verbose: " + text)
else:
    _verbose_print = lambda *a: None


def print_parameters(parameters):
    """ 
        print_parameters
            Prints the element in the parameters object. Needs to have verbosity activated
        :param parameters: Object with the attributes name and valueAsText
        :return: none 
    """
    for var, par in enumerate(parameters):
        _verbose_print("parameters[" + str(var) + "](" + str(par.name) + "): " + str(par.valueAsText))


def _input_validation(parameters):
    """
        _input_validation
            Makes correctness checks before making the execution and raise an error if one of the checks fails
        :param parameters: Parameters object with all the parameters for the function
        :return: none 
    """
    parameter_dic = {par.name: par for par in parameters}
    input_model = parameter_dic["input_model"].valueAsText
    rasters = parameter_dic["info_rasters"].valueAsText
    global MESSAGES

    # Get raster band names
    oldws = arcpy.env.workspace  # Save previous workspace
    arcpy.env.workspace = arcpy.Describe(rasters.strip("'")).catalogPath
    rasters_list = arcpy.ListRasters()
    arcpy.env.workspace = oldws  # Restore previous workspace

    # Get regressors' names from the file
    input_text = input_model.replace(".pkl", ".txt")
    model_regressors = []
    with open(input_text, "r") as f:
        for line in f:
            if line.startswith("Regressor:"):
                model_regressors.append(line.split("'")[1])

    # Check if the amount of regressors in the raster and the model coincide
    if len(model_regressors) != len(rasters_list):
        raise ValueError("The amount of {} does not coincide with the model ({} vs {})".format(
            parameter_dic["info_rasters"].displayName, len(rasters_list), len(model_regressors)))

    row_format = "{:^16}" * 2
    MESSAGES.AddMessage("Parameters association")
    MESSAGES.AddMessage(row_format.format("Model", "Rasters"))
    for m_r, t_r in zip(model_regressors, rasters_list):
        MESSAGES.AddMessage(row_format.format(m_r, t_r))

    return


def create_response_raster(classifier, rasters, output, scale):
    """
        create_response_raster
            Uses a classifier in a multiband raster to obtain the response raster
            
        :param classifier: Classifier object previously trained
        :param rasters: Multiband raster with the information to be put in the classifier 
        :param output: Name of the file for the response raster
        :param scale: Object with the transformation to be done to normalize the data. If None, then no transformation 
            is made
        :return: None 
    """
    # Obtain spatial information from the raster
    spatial_reference = arcpy.Describe(rasters).spatialReference

    raster = arcpy.Raster(rasters)

    lower_left_corner = arcpy.Point(raster.extent.XMin, raster.extent.YMin)
    x_cell_size = raster.meanCellWidth
    y_cell_size = raster.meanCellHeight
    # Import the multiband raster to numpy array if the raster is of type integer it will throw an error because of the
    # NaN values, then is imported wit NaNs as maximum integers, casted to float and then the NaNs applied
    try:
        raster_array = arcpy.RasterToNumPyArray(rasters, nodata_to_value=np.NaN)
    except ValueError:
        _verbose_print("Integer type raster, changed to float")
        raster_array = 1.0 * arcpy.RasterToNumPyArray(rasters, nodata_to_value=sys.maxint)
        raster_array[raster_array == sys.maxint] = np.NaN

    MESSAGES.AddMessage("Creating response raster...")

    # If it is a single band, then the raster must be reshaped as a tri-dimensional numpy array
    if raster_array.ndim == 2:
        raster_array = np.reshape(raster_array,[1]+ list(raster_array.shape))

    n_regr = raster_array.shape[0]
    n_rows = raster_array.shape[1]
    n_cols = raster_array.shape[2]
    _verbose_print("{} regressors of shape {} x {}".format(n_regr,n_rows,n_cols))

    # The raster looping must be done in the last dimension
    raster_array2 = np.empty([n_rows, n_cols, n_regr])
    ### MODIFICATION ###
    #arr_filepath = os.path.join(arcpy.env.workspace, 'model_arr.arr')
    #raster_array2 = np.memmap(arr_filepath, dtype=np.float64, mode='w+',
             # shape=(n_rows, n_cols, n_regr))
    try:
        for raster_index in xrange(n_regr):
            raster_array2[:, :, raster_index] = raster_array[raster_index, :, :]
    except:
        for raster_index in range(n_regr):
            raster_array2[:, :, raster_index] = raster_array[raster_index, :, :]

    # The matrix is reshaped from 3D to 2D
    raster_array2 = np.reshape(raster_array2, [n_rows * n_cols, n_regr])

    #_verbose_print(str(raster_array2[0]))
    #_verbose_print(str(np.isfinite(raster_array2[0])))

    # Create a mask where the values of all regressors are finite. The calculations will be made just there
    finite_mask = np.all(np.isfinite(raster_array2), axis=1)
    #_verbose_print(str(finite_mask))
    nan_mask = np.logical_not(finite_mask)
    _verbose_print("{} elements will be calculated and {} let as NaN".format(sum(finite_mask), sum(nan_mask)))
    # Make the transformation if available
    if scale is None:
        finite_array = raster_array2[finite_mask]
        _verbose_print("Data not normalized")
    else:
        finite_array = scale.transform(raster_array2[finite_mask])
        MESSAGES.AddMessage("Data normalized")
    # Calculate decision_function for the elements with finite values. If not available use  predict_proba
    # TODO: Measure time
    if getattr(classifier, "decision_function", None) is None:
        _verbose_print("Decision function not available, probabilities used instead")
        responses = classifier.predict_proba(finite_array)[:, classifier.classes_ == 1]
    else:
        _verbose_print("Decision function used")
        responses = classifier.decision_function(finite_array)

    # Reshape the matrix with the response values
    response_vector = np.empty(n_rows * n_cols)
    response_vector[finite_mask] = responses
    response_vector[nan_mask] = -9
    response_array = np.reshape(response_vector, [n_rows, n_cols])
    # Known bug: There is a minimal displacement between the input layers and the response
    response_raster = arcpy.NumPyArrayToRaster(response_array, lower_left_corner=lower_left_corner,
                                               x_cell_size=x_cell_size, y_cell_size=y_cell_size, value_to_nodata=-9)
    response_raster.save(output)
    # Known bug: if the output name is too long (the name not the path) then arcpy shows "unexpected error"
    MESSAGES.AddMessage("Raster file created in " + output)
    arcpy.DefineProjection_management(output, spatial_reference)

    return


def _get_fields(feature_layer, fields_name):
    """
        _get_fields
            Gets all the values for the given fields and returns a numpy-array with the values 
        :param feature_layer: Name of the feature layer to extract the values
        :param fields_name: List with the name of the fields to extract features
        :return: 
    """
    _verbose_print("feature_layer: {}".format(feature_layer))
    _verbose_print("fields_name: {}".format(fields_name))

    # Since integers does not support NAN values, the method throws an error, then NAN values are assigned to the
    # maximum integer
    try:
        fi = arcpy.da.FeatureClassToNumPyArray(feature_layer, fields_name)
    except TypeError:
        _verbose_print("Failed importing with nans, possibly a integer feature class")
        fi = arcpy.da.FeatureClassToNumPyArray(feature_layer, fields_name, null_value=sys.maxint)

    # Cast to floating point numbers
    field = np.array([[elem * 1.0 for elem in row] for row in fi])

    # If only one field is given the matrix needs to be flattened  to a vector
    if not isinstance(fields_name, list):
        field = field.flatten()
    # Assign NAN to the numbers with maximum integer value
    field[field == sys.maxsize] = np.NaN

    return field


def execute(self, parameters, messages):
    """
        Apply Model tool
            Uses a trained classifier in a multiband raster to obtain the response raster
            
        :param parameters: parameters object with all the parameters from the python-tool. It contains:
            input_model: (path) Name of the file with the model to be used 
            info_rasters: (Multiband raster) Name of the raster that will be used as information source
            output_map: (raster) name of the raster that will be output by the tool
        :param messages: messages object to print in the console, must implement AddMessage
         
        :return: None
         
        Notes:
        - The resolution of the response function is given by the resolution of the multiband raster 
        - The information raster must be the same as the one used to train the model 
    """

    global MESSAGES
    MESSAGES = messages

    # Print parameters for debugging purposes
    print_parameters(parameters)

    # Check for correctness in the parameters
    _input_validation(parameters)

    parameter_dic = {par.name: par for par in parameters}
    input_model = parameter_dic["input_model"].valueAsText
    info_rasters = parameter_dic["info_rasters"].valueAsText
    output_map = parameter_dic["output_map"].valueAsText

    # Import the classifier
    classifier = joblib.load(input_model)
    # Look for a normalization file, that indicates that the model was trained with normalized data
    try:
        scale = joblib.load(input_model.replace(".pkl", "_scale.pkl"))
    except:
        scale = None

    create_response_raster(classifier, info_rasters, output_map, scale)

    return