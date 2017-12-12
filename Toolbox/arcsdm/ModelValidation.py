"""
    Model Validation tool

    This module contains the execution code for the tool Apply Model
    - Just the execute function should be called from the exterior, all other functions are called by this.
    - This module makes use of the non-standard modules:
        arcpy: for GIS operations and communication with ArcGIS. Is distributed along with ArcGIS desktop software


    Authors: Irving Cabrera <irvcaza@gmail.com>
"""
import arcpy
import numpy as np
import sys
import sklearn.metrics as sklm
import matplotlib.pyplot as plt


# Global Name for the messages object and be called from any function
MESSAGES = None

# Verbosity switch, True to print more detailed information
VERBOSE = False
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
    # TODO: Implement this
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
        fi = arcpy.da.FeatureClassToNumPyArray(feature_layer, fields_name, null_value=sys.maxsize)

    # Cast to floating point numbers
    field = np.array([[elem * 1.0 for elem in row] for row in fi])

    # If only one field is given the matrix needs to be flattened  to a vector
    if not isinstance(fields_name, list) or len(fields_name) == 1:
        field = field.flatten()
    # Assign NAN to the numbers with maximum integer value
    field[field == sys.maxsize] = np.NaN

    return field


def _extract_fields(base, rasters):
    """
    _extract_fields 
        Extracts the values of rasters to add the information of its value to the given points 
    :param base: Set of points to add the information
    :param rasters: Rasters with the information 
    :return: name of the feature with the information, name of the added fields
    """
    global MESSAGES
    MESSAGES.AddMessage("Assigning Raster information...")
    _verbose_print("Base: {}".format(base))
    _verbose_print("Rasters: {}".format(rasters))

    # Make a list of the rasters
    rasters = [x.strip("'") for x in rasters.split(";")]
    scratch_files = []

    try:
        # Initialize progress bar
        regressor_names = []
        arcpy.SetProgressor("step", "Adding raster values to the points", min_range=0, max_range=len(rasters),
                            step_value=1)
        _verbose_print("Adding raster values to the points")
        for raster in rasters:
            _verbose_print("Adding information of {}".format(raster))
            extracted_name = arcpy.CreateScratchName("temp", data_type="FeatureClass",
                                                     workspace=arcpy.env.scratchWorkspace)
            # Add the information of the raster to the points
            arcpy.gp.ExtractValuesToPoints(base, raster, extracted_name, "INTERPOLATE",
                                           "VALUE_ONLY")
            _verbose_print("Scratch file created (merge): {}".format(extracted_name))
            scratch_files.append(extracted_name)
            # Rename field to coincide with the raster name
            arcpy.AlterField_management(extracted_name, "RASTERVALU", arcpy.Describe(raster).baseName)
            regressor_names.append(arcpy.Describe(raster).baseName)
            base = extracted_name
            arcpy.SetProgressorPosition()

        # Reset progress bar
        scratch_files.remove(extracted_name)
        arcpy.SetProgressorLabel("Executing Enrich Points")
        arcpy.ResetProgressor()
    except:
        raise
    finally:
        # Delete intermediate files
        for s_file in scratch_files:
            arcpy.Delete_management(s_file)
            _verbose_print("Scratch file deleted: {}".format(s_file))
    return extracted_name, regressor_names


def _plot_results(classification_model, des_func, probabilities, response_test, threshold, plot_file):
    """ 
    
    :param classification_model: trained classifier 
    :param des_func: expected values of the response
    :param probabilities: probabilities of the examples to be prospective
    :param response_test: real response of the model for the test set 
    :param threshold: threshold to consider prospective or non prospective
    :param plot_file: path of the output plot (none for no output)
    :return: none
    """
    global MESSAGES

    MESSAGES.AddMessage("Plotting Results...")
    _verbose_print("_print_test_results")
    _verbose_print("classification_model: {}".format(classification_model))
    _verbose_print("threshold: {}".format(threshold))
    _verbose_print("plot_file: {}".format(plot_file))

    # Calculate the points for the ROC and precision-recall curve
    fpr, tpr, thresholds = sklm.roc_curve(response_test, des_func)
    pre, rec, unused = sklm.precision_recall_curve(response_test, probabilities)
    # Create figure for plots
    fig, ((ax_roc, ax_prc), (ax_suc, unused)) = plt.subplots(2, 2, figsize=(12, 12))
    # Create ROC plot
    ax_roc.plot(fpr, tpr, label="ROC Curve (AUC={0:.3f})".format(sklm.roc_auc_score(response_test, des_func)))
    ax_roc.plot([0, 1], [0, 1], linestyle='--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc='lower right', prop={'size': 12})
    # Create precision-recall plot
    ax_prc.plot(rec, pre, label="Precision-Recall Curve "
                                "(Area={0:.3f})".format(sklm.average_precision_score(response_test, probabilities)))
    ax_prc.set_xlabel('Recall')
    ax_prc.set_ylabel("Precision")
    ax_prc.legend(loc='lower right', prop={'size': 12})

    # Import the raster for data about the prospective areas
    try:
        raster_array = arcpy.RasterToNumPyArray(classification_model, nodata_to_value=np.NaN)
    except ValueError:
        _verbose_print("Integer type raster, changed to float")
        raster_array = 1.0 * arcpy.RasterToNumPyArray(classification_model, nodata_to_value=sys.maxsize)
        raster_array[raster_array == sys.maxsize] = np.NaN
    # Calculate prospective area
    total_area = np.sum(np.isfinite(raster_array))
    areas = [(1.0 * np.sum(raster_array > thr)) / total_area for thr in thresholds]

    MESSAGES.AddMessage(
        "Proportion of prospective area: {}".format((1.0 * np.sum(raster_array > threshold)) / total_area))
    # Add plot of success curve
    ax_suc.plot(areas, tpr, label="Success Curve ")
    ax_suc.set_xlabel('Proportion of area covered')
    ax_suc.set_ylabel("True Positive Rate")
    ax_suc.legend(loc='lower right', prop={'size': 12})

    if not plot_file.endswith(".png"):
        plot_file += ".png"
    # Save figure
    plt.savefig(plot_file)
    _verbose_print("Figure saved {}".format(plot_file))
    # Close figure. This is important, otherwise arcMAP sends an error when is closed
    plt.close(fig)

    return


def _print_test_results(classification_model, test_points, test_response_name, plot_file, threshold):
    """
        _print_test_results
            Performs the evaluation and prints the results 
            
    :param classification_model: Raster with the response function  
    :param test_points: Point to be used for the evaulations 
    :param test_response_name: Name of the field with the expected response 
    :param plot_file: Name of the file to be saves. None to not save any
    :param threshold: Threshold value. Responses higher than this value are considered prospective and under this 
        value are non-prospective
    :return: Null
    """
    global MESSAGES
    _verbose_print("_print_test_results")
    _verbose_print("classification_model: {}".format(classification_model))
    _verbose_print("test_points: {}".format(test_points))
    _verbose_print("test_response_name: {}".format(test_response_name))
    _verbose_print("plot_file: {}".format(plot_file))
    _verbose_print("threshold: {}".format(threshold))

    scratch_files = []

    # Obtain the regressors and calculate the response
    try:
        extracted_name, regressors = _extract_fields(test_points, classification_model)
        scratch_files.append(extracted_name)
        response = _get_fields(extracted_name, regressors)
        finite_elements_resp = np.isfinite(response)
        if any(np.logical_not(finite_elements_resp)):
            MESSAGES.addWarningMessage("Input points found in areas with NaN infinity or too large number")
            response = response[finite_elements_resp]

    except:
        raise
    finally:
        # Delete intermediate files
        for s_file in scratch_files:
            arcpy.Delete_management(s_file)
            _verbose_print("Scratch file deleted: {}".format(s_file))

    # obtain the expected response
    response_test = _get_fields(test_points, test_response_name)
    response_test = response_test[finite_elements_resp]
    finite_elements_test = np.isfinite(response_test)
    if any(np.logical_not(finite_elements_test)):
        MESSAGES.addWarningMessage("Class field contains NaN infinity or too large numbers")
        response_test = response_test[finite_elements_test]
        response = response[finite_elements_test]
    # Calculate the prediction
    predicted = np.sign(response - threshold)
    # Calculate the probabilities
    # TODO: Im not sure if this is completely correct
    probabilities = (response - min(response)) / (max(response) - min(response))

    # Plot and save results
    if plot_file is not None:
        _plot_results(classification_model, response, probabilities, response_test, threshold, plot_file)

    # Display results
    MESSAGES.AddMessage("Accuracy: {}".format(sklm.accuracy_score(response_test, predicted)))
    MESSAGES.AddMessage("Precision: {}".format(sklm.precision_score(response_test, predicted)))
    MESSAGES.AddMessage("Recall: {}".format(sklm.recall_score(response_test, predicted)))
    MESSAGES.AddMessage("Area Under the curve (AUC): {}".format(sklm.roc_auc_score(response_test, response)))
    MESSAGES.AddMessage(
        "Average Precision Score: {}".format(sklm.average_precision_score(response_test, probabilities)))
    # Display confusion matrix
    confusion = sklm.confusion_matrix(response_test, predicted)
    MESSAGES.AddMessage("Confusion Matrix :")
    labels = ["Non Deposit", "Deposit"]
    row_format = "{:6}" + "{:^16}" * (len(labels) + 1)
    MESSAGES.AddMessage(row_format.format("", "", "Predicted", ""))
    MESSAGES.AddMessage(row_format.format("True", "", *labels))
    for label, row in zip(labels, confusion):
        MESSAGES.AddMessage(row_format.format("", label, *row))

    return


def execute(self, parameters, messages):
    """
        Model Validation Tool
            Evaluates a model in a raster format and delivers results of the tests 
            
        :param parameters: parameters object with all the parameters from the python-tool. It contains:
            test_points: Point to be evaluated 
            classification_model: Raster with the response function of the model 
            test_response_name: Field in the test points that indicates the belonging class 
            plot_file: Path of the file to save the plots. Null value does no output any file 
            threshold: Threshold value. Responses higher than this value are considered prospective and under this 
        value are non-prospective
        :param messages: messages object to print in the console, must implement AddMessage
             
        :return: None
    """

    global MESSAGES
    MESSAGES = messages

    print_parameters(parameters)

    _input_validation(parameters)

    parameter_dic = {par.name: par for par in parameters}
    test_points = parameter_dic["test_points"].valueAsText
    classification_model = parameter_dic["classification_model"].valueAsText
    test_response_name = parameter_dic["test_response_name"].valueAsText
    plot_file = parameter_dic["plot_file"].valueAsText
    threshold = parameter_dic["threshold"].value

    _print_test_results(classification_model, test_points, test_response_name, plot_file, threshold)

    return
