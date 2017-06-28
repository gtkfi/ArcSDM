import arcpy
import numpy as np
import sys
import sklearn.metrics as sklm
import matplotlib.pyplot as plt

# TODO: Add documentation

VERBOSE = False
MESSAGES = None

if VERBOSE:
    def _verbose_print(text):
        global  MESSAGES
        MESSAGES.AddMessage("Verbose: " + text)
else:
    _verbose_print = lambda *a: None


def print_parameters(parameters):
    for var, par in enumerate(parameters):
        _verbose_print("parameters[" + str(var) + "](" + str(par.name) + "): " + str(par.valueAsText))


def _input_validation(parameters):

    return


def _get_fields(feature_layer, fields_name):

    _verbose_print("feature_layer: {}".format(feature_layer))
    _verbose_print("fields_name: {}".format(fields_name))

    try:
        fi = arcpy.da.FeatureClassToNumPyArray(feature_layer, fields_name)
    except TypeError:
        _verbose_print("Failed importing with nans, possibly a integer feature class")
        fi = arcpy.da.FeatureClassToNumPyArray(feature_layer, fields_name, null_value=sys.maxint)

    field = np.array([[elem * 1.0 for elem in row] for row in fi])

    if not isinstance(fields_name, list) or len(fields_name) == 1:
        field = field.flatten()
    field[field == sys.maxint] = np.NaN

    return field


def _extract_fields(base, rasters):
    global MESSAGES
    MESSAGES.AddMessage("Assigning Raster information...")
    _verbose_print("Base: {}".format(base))
    _verbose_print("Rasters: {}".format(rasters))

    rasters = [x.strip("'") for x in rasters.split(";")]
    scratch_files = []

    try:
        regressor_names = []
        arcpy.SetProgressor("step", "Adding raster values to the points", min_range=0, max_range=len(rasters),
                            step_value=1)
        _verbose_print("Adding raster values to the points")
        for raster in rasters:
            try:
                _verbose_print("Adding information of {}".format(raster))
                extracted_name = arcpy.CreateScratchName("temp", data_type="FeatureClass",
                                                         workspace=arcpy.env.scratchWorkspace)
                arcpy.gp.ExtractValuesToPoints(base, raster, extracted_name, "INTERPOLATE",
                                               "VALUE_ONLY")
                _verbose_print("Scratch file created (merge): {}".format(extracted_name))
                scratch_files.append(extracted_name)
                arcpy.AlterField_management(extracted_name, "RASTERVALU", arcpy.Describe(raster).baseName)
                regressor_names.append(arcpy.Describe(raster).baseName)
                base = extracted_name
                arcpy.SetProgressorPosition()
            except:
                MESSAGES.addErrorMessage("Problem with raster {}".format(raster))
                raise
        scratch_files.remove(extracted_name)
        arcpy.SetProgressorLabel("Executing Enrich Points")
        arcpy.ResetProgressor()
    except:
        raise
    finally:
        for s_file in scratch_files:
            arcpy.Delete_management(s_file)
            _verbose_print("Scratch file deleted: {}".format(s_file))
    return extracted_name, regressor_names


def _print_test_results(classification_model, test_points, test_response_name, plot_file, threshold):
    global MESSAGES

    scratch_files = []

    try:
        extracted_name, regressors = _extract_fields(test_points, classification_model)
        scratch_files.append(extracted_name)
        response = _get_fields(extracted_name, regressors)
    except:
        raise
    finally:
        for s_file in scratch_files:
            arcpy.Delete_management(s_file)
            _verbose_print("Scratch file deleted: {}".format(s_file))

    response_test = _get_fields(test_points, test_response_name)
    predicted = np.sign(response - threshold)
    probabilities = (response - min(response)) / (max(response) - min(response))

    if plot_file is not None:
        _plot_results(classification_model, response, probabilities, response_test, threshold, plot_file)

    MESSAGES.AddMessage("Accuracy : {}".format(sklm.accuracy_score(response_test, predicted)))
    MESSAGES.AddMessage("Precision : {}".format(sklm.precision_score(response_test, predicted)))
    MESSAGES.AddMessage("Recall: {}".format(sklm.recall_score(response_test, predicted)))
    MESSAGES.AddMessage("Area Under the curve (AUC): {}:".format(sklm.roc_auc_score(response_test, response)))
    MESSAGES.AddMessage(
        "Average Precision Score: {}".format(sklm.average_precision_score(response_test, probabilities)))

    confusion = sklm.confusion_matrix(response_test, predicted)
    MESSAGES.AddMessage("Confusion Matrix :")
    labels = ["Non Deposit", "Deposit"]
    row_format = "{:6}" + "{:^16}" * (len(labels) + 1)
    MESSAGES.AddMessage(row_format.format("", "", "Predicted", ""))
    MESSAGES.AddMessage(row_format.format("True", "", *labels))
    for label, row in zip(labels, confusion):
        MESSAGES.AddMessage(row_format.format("", label, *row))


    return


def _plot_results(classification_model, des_func, probabilities, response_test, threshold, plot_file):

    global MESSAGES

    MESSAGES.AddMessage("Plotting Results...")

    fpr, tpr, thresholds = sklm.roc_curve(response_test, des_func)
    pre, rec, unused = sklm.precision_recall_curve(response_test, probabilities)

    fig, ((ax_roc, ax_prc),(ax_suc, unused)) = plt.subplots(2,2, figsize=(12,12))

    ax_roc.plot(fpr, tpr, label="ROC Curve (AUC={0:.3f})".format(sklm.roc_auc_score(response_test, des_func)))
    ax_roc.plot([0, 1], [0, 1], linestyle='--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc='lower right', prop={'size':12})


    ax_prc.plot(rec, pre, label="Precision-Recall Curve "
                                "(Area={0:.3f})".format(sklm.average_precision_score(response_test, probabilities)))
    ax_prc.set_xlabel('Recall')
    ax_prc.set_ylabel("Precision")
    ax_prc.legend(loc='lower right', prop={'size':12})

    try:
        raster_array = arcpy.RasterToNumPyArray(classification_model, nodata_to_value=np.NaN)
    except ValueError:
        _verbose_print("Integer type raster, changed to float")
        raster_array = 1.0 * arcpy.RasterToNumPyArray(classification_model, nodata_to_value=sys.maxint)
        raster_array[raster_array == sys.maxint] = np.NaN

    total_area = np.sum(np.isfinite(raster_array))
    areas = [(1.0*np.sum(raster_array > thr))/total_area for thr in thresholds]

    MESSAGES.AddMessage("Proportion of prospective area: {}".format((1.0*np.sum(raster_array > threshold))/total_area))

    ax_suc.plot(areas, tpr, label="Success Curve ")
    ax_suc.set_xlabel('Proportion of area covered')
    ax_suc.set_ylabel("True Positive Rate")
    ax_suc.legend(loc='lower right', prop={'size':12})

    if not plot_file.endswith(".png"):
        plot_file += ".png"

    plt.savefig(plot_file)
    _verbose_print("Figure saved {}".format(plot_file))

    return


def execute(self, parameters, messages):

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
