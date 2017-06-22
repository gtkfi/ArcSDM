import arcpy
import numpy as np
import sys
from sklearn.externals import joblib
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

    parameter_dic = {par.name: par for par in parameters}
    input_model = parameter_dic["input_model"].valueAsText
    test_regressors_name = [x.strip("'") for x in parameter_dic["test_regressors_name"].valueAsText.split(";")]
    global MESSAGES

    input_text = input_model.replace(".pkl", ".txt")
    model_regressors = []
    with open(input_text, "r") as f:
        for line in f:
            if line.startswith("Regressor:"):
                model_regressors.append(line.split("'")[1])

    if len(model_regressors) != len(test_regressors_name):
        raise ValueError("The amount of {} does not coincide with the model ({} vs {})".format(
            parameter_dic["test_regressors_name"].displayName, len(test_regressors_name), len(model_regressors)))

    row_format = "{:^16}"*2
    MESSAGES.AddMessage("Parameters association")
    MESSAGES.AddMessage(row_format.format("Model","Test"))
    for m_r, t_r in zip(model_regressors, test_regressors_name):
        MESSAGES.AddMessage(row_format.format(m_r, t_r))
        
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

    if not isinstance(fields_name, list):
        field = field.flatten()
    field[field == sys.maxint] = np.NaN

    return field


def _print_test_results(classifier, regressors_test, response_test):
    global MESSAGES
    MESSAGES.AddMessage("Adaboost classifier with " + str(classifier.n_estimators) + " estimators and learning rate "
                        + str(classifier.learning_rate))

    predicted = classifier.predict(regressors_test)
    confusion = sklm.confusion_matrix(response_test, predicted)
    des_func = classifier.decision_function(regressors_test)
    probabilities = classifier.predict_proba(regressors_test)[:, classifier.classes_ == 1]

    MESSAGES.AddMessage("Confusion Matrix :")
    labels = ["Non Deposit", "Deposit"]
    row_format = "{:6}" + "{:^16}" * (len(labels) + 1)
    MESSAGES.AddMessage(row_format.format("", "", "Predicted", ""))
    MESSAGES.AddMessage(row_format.format("True", "", *labels))
    for label, row in zip(labels, confusion):
        MESSAGES.AddMessage(row_format.format("", label, *row))

    MESSAGES.AddMessage("Accuracy : {}".format(sklm.accuracy_score(response_test, predicted)))
    MESSAGES.AddMessage("Precision : {}".format(sklm.precision_score(response_test, predicted)))
    MESSAGES.AddMessage("Recall: {}".format(sklm.recall_score(response_test, predicted)))
    MESSAGES.AddMessage("Area Under the curve (AUC): {}:".format(sklm.roc_auc_score(response_test, des_func)))
    MESSAGES.AddMessage("Average Precision Score: {}".format(sklm.average_precision_score(response_test, probabilities)))

    return


def _plot_results(classifier, regressors_test, response_test, plot_file):

    global MESSAGES

    MESSAGES.AddMessage("Plotting Results")

    des_func = classifier.decision_function(regressors_test)
    probabilities = classifier.predict_proba(regressors_test)[:, classifier.classes_ == 1]

    fpr, tpr, unused = sklm.roc_curve(response_test, des_func)
    pre, rec, unused = sklm.precision_recall_curve(response_test, probabilities)

    fig, (ax_roc, ax_prc) = plt.subplots(2)

    ax_roc.plot(fpr, tpr, label="ROC Curve (AUC={0:.3f})".format(sklm.roc_auc_score(response_test, des_func)))
    ax_roc.plot([0, 1], [0, 1], linestyle='--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc='lower right')

    ax_prc.plot(pre, rec, label="Precision-Recall Curve "
                                "(Area={0:.3f})".format(sklm.average_precision_score(response_test, probabilities)))
    ax_prc.set_xlabel('Recall')
    ax_prc.set_ylabel("Precision")
    ax_prc.legend(loc='lower left')

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
    input_model = parameter_dic["input_model"].valueAsText
    test_points = parameter_dic["test_points"].valueAsText
    test_regressors_name = [x.strip("'") for x in parameter_dic["test_regressors_name"].valueAsText.split(";")]
    test_response_name = parameter_dic["test_response_name"].valueAsText
    plot_file = parameter_dic["plot_file"].valueAsText

    classifier = joblib.load(input_model)


    test_regressors = _get_fields(test_points, test_regressors_name)
    test_response = _get_fields(test_points, test_response_name)

    _print_test_results(classifier, test_regressors, test_response)

    if plot_file is not None:
        _plot_results(classifier, test_regressors, test_response, plot_file)

    return
