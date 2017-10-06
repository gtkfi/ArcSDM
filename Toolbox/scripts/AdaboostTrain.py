import arcpy
import numpy as np
import sys
import datetime
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import confusion_matrix, roc_auc_score

# TODO: Add documentation

MESSAGES = None
VERBOSE = False


if VERBOSE:
    def _verbose_print(text):
        global MESSAGES
        MESSAGES.AddMessage("Verbose: " + text)
else:
    _verbose_print = lambda *a: None


def print_parameters(parameters):
    for var, par in enumerate(parameters):
        _verbose_print("parameters[" + str(var) + "](" + str(par.name) + "): " + str(par.valueAsText))


def _input_validation(parameters):
    parameter_dic = {par.name: par for par in parameters}
    train_points = parameter_dic["train_points"].valueAsText
    train_regressors_name = parameter_dic["train_regressors"].valueAsText.split(";")
    train_response_name = parameter_dic["train_response"].valueAsText
    num_estimators = parameter_dic["num_estimators"].value
    learning_rate = parameter_dic["learning_rate"].value

    if train_points is not None:
        if train_regressors_name is None or train_response_name is None:
            raise ValueError("Train regressors and response must be specified")
        if num_estimators is None or learning_rate is None:
            raise ValueError("Train parameters must be specified")
    
    return


def _get_fields(feature_layer, fields_name):

    _verbose_print("feature_layer: {}".format(feature_layer))
    _verbose_print("fields_name: {}".format(fields_name))

    try:
        fi = arcpy.da.FeatureClassToNumPyArray(feature_layer, fields_name)
    except TypeError:
        _verbose_print("Failed importing with nans, possibly a integer feature class")
        fi = arcpy.da.FeatureClassToNumPyArray(feature_layer, fields_name, null_value=sys.maxint)

    field = np.array([[elem*1.0 for elem in row] for row in fi])

    if not isinstance(fields_name, list):
        field = field.flatten()
    field[field == sys.maxint] = np.NaN

    return field


def _save_model(classifier, output_model, train_points, train_regressors_name):

    joblib.dump(classifier, output_model)
    output_text = output_model.replace(".pkl", ".txt")
    with open(output_text, "w") as f:
        f.write("Adaboost classifier \n")
        f.write("Model stored in {} \n".format(output_model))
        f.write("Created: {} \n".format(datetime.datetime.now()))
        f.write("Trained with {} points \n".format(arcpy.GetCount_management(train_points).getOutput(0)))
        f.write("From file {} \n".format(train_points))
        f.write("Learning rate: {} \n".format(classifier.learning_rate))
        f.write("Number of estimators {} \n".format(classifier.n_estimators))
        f.write("Number of regressors {} \n".format(len(train_regressors_name)))
        for regressor in train_regressors_name:
            f.write("Regressor: '{}' :\n".format(regressor))

    MESSAGES.AddMessage("Model saved in " + output_model)

def _print_train_results(classifier, regressors, response, regressor_names, leave_one_out):
    global MESSAGES
    MESSAGES.AddMessage("Adaboost classifier with " + str(classifier.n_estimators) + " estimators and learning rate "
                        + str(classifier.learning_rate))

    if leave_one_out:
        loo = LeaveOneOut()
        cv_score = cross_val_score(classifier, regressors, response, cv=loo.split(regressors))
        MESSAGES.AddMessage("Score (Leave one Out):" + str(cv_score.mean()))
    else:
        cv_score = cross_val_score(classifier, regressors, response)
        MESSAGES.AddMessage("Score (3-Fold):" + str(cv_score.mean()))

    MESSAGES.AddMessage("Confusion Matrix (Train Set):")

    confusion = confusion_matrix(response, classifier.predict(regressors))
    labels = ["Non Deposit", "Deposit"]
    row_format = "{:6}" + "{:^16}" * (len(labels) + 1)
    MESSAGES.AddMessage(row_format.format("", "", "Predicted", ""))
    MESSAGES.AddMessage(row_format.format("True", "", *labels))
    for label, row in zip(labels, confusion):
        MESSAGES.AddMessage(row_format.format("", label, *row))
    MESSAGES.AddMessage("Area Under the curve (AUC):" + str(roc_auc_score(response,
                                                            classifier.decision_function(regressors))))

    MESSAGES.AddMessage("Feature importances: ")
    importances = [[name, val] for name, val in zip(regressor_names, classifier.feature_importances_)]
    for elem in sorted(importances, key=lambda imp: imp[1], reverse=True):
        if elem[1] > 0:
            MESSAGES.AddMessage(elem[0] + ": \t" + str(elem[1]*100) + "%")
    return


def execute(self, parameters, messages):
    global MESSAGES
    MESSAGES = messages

    print_parameters(parameters)

    parameter_dic = {par.name: par for par in parameters}
    train_points = parameter_dic["train_points"].valueAsText
    train_regressors_name = parameter_dic["train_regressors"].valueAsText.split(";")
    train_response_name = parameter_dic["train_response"].valueAsText
    num_estimators = parameter_dic["num_estimators"].value
    learning_rate = parameter_dic["learning_rate"].value
    output_model = parameter_dic["output_model"].valueAsText
    leave_one_out = parameter_dic["leave_one_out"].value

    _input_validation(parameters)

    train_regressors = _get_fields(train_points, train_regressors_name)
    train_response = _get_fields(train_points, train_response_name)

    classifier = AdaBoostClassifier(n_estimators=num_estimators, learning_rate=learning_rate)
    classifier.fit(train_regressors, train_response)

    if output_model is not None:
        _save_model(classifier, output_model, train_points, train_regressors_name)

    _print_train_results(classifier, train_regressors, train_response, train_regressors_name, leave_one_out)

    return
