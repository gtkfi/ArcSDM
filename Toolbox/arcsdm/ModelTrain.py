import arcpy
import numpy as np
import sys
import datetime
from timeit import default_timer as timer
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from weight_boosting import BrownBoostClassifier

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


def _save_model(classifier_name, classifier, output_model, train_points, train_regressors_name):
    _verbose_print("classifier_name: {}".format(classifier_name))
    _verbose_print("classifier: {}".format(classifier))
    _verbose_print("output_model: {}".format(output_model))
    _verbose_print("train_points: {}".format(train_points))
    _verbose_print("train_regressors_name: {}".format(train_regressors_name))

    joblib.dump(classifier, output_model)
    output_text = output_model.replace(".pkl", ".txt")
    with open(output_text, "w") as f:
        f.write("{} classifier \n".format(classifier_name))
        f.write("Parameters: {} \n".format(str(classifier.get_params())))
        f.write("Model stored in {} \n".format(output_model))
        f.write("Created: {} \n".format(datetime.datetime.now()))
        f.write("Trained with {} points \n".format(arcpy.GetCount_management(train_points).getOutput(0)))
        f.write("From file {} \n".format(train_points))
        f.write("Number of regressors {} \n".format(len(train_regressors_name)))
        for regressor in train_regressors_name:
            f.write("Regressor: '{}' \n".format(regressor))

    MESSAGES.AddMessage("Model saved in " + output_model)


def _print_train_results(classifier_name, classifier, regressors, response, regressor_names, leave_one_out):
    global MESSAGES
    _verbose_print("classifier_name: {}".format(classifier_name))
    _verbose_print("classifier: {}".format(classifier))
    _verbose_print("regressor_names: {}".format(regressor_names))
    _verbose_print("leave_one_out: {}".format(leave_one_out))

    MESSAGES.AddMessage("{} classifier with parameters: \n {}".format(classifier_name,
                                                                      str(classifier.get_params()).replace("'", "")))

    if leave_one_out:
        loo = LeaveOneOut()
        start = timer()
        cv_score = cross_val_score(classifier, regressors, response, cv=loo.split(regressors))
        end = timer()
        n_tests = len(response)
        MESSAGES.AddMessage("Score (Leave one Out):" + str(cv_score.mean()))
    else:
        start = timer()
        cv_score = cross_val_score(classifier, regressors, response)
        end = timer()
        n_tests = 3
        MESSAGES.AddMessage("Score (3-Fold):" + str(cv_score.mean()))

    MESSAGES.AddMessage("Testing time: {:.3f} seconds, {:.3f} seconds per test".format(end - start,
                                                                                       (end - start) / n_tests))
    MESSAGES.AddMessage("Confusion Matrix (Train Set):")

    confusion = confusion_matrix(response, classifier.predict(regressors))
    labels = ["Non Deposit", "Deposit"]
    row_format = "{:6}" + "{:^16}" * (len(labels) + 1)
    MESSAGES.AddMessage(row_format.format("", "", "Predicted", ""))
    MESSAGES.AddMessage(row_format.format("True", "", *labels))
    for label, row in zip(labels, confusion):
        MESSAGES.AddMessage(row_format.format("", label, *row))

    MESSAGES.AddMessage("Area Under the curve (AUC): {}".format(roc_auc_score(response,
                            classifier.predict_proba(regressors)[:, classifier.classes_ == 1].flatten())))

    if classifier_name == "Adaboost":
        MESSAGES.AddMessage("Feature importances: ")
        importances = [[name, val*100] for name, val in zip(regressor_names, classifier.feature_importances_)]
        long_word = max([len(x) for x in regressor_names])
        row_format = "{" + ":" + str(long_word) + "} {:4.1f}%"

        for elem in sorted(importances, key=lambda imp: imp[1], reverse=True):
            if elem[1] > 0:
                MESSAGES.AddMessage(row_format.format(*elem))

    return


def execute(self, parameters, messages):
    global MESSAGES
    MESSAGES = messages

    print_parameters(parameters)

    parameter_dic = {par.name: par for par in parameters}

    classifier_name = parameter_dic["classifier_name"].valueAsText
    train_points = parameter_dic["train_points"].valueAsText
    train_regressors_name = parameter_dic["train_regressors"].valueAsText.split(";")
    train_response_name = parameter_dic["train_response"].valueAsText
    output_model = parameter_dic["output_model"].valueAsText
    leave_one_out = parameter_dic["leave_one_out"].value

    _input_validation(parameters)

    train_regressors = _get_fields(train_points, train_regressors_name)
    train_response = _get_fields(train_points, train_response_name)

    if classifier_name == "Adaboost":
        _verbose_print("Adaboost selected")
        num_estimators = parameter_dic["num_estimators"].value
        learning_rate = parameter_dic["learning_rate"].value
        classifier = AdaBoostClassifier(base_estimator=None, n_estimators=num_estimators, learning_rate=learning_rate,
                                        algorithm='SAMME.R', random_state=None)

    elif classifier_name == "Logistic Regression":
        _verbose_print("Logistic Regression selected")
        penalty = parameter_dic["penalty"].valueAsText
        deposit_weight = parameter_dic["deposit_weight"].value
        random_state = parameter_dic["random_state"].value
        if deposit_weight is None:
            _verbose_print("deposit_weight is None, balanced wighting will be used")
            class_weight = "balanced"
        else:
            class_weight = {1: float(deposit_weight), -1: (100-float(deposit_weight))}

        classifier = LogisticRegression(penalty=penalty, dual=False, tol=0.0001, C=1, fit_intercept=True,
                                        intercept_scaling=1, class_weight=class_weight, random_state=random_state,
                                        solver='liblinear', max_iter=100, multi_class='ovr', verbose=0,
                                        warm_start=False, n_jobs=1)

    elif classifier_name == "Brownboost":
        _verbose_print("BrownBoost selected")
        countdown = parameter_dic["countdown"].value
        classifier = BrownBoostClassifier(base_estimator=None, n_estimators=1000, learning_rate=1,
                                          algorithm='BROWNIAN', random_state=None, countdown = countdown)

    else:
        raise NotImplementedError("Not implemented classifier: {}".format(classifier_name))

    start = timer()
    classifier.fit(train_regressors, train_response)
    end = timer()
    MESSAGES.AddMessage("Training time: {:.3f} seconds".format(end-start))

    if output_model is not None:
        _save_model(classifier_name, classifier, output_model, train_points, train_regressors_name)
    else:
        _verbose_print("No output model selected")

    _print_train_results(classifier_name, classifier, train_regressors, train_response, train_regressors_name,
                         leave_one_out)

    return
