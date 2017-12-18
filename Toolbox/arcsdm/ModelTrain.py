"""
    Model Train tool
    
    This module contains the execution code for the tools Adaboost train, BrownBoost train Logistic Regression train, 
        SVM train 
    - Just the execute function should be called from the exterior, all other functions are called by this.
    - This module makes use of the non-standard modules:
        arcpy: for GIS operations and communication with ArcGIS. Is distributed along with ArcGIS desktop software
        sklearn: for models training, for more information visit http://scikit-learn.org/stable/index.html
    

    Authors: Irving Cabrera <irvcaza@gmail.com>
"""


import arcpy
import numpy as np
import sys
import datetime
from timeit import default_timer as timer
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from .weight_boosting import BrownBoostClassifier

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

    # If only one field is given the matrix needs to be flatten to a vector
    if not isinstance(fields_name, list):
        field = field.flatten()
    # Assign NAN to the numbers with maximum integer value
    field[field == sys.maxsize] = np.NaN

    return field


def _save_model(classifier_name, classifier, output_model, train_points, train_regressors_name):
    """
        _save_model
            Saves the model to a file as well as some information about the model in a text file
             
        :param classifier_name: Name of the classifier method 
        :param classifier: Classifier object to be saved 
        :param output_model: Path of the file to be saved
        :param train_points: Name of the feature class used to train the points 
        :param train_regressors_name: Names of the regressor fields used to train the model 
        :return: None
    """
    _verbose_print("classifier_name: {}".format(classifier_name))
    _verbose_print("classifier: {}".format(classifier))
    _verbose_print("output_model: {}".format(output_model))
    _verbose_print("train_points: {}".format(train_points))
    _verbose_print("train_regressors_name: {}".format(train_regressors_name))

    # Save the model
    joblib.dump(classifier, output_model)
    # Change extension to save the companion text file
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
    """
        _print_train_results
            Performs validation tests of the model and prints the results
             
        :param classifier_name: Name of the classifier method 
        :param classifier: Classifier object
        :param regressors: numpy array with the regressors used to train the model
        :param response: numpy array with the response used to train the model
        :param regressor_names: List with the name of the regressors
        :param leave_one_out: Boolean, true to perform leave-one-out cross-validation, otherwise perform default cross 
            validation
        :return: None 
    """
    global MESSAGES
    _verbose_print("classifier_name: {}".format(classifier_name))
    _verbose_print("classifier: {}".format(classifier))
    _verbose_print("regressor_names: {}".format(regressor_names))
    _verbose_print("leave_one_out: {}".format(leave_one_out))

    MESSAGES.AddMessage("{} classifier with parameters: \n {}".format(classifier_name,
                                                                      str(classifier.get_params()).replace("'", "")))

    if leave_one_out:
        # create a leave-one-out instance to execute the cross-validation
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
    # Print validation time
    MESSAGES.AddMessage("Testing time: {:.3f} seconds, {:.3f} seconds per test".format(end - start,
                                                                                       (end - start) / n_tests))
    # Print confusion matrix
    MESSAGES.AddMessage("Confusion Matrix (Train Set):")

    confusion = confusion_matrix(response, classifier.predict(regressors))
    labels = ["Non Deposit", "Deposit"]
    row_format = "{:6}" + "{:^16}" * (len(labels) + 1)
    MESSAGES.AddMessage(row_format.format("", "", "Predicted", ""))
    MESSAGES.AddMessage(row_format.format("True", "", *labels))
    for label, row in zip(labels, confusion):
        MESSAGES.AddMessage(row_format.format("", label, *row))

    # Some classifiers do not have  decision_function attribute but count with predict_proba instead
    # TODO: Generalize to anything that does not have decision_function "Easier to ask for forgiveness than permission"
    if classifier_name in ["Random Forest"]:
        des_fun = classifier.predict_proba(regressors)[:, classifier.classes_ == 1]
    else:
        des_fun = classifier.decision_function(regressors)
    MESSAGES.AddMessage("Area Under the curve (AUC): {}".format(roc_auc_score(response, des_fun)))

    # Give the importance of the features if it is supported
    # TODO: Generalize to anything that does have feature_importances_ "Easier to ask for forgiveness than permission"
    if classifier_name == "Adaboost":
        MESSAGES.AddMessage("Feature importances: ")
        importances = [[name, val*100] for name, val in zip(regressor_names, classifier.feature_importances_)]
        long_word = max([len(x) for x in regressor_names])
        row_format = "{" + ":" + str(long_word) + "} {:4.1f}%"
        # Print regressors in descending importance, omit the ones with 0 importance
        for elem in sorted(importances, key=lambda imp: imp[1], reverse=True):
            if elem[1] > 0:
                MESSAGES.AddMessage(row_format.format(*elem))

    return


def execute(self, parameters, messages):
    """
        Model Train tool
            Trains one of the predefined models with their respective parameters. This tool should be executed from a 
                python toolbox 
            Currently supports for Adaboost(SAMME), BrownBoost, logistic regression, random forest and support vector machine
            New models need to have implemented the methods  fit, get_params, predict and one of predict_proba or decision_function
            Additionally, it can implement feature_importances_ 
            
        :param parameters: parameters object with all the parameters from the python-tool. It necessarily contains
            train_points: (Points) Points that will be used for the training 
            train_regressors: (Field) Name of the regressors fields that will be used for the training 
            train_response: (Field) Name of the response/class field that will be used for the training 
            output_model: (File path) Name of the file where the model will be stored
            leave_one_out: (Boolean) Choose between test with leave-one-out (true) or 3-fold cross-validation (false)  
            classifier_name: (String) Name of the model to be trained 
            
        :param messages: messages object to print in the console, must implement AddMessage 
        
        :return: None
    """
    global MESSAGES
    MESSAGES = messages
    # Print parameters for debugging purposes
    print_parameters(parameters)

    # Decompose the parameters object and assign the value to variables
    parameter_dic = {par.name: par for par in parameters}
    classifier_name = parameter_dic["classifier_name"].valueAsText
    train_points = parameter_dic["train_points"].valueAsText
    train_regressors_name = parameter_dic["train_regressors"].valueAsText.split(";")
    train_response_name = parameter_dic["train_response"].valueAsText
    output_model = parameter_dic["output_model"].valueAsText
    leave_one_out = parameter_dic["leave_one_out"].value

    # Check for correctness in the parameters
    _input_validation(parameters)

    train_regressors = _get_fields(train_points, train_regressors_name)
    train_response = _get_fields(train_points, train_response_name)

    # Choice of the model type, the specific parameters are then passed to variables
    if classifier_name == "Adaboost":
        """
            Parameters:
                num_estimators: (Integer) Number of estimators to be used 
                learning_rate: (Float) Learning rate of the model           
            For more information about the model visit 
            http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
        """
        _verbose_print("Adaboost selected")
        num_estimators = parameter_dic["num_estimators"].value
        learning_rate = parameter_dic["learning_rate"].value
        classifier = AdaBoostClassifier(base_estimator=None, n_estimators=num_estimators, learning_rate=learning_rate,
                                        algorithm='SAMME.R', random_state=None)

    elif classifier_name == "Logistic Regression":
        """
            Parameters:
                deposit_weight: (Integer) weight to be given to the deposits to deal with unbalanced data 
                penalty: (string) type of norm for the penalty 
                random_state: (Integer) seed for random generator, useful to obtain reproducible results 
                
            For more information about the model visit 
            http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        """
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
        """
            Parameters:
                countdown: (Float) Initial value of the countdown timer
        """
        _verbose_print("BrownBoost selected")
        countdown = parameter_dic["countdown"].value
        classifier = BrownBoostClassifier(base_estimator=None, n_estimators=1000, learning_rate=1,
                                          algorithm='BROWNIAN', random_state=None, countdown = countdown)

    elif classifier_name == "SVM":
        """
            Parameters:
                kernel: (String) Kernel to be used  
                deposit_weight: (Integer) weight to be given to the deposits to deal with unbalanced data 
                penalty: (string) type of norm for the penalty 
                random_state:(Integer) seed for random generator, useful to obtain reproducible results 
                normalize: (Boolean) Indicates if the data needs to be normalized (True) or not (False). Notice that 
                    SVM is sensitive linear transformations  

            For more information about the model visit 
            http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        """
        penalty = parameter_dic["penalty"].value
        kernel = str(parameter_dic["kernel"].valueAsText)
        random_state = parameter_dic["random_state"].value
        deposit_weight = parameter_dic["deposit_weight"].value
        if deposit_weight is None:
            _verbose_print("deposit_weight is None, balanced wighting will be used")
            class_weight = "balanced"
        else:
            class_weight = {1: float(deposit_weight), -1: (100-float(deposit_weight))}

        classifier = SVC(C=penalty, kernel=kernel, degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True,
                         tol=0.001, cache_size=200, class_weight=class_weight, verbose=False, max_iter=-1,
                         decision_function_shape='ovr', random_state=random_state)

    elif classifier_name == "Random Forest":
        """
            Parameters:
    
                num_estimators: (Integer) Number of trees to be trained 
                max_depth: (Integer) max depth of the trained trees 
                deposit_weight: (Integer) weight to be given to the deposits to deal with unbalanced data 
                random_state:(Integer) seed for random generator, useful to obtain reproducible results 

            For more information about the model visit 
            http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        """
        _verbose_print("Random Forest selected")
        num_estimators = parameter_dic["num_estimators"].value
        max_depth = parameter_dic["max_depth"].value
        random_state = parameter_dic["random_state"].value
        deposit_weight = parameter_dic["deposit_weight"].value
        if deposit_weight is None:
            _verbose_print("deposit_weight is None, balanced wighting will be used")
            class_weight = "balanced"
        else:
            class_weight = {1: float(deposit_weight), -1: (100-float(deposit_weight))}

        classifier = RandomForestClassifier(n_estimators=num_estimators, criterion='gini', max_depth=max_depth, min_samples_split=2,
                               min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                               max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1,
                               random_state=random_state, verbose=0, warm_start=False, class_weight=class_weight)
    else:
        raise NotImplementedError("Not implemented classifier: {}".format(classifier_name))

    # Some classifiers need the data be normalized before training, this is done here
    if classifier_name in ["SVM"]:
        normalize = parameter_dic["normalize"].value
        if normalize:
            scaler = StandardScaler().fit(train_regressors)
            train_regressors = scaler.transform(train_regressors)
            MESSAGES.AddMessage("Data normalized")
            if output_model is not None:
                # Save the information of the normalize transformation
                joblib.dump(scaler, output_model.replace(".pkl", "_scale.pkl"))

    # train the model
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
