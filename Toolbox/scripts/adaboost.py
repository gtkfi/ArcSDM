# -*- coding: utf-8 -*-
# TODO: Add documentation
# TODO: What if the user has no license? Check licences for all functions
# TODO: Break in Several Tools
# TODO: Try parallel

import arcpy
import numpy as np
import sys
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import confusion_matrix, roc_auc_score


# Developing functions
def print_parameters(parameters, messages):
    for var, par in enumerate(parameters):
        messages.AddMessage("parameters[" + str(var) + "](" + str(par.name) + "): " + str(par.valueAsText))


def print_definition(obj, messages):
    string = str(type(obj)) + "\n"
    describe = arcpy.Describe(obj)
    if hasattr(describe, "baseName"): string = string + "describe.baseName: " + describe.baseName + "\n"
    if hasattr(describe, "catalogPath"): string = string + "describe.catalogPath: " + describe.catalogPath + "\n"
    if hasattr(describe, "dataElementType"): string = string + "describe.dataElementType: " + describe.dataElementType + "\n"
    if hasattr(describe, "dataType"): string = string + "describe.dataType: " + describe.dataType + "\n"
    if hasattr(describe, "extension"): string = string + "describe.extension: " + describe.extension + "\n"
    if hasattr(describe, "file"): string = string + "describe.file: " + describe.file + "\n"
    if hasattr(describe, "name"): string = string + "describe.name: " + describe.name + "\n"
    if hasattr(describe, "path"): string = string + "describe.pat: " + describe.path + "\n"

    messages.Addmesssage(string)


# Private functions
def _create_random_points(area, buffer_val, points, messages):
    # Process: Buffer
    buffer_name = arcpy.CreateScratchName("temp", data_type="Shapefile", workspace=arcpy.env.scratchFolder)
    arcpy.Buffer_analysis(points, buffer_name, buffer_val, "FULL", "ROUND", "ALL", "", "PLANAR")

    # Process: Erase
    erase_name = arcpy.CreateScratchName("temp", data_type="Shapefile", workspace=arcpy.env.scratchFolder)
    arcpy.Erase_analysis(area, buffer_name, erase_name, "")

    # Process: Create Random Points
    random_name = arcpy.CreateScratchName("temp", data_type="FeatureClass", workspace=arcpy.env.scratchFolder)
    num_points = int(arcpy.GetCount_management(points))
    arcpy.CreateRandomPoints_management(arcpy.env.scratchFolder, random_name, erase_name, "0 0 250 250", num_points,
                                        "1 Meters", "POINT", "0")

    messages.AddMessage(str(arcpy.GetCount_management(random_name)) + " Random points created")

    # array = arcpy.da.FeatureClassToNumPyArray(random_name, "SHAPE@XY")

    # arcpy.Delete_management(buffer_name)
    # arcpy.Delete_management(erase_name)
    # arcpy.Delete_management(random_name)

    # return array
    return random_name


def _get_file_name(obj):
    return arcpy.Describe(obj.strip("'")).catalogPath


def _add_calculate_field(base, value):
    scratch = arcpy.CreateScratchName("temp", data_type="FeatureClass", workspace=arcpy.env.scratchWorkspace)
    try:
        arcpy.Copy_management(_get_file_name(base), scratch)
    except arcpy.ExecuteError:
        arcpy.Delete_management(scratch)
        arcpy.CopyFeatures_management(_get_file_name(base), scratch)

    arcpy.AddField_management(scratch, "Prospect", "LONG", "", "", "", "", "NULLABLE", "NON_REQUIRED", "")
    arcpy.CalculateField_management(scratch, "Prospect", str(value), "PYTHON", "")
    return scratch


def enrich_points(prospective, non_prospective, rasters, messages):

    if prospective is None and non_prospective is None:
        return [None, None, None]

    messages.AddMessage("Assigning Raster information...")
    scratch_files = []

    try:
        if prospective is None:
            merged_name = _add_calculate_field(non_prospective, -1)
            scratch_files.append(merged_name)
        elif non_prospective is None:
            merged_name = _add_calculate_field(prospective, 1)
            scratch_files.append(merged_name)
        else:
            prospective_scratch = _add_calculate_field(prospective, 1)
            scratch_files.append(prospective_scratch)
            non_prospective_scratch = _add_calculate_field(non_prospective, -1)
            scratch_files.append(non_prospective_scratch)
            merged_name = arcpy.CreateScratchName("temp", data_type="FeatureClass",
                                                  workspace=arcpy.env.scratchWorkspace)
            arcpy.Merge_management([prospective_scratch, non_prospective_scratch], merged_name)
            scratch_files.append(merged_name)

        regressor_names = []
        arcpy.SetProgressor("step", "Adding raster values to the points", min_range=0, max_range=len(rasters),
                            step_value=1)
        for raster in rasters:
            try:
                extracted_name = arcpy.CreateScratchName("temp", data_type="FeatureClass",
                                                         workspace=arcpy.env.scratchWorkspace)
                arcpy.gp.ExtractValuesToPoints(merged_name, _get_file_name(raster), extracted_name, "INTERPOLATE",
                                               "VALUE_ONLY")
                scratch_files.append(extracted_name)
                arcpy.AlterField_management(extracted_name, "RASTERVALU", arcpy.Describe(raster).baseName)
                regressor_names.append(arcpy.Describe(raster).baseName)
                merged_name = extracted_name
                arcpy.SetProgressorPosition()
            except:
                messages.addErrorMessage("Problem with raster {}".format(raster))
                raise

        arcpy.SetProgressorLabel("Executing Adaboost")
        arcpy.ResetProgressor()

        try:
            reg = arcpy.da.FeatureClassToNumPyArray(extracted_name, regressor_names)
        except TypeError:
            reg = arcpy.da.FeatureClassToNumPyArray(extracted_name, regressor_names, null_value=sys.maxint)
        res = arcpy.da.FeatureClassToNumPyArray(extracted_name, "Prospect")

        regressors = np.array([[elem for elem in row] for row in reg])
        regressors[regressors == sys.maxint] = np.NaN
        response = np.array([[elem for elem in row] for row in res]).flatten()

    finally:
        for s_file in scratch_files:
            arcpy.Delete_management(s_file)

    return [regressors, response, regressor_names]


def clear_missings(regressors, response, mask_value, messages):
    if regressors is None or response is None:
        return [regressors, response]

    messages.AddMessage("Clearing Missing Data...")
    if mask_value is None:
        missing_rows = [index for index, row in enumerate(regressors) if any(np.isnan(row))]
        if len(missing_rows) > 0:
            messages.addMessage(str(len(missing_rows)) + " observations deleted for missing values")
            regressors = np.delete(regressors, missing_rows, 0)
            response = np.delete(response, missing_rows, 0)
    else:
        nan_index = np.where(np.isnan(regressors))
        if len(nan_index[0]) > 0:
            messages.addMessage(str(len(nan_index[0])) + " observations replaced for missing values")
            regressors[nan_index] = mask_value
    messages.AddMessage("Number of observations " + str(regressors.shape[0]))

    return [regressors, response]


def _get_best_parameters(num_estimators, learning_rate, regressors, response,  messages):
    if (num_estimators is not None) and (learning_rate is not None):
        return [num_estimators, learning_rate]

    messages.AddMessage("Calculating best parameters...")
    if num_estimators is None:
        num_estimators_space = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100])
    else:
        num_estimators_space = [num_estimators]

    if learning_rate is None:
        learning_rate_space = np.array([0.1, 0.5, 0.75, 1, 1.25, 1.5, 2])
    else:
        learning_rate_space = [learning_rate]

    arcpy.SetProgressor("step", "Calculating best parameter combination", min_range=0,
                        max_range=len(num_estimators_space)*len(learning_rate_space), step_value=1)
    scores = np.empty([len(num_estimators_space), len(learning_rate_space), 3])
    for it_n_est, n_est in enumerate(num_estimators_space):
        for it_l_rate, l_rate in enumerate(learning_rate_space):
            scores[it_n_est, it_l_rate] = cross_val_score(AdaBoostClassifier(n_estimators=n_est, learning_rate=l_rate),
                                                          regressors, response)
            arcpy.SetProgressorPosition()
    arcpy.SetProgressorLabel("Executing Adaboost")
    arcpy.ResetProgressor()

    scores_mean = np.mean(scores, 2)
    # scores_std = np.std(scores,2)
    i, j = np.unravel_index(scores_mean.argmax(), scores_mean.shape)
    max_num_estimators = num_estimators_space[i]
    max_learning_rate = learning_rate_space[j]

    if num_estimators is None:
        messages.AddMessage("Number of estimators selected: " + str(max_num_estimators))
    if learning_rate is None:
        messages.AddMessage("Learning rate selected: " + str(max_learning_rate))

    if num_estimators is None or learning_rate is None:
        messages.AddMessage("Score: " + str(scores_mean[i, j]))
        messages.AddMessage("Other Options:")
        for unused in xrange(3):
            scores_mean[i, j] = 0
            i, j = np.unravel_index(scores_mean.argmax(), scores_mean.shape)
            messages.AddMessage("  Num Estimators: " + str(num_estimators_space[i]) + " Learning Rate: " +
                                str(learning_rate_space[j]) + " Score(K-folds): " + str(scores_mean[i, j]))

    return [max_num_estimators, max_learning_rate]


def _print_classification_results(classifier, regressors, response, regressors_test, response_test, regressor_names,
                                  messages):

    loo = LeaveOneOut()
    cv_score = cross_val_score(classifier, regressors, response, cv=loo.split(regressors))
    classifier.fit(regressors, response)
    messages.AddMessage("Adaboost classifier with " + str(classifier.n_estimators) + " estimators and learning rate "
                        + str(classifier.learning_rate))

    if regressors_test is None or response_test is None:
        regressors_test = regressors
        response_test = response
        t_set = "Train"
    else:
        t_set = "Test"

    messages.AddMessage("Score (" + t_set + " Set):" + str(classifier.score(regressors_test, response_test)))
    messages.AddMessage("Score (Leave one Out):" + str(cv_score.mean()))
    messages.AddMessage("Confusion Matrix (" + t_set + " Set):")

    confusion = confusion_matrix(response_test, classifier.predict(regressors_test))
    labels = ["Non Prospective", "Prospective"]
    row_format = "{:6}" + "{:^16}" * (len(labels) + 1)
    messages.AddMessage(row_format.format("", "", "Predicted", ""))
    messages.AddMessage(row_format.format("True", "", *labels))
    for label, row in zip(labels, confusion):
        messages.AddMessage(row_format.format("", label, *row))
    messages.AddMessage("Area Under the curve (AUC):" + str(roc_auc_score(response_test,
                                                            classifier.decision_function(regressors_test))))

    messages.AddMessage("Feature importances: ")
    importances = [[name, val] for name, val in zip(regressor_names, classifier.feature_importances_)]
    for elem in sorted(importances, key=lambda imp: imp[1], reverse=True):
        if elem[1] > 0:
            messages.AddMessage(elem[0] + ": \t" + str(elem[1]*100) + "%")
    return


def create_response_raster(classifier, rasters, output, messages):
    messages.AddMessage("Creating response raster...")
    scratch_files = []

    try:
        messages.AddMessage("Debug: Checkpoint 1")
        scratch_multi_rasters = arcpy.CreateScratchName("temp", workspace=arcpy.env.scratchWorkspace)
        scratch_files.append(scratch_multi_rasters)
        arcpy.CompositeBands_management(rasters, scratch_multi_rasters)

        raster = arcpy.Raster(scratch_multi_rasters)

        lower_left_corner = arcpy.Point(raster.extent.XMin, raster.extent.YMin)
        x_cell_size = raster.meanCellWidth
        y_cell_size = raster.meanCellHeight
        messages.AddMessage("Debug: Checkpoint 2")
        try:
            raster_array = arcpy.RasterToNumPyArray(scratch_multi_rasters, nodata_to_value=np.NaN)
            messages.AddMessage("Debug: Checkpoint 3")
        except ValueError:
            messages.AddMessage("Integer type raster, changed to float")
            raster_array = 1.0 * arcpy.RasterToNumPyArray(scratch_multi_rasters, nodata_to_value=sys.maxint)
            raster_array[raster_array == sys.maxint] = np.NaN
            messages.AddMessage("Debug: Checkpoint 4")

        n_regr = raster_array.shape[0]
        n_rows = raster_array.shape[1]
        n_cols = raster_array.shape[2]

        raster_array2 = np.empty([n_rows, n_cols, n_regr])
        for raster_index in xrange(n_regr):
            raster_array2[:, :, raster_index] = raster_array[raster_index, :, :]

        raster_array2 = np.reshape(raster_array2, [n_rows * n_cols, n_regr])

        finite_mask = np.all(np.isfinite(raster_array2), axis=1)
        nan_mask = np.logical_not(finite_mask)
        messages.AddMessage("Debug: Checkpoint 5")
        responses = classifier.predict_proba(raster_array2[finite_mask])[:, classifier.classes_ == 1]

        response_vector = np.empty(n_rows * n_cols)
        response_vector[finite_mask] = responses
        response_vector[nan_mask] = -9
        response_array = np.reshape(response_vector, [n_rows, n_cols])
        messages.AddMessage("Debug: Checkpoint 6")
        response_raster = arcpy.NumPyArrayToRaster(response_array, lower_left_corner=lower_left_corner,
                                                   x_cell_size=x_cell_size, y_cell_size=y_cell_size, value_to_nodata=-9)
        response_raster.save(output)
        messages.AddMessage("Raster file created in " + output)
        arcpy.DefineProjection_management(output, arcpy.Describe(scratch_multi_rasters).spatialReference)

    finally:
        messages.AddMessage("Debug: Checkpoint 7")
        for s_f in scratch_files:
            arcpy.Delete_management(s_f)

        return


def Execute(self, parameters, messages):

    messages.AddMessage("=" * 10 + " Adaboost " + "=" * 10)

    name_prospective_points = parameters[0].valueAsText.strip("'")
    name_non_prospective_points = parameters[1].valueAsText.strip("'")
    name_prospective_test_points = parameters[2].valueAsText
    if not(name_prospective_test_points is None):
        name_prospective_test_points = name_prospective_test_points.strip("'")
    name_non_prospective_test_points = parameters[3].valueAsText
    if not(name_non_prospective_test_points is None):
        name_non_prospective_test_points = name_non_prospective_test_points.strip("'")
    name_information_rasters = [x.strip("'") for x in parameters[4].valueAsText.split(";")]
    num_estimators = parameters[5].value
    if not(num_estimators is None):
        num_estimators = int(num_estimators)
    learning_rate = parameters[6].value
    if not(learning_rate is None):
        learning_rate = float(learning_rate)
    missing_mask = parameters[7].value
    if not(missing_mask is None):
        missing_mask = float(missing_mask)
    output_model = parameters[8].valueAsText
    output_map = parameters[9].valueAsText

    regressors, response, regressor_names = enrich_points(name_prospective_points, name_non_prospective_points,
                                                          name_information_rasters, messages)
    regressors, response = clear_missings(regressors, response, missing_mask, messages)

    regressors_test, response_test, unused = enrich_points(name_prospective_test_points,
                                                           name_non_prospective_test_points, name_information_rasters,
                                                           messages)
    regressors_test, response_test = clear_missings(regressors_test, response_test, missing_mask, messages)

    num_estimators, learning_rate = _get_best_parameters(num_estimators, learning_rate, regressors, response, messages)

    classifier = AdaBoostClassifier(n_estimators=num_estimators, learning_rate=learning_rate)
    _print_classification_results(classifier, regressors, response, regressors_test, response_test, regressor_names,
                                  messages)

    if output_model is not None:
        joblib.dump(classifier, output_model)
        messages.AddMessage("Model saved in " + output_model)

    if output_map is not None:
        create_response_raster(classifier, parameters[2].valueAsText, output_map, messages)
    return
