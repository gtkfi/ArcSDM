import arcpy
import numpy as np
import sys
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# TODO: Add documentation

VERBOSE = False
if VERBOSE:
    def _verbose_print(text, messages):
        messages.AddMessage("Verbose: " + text)
else:
    _verbose_print = lambda *a: None


def print_parameters(parameters, messages):
    for var, par in enumerate(parameters):
        _verbose_print("parameters[" + str(var) + "](" + str(par.name) + "): " + str(par.valueAsText), messages)


def input_validation(parameters, messages):
    # TODO: Implement checks
    return


def _get_fields(feature_layer, fields_name, messages):

    _verbose_print("feature_layer: {}".format(feature_layer), messages)
    _verbose_print("fields_name: {}".format(fields_name), messages)

    try:
        fi = arcpy.da.FeatureClassToNumPyArray(feature_layer, fields_name)
    except TypeError:
        _verbose_print("Failed importing with nans, possibly a integer feature class", messages)
        fi = arcpy.da.FeatureClassToNumPyArray(feature_layer, fields_name, null_value=sys.maxint)

    field = np.array([[elem*1.0 for elem in row] for row in fi])

    if not isinstance(fields_name, list):
        field = field.flatten()
    field[field == sys.maxint] = np.NaN

    return field


def _write_table(data, row_names, col_names, types, filename, messages):
    messages.AddMessage("Writing table...")

    _verbose_print("row_names: {}".format(row_names), messages)
    _verbose_print("col_names: {}".format(col_names), messages)
    _verbose_print("types: {}".format(types), messages)
    _verbose_print("filename: {}".format(filename), messages)

    table_path, table_name = os.path.split(filename)
    arcpy.CreateTable_management(table_path, table_name)
    _verbose_print("Table created {}".format(filename), messages)
    for name, field_type in zip(col_names, types):
        arcpy.AddField_management(filename, name, field_type)
    arcpy.DeleteField_management(filename, "Field1")

    cursor = arcpy.InsertCursor(filename)
    for data_row, name_row in zip(data, row_names):
        row = cursor.newRow()
        row.setValue(col_names[0], name_row)
        for element, name in zip(data_row, col_names[1:]):
            # _verbose_print("Element {}".format(element), messages)
            # _verbose_print("name {}".format(name), messages)
            row.setValue(name, element)
        cursor.insertRow(row)
    del cursor, row


def execute(self, parameters, messages):

    print_parameters(parameters, messages)

    parameter_dic = {par.name: par for par in parameters}
    train_points = parameter_dic["train_points"].valueAsText
    train_regressors = parameter_dic["train_regressors"].valueAsText.split(";")
    train_response = parameter_dic["train_response"].valueAsText
    num_estimators_min = parameter_dic["num_estimators_min"].value
    num_estimators_max = parameter_dic["num_estimators_max"].value
    num_estimators_increment = parameter_dic["num_estimators_increment"].value
    learning_rate_min = parameter_dic["learning_rate_min"].value
    learning_rate_max = parameter_dic["learning_rate_max"].value
    learning_rate_increment = parameter_dic["learning_rate_increment"].value
    plot_file = parameter_dic["plot_file"].valueAsText
    output_table = parameter_dic["output_table"].valueAsText

    num_estimators_space = np.arange(num_estimators_min, num_estimators_max, num_estimators_increment)
    learning_rate_space = np.arange(learning_rate_min, learning_rate_max, learning_rate_increment)

    regressors = _get_fields(train_points, train_regressors, messages)
    response = _get_fields(train_points, train_response, messages)

    messages.AddMessage("Calculating best parameters...")

    arcpy.SetProgressor("step", "Calculating best parameter combination", min_range=0,
                        max_range=len(num_estimators_space)*len(learning_rate_space), step_value=1)
    scores = np.empty([len(num_estimators_space), len(learning_rate_space), 3])
    for it_n_est, n_est in enumerate(num_estimators_space):
        for it_l_rate, l_rate in enumerate(learning_rate_space):
            scores[it_n_est, it_l_rate] = cross_val_score(AdaBoostClassifier(n_estimators=n_est, learning_rate=l_rate),
                                                          regressors, response)
            arcpy.SetProgressorPosition()
    arcpy.SetProgressorLabel("Executing Adaboost Best Parameters")
    arcpy.ResetProgressor()

    scores_mean = np.mean(scores, 2)

    # if plot_file is not None or output_table is not None:
    #     if output_table is None:
    #         output_table = arcpy.CreateScratchName("temp", ".dbf", workspace=arcpy.env.scratchFolder)
    #         scratch_erase = True
    #     else:
    #         scratch_erase = False
    if output_table is not None:
        if not output_table.endswith(".dbf"):
            output_table += ".dbf"

        col_names = ["N_Estim"] + ["LR_{}".format(x) for x in learning_rate_space]
        col_names = [x.replace(".", "_") for x in col_names]
        types = ["DOUBLE"] * (len(learning_rate_space) + 1)

        _write_table(scores_mean, num_estimators_space, col_names, types, output_table, messages)

        # if scratch_erase:
        #     scratch_files.append(output_table)

    if plot_file is not None:

        fig, (ax_line, ax_box) = plt.subplots(2)
        colors = cm.rainbow(np.linspace(0, 1, len(learning_rate_space)))

        for iterator, learning_rate in enumerate(learning_rate_space):
            ax_line.plot(num_estimators_space, scores_mean[:, iterator], label=str(learning_rate),
                         color=colors[iterator])

        ax_line.set_xlabel('Number of estimators')
        ax_line.set_ylabel("Score")
        ax_line.grid(True)
        # Shrink current axis by 20%
        box = ax_line.get_position()
        ax_line.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax_line.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Learning rate")

        fig.suptitle("Score of cross validation for Adaboost")

        ax_box.boxplot(scores_mean, labels=[str(x) for x in learning_rate_space])
        ax_box.set_xlabel('Learning Rate')
        ax_box.set_ylabel("Score")

        if not plot_file.endswith(".png"):
            plot_file += ".png"

        plt.savefig(plot_file)
        _verbose_print("Figure saved {}".format(plot_file), messages)
        # plt.show()

    # scores_std = np.std(scores,2)

    i, j = np.unravel_index(scores_mean.argmax(), scores_mean.shape)
    max_num_estimators = num_estimators_space[i]
    max_learning_rate = learning_rate_space[j]

    messages.AddMessage("Number of estimators for maximum score: " + str(max_num_estimators))
    messages.AddMessage("Learning rate for maximum score: " + str(max_learning_rate))
    messages.AddMessage("Score: " + str(scores_mean[i, j]))

    _verbose_print("Other Options:", messages)
    n_others = min(int(len(num_estimators_space)*len(learning_rate_space)*0.1), 10)
    for unused in xrange(n_others):
        scores_mean[i, j] = 0
        i, j = np.unravel_index(scores_mean.argmax(), scores_mean.shape)
        _verbose_print("Num Estimators: {} Learning Rate: {} Score(K-folds): {}".format(num_estimators_space[i],
                        learning_rate_space[j], scores_mean[i, j]), messages)

    return [max_num_estimators, max_learning_rate]
