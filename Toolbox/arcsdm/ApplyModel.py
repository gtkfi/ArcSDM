import arcpy
import numpy as np
import sys
from sklearn.externals import joblib

# TODO: Add documentation

VERBOSE = False
MESSAGES = None

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
    input_model = parameter_dic["input_model"].valueAsText
    test_regressors_name = [x.strip("'") for x in parameter_dic["info_rasters"].valueAsText.split(";")]
    global MESSAGES

    input_text = input_model.replace(".pkl", ".txt")
    model_regressors = []
    with open(input_text, "r") as f:
        for line in f:
            if line.startswith("Regressor:"):
                model_regressors.append(line.split("'")[1])

    if len(model_regressors) != len(test_regressors_name):
        raise ValueError("The amount of {} does not coincide with the model ({} vs {})".format(
            parameter_dic["info_rasters"].displayName, len(test_regressors_name), len(model_regressors)))

    row_format = "{:^16}" * 2
    MESSAGES.AddMessage("Parameters association")
    MESSAGES.AddMessage(row_format.format("Model", "Rasters"))
    for m_r, t_r in zip(model_regressors, test_regressors_name):
        MESSAGES.AddMessage(row_format.format(m_r, t_r))

    return


def _resample_rasters(rasters):
    MESSAGES.AddMessage("Resampling rasters...")

    raster_list = rasters.split(";")

    cell_sizes = []
    for index, raster_name in enumerate(raster_list):
        raster = arcpy.Raster(raster_name)
        cell_sizes.append(raster.meanCellWidth)
        cell_sizes.append(raster.meanCellHeight)

    size = min(cell_sizes)
    _verbose_print("Minimum cell size {}".format(size))
    out_list = []
    try:
        for raster_name in raster_list:
            scratch_resample = arcpy.CreateScratchName("temp", workspace=arcpy.env.scratchWorkspace)
            arcpy.Resample_management(raster_name, scratch_resample, size, "BILINEAR")
            _verbose_print("Scratch file created (resample) {}".format(scratch_resample))
            out_list.append(scratch_resample)
    except:
        for raster in out_list:
            arcpy.Delete_management(raster)
            _verbose_print("Scratch file deleted {}".format(raster))
        raise
    return ";".join(out_list)


def create_response_raster(classifier, rasters, output, scale):
    scratch_files = []

    try:
        # rasters_resampled = _resample_rasters(rasters)
        # scratch_files.extend(rasters_resampled.split(";"))
        rasters_resampled = rasters
        _verbose_print("Rasters: {}".format(rasters_resampled))
        scratch_multi_rasters = arcpy.CreateScratchName("temp", workspace=arcpy.env.scratchWorkspace)
        arcpy.CompositeBands_management(rasters_resampled, scratch_multi_rasters)
        _verbose_print("Scratch file created (CompositeBands) {}".format(scratch_multi_rasters))
        scratch_files.append(scratch_multi_rasters)
        spatial_reference = arcpy.Describe(scratch_multi_rasters).spatialReference

        raster = arcpy.Raster(scratch_multi_rasters)

        lower_left_corner = arcpy.Point(raster.extent.XMin, raster.extent.YMin)
        x_cell_size = raster.meanCellWidth
        y_cell_size = raster.meanCellHeight
        try:
            raster_array = arcpy.RasterToNumPyArray(scratch_multi_rasters, nodata_to_value=np.NaN)
        except ValueError:
            _verbose_print("Integer type raster, changed to float")
            raster_array = 1.0 * arcpy.RasterToNumPyArray(scratch_multi_rasters, nodata_to_value=sys.maxint)
            raster_array[raster_array == sys.maxint] = np.NaN
    except:
        raise
    finally:
        for s_f in scratch_files:
            arcpy.Delete_management(s_f)
            _verbose_print("Scratch file deleted {}".format(s_f))
    MESSAGES.AddMessage("Creating response raster...")

    n_regr = raster_array.shape[0]
    n_rows = raster_array.shape[1]
    n_cols = raster_array.shape[2]

    raster_array2 = np.empty([n_rows, n_cols, n_regr])
    for raster_index in xrange(n_regr):
        raster_array2[:, :, raster_index] = raster_array[raster_index, :, :]

    raster_array2 = np.reshape(raster_array2, [n_rows * n_cols, n_regr])

    finite_mask = np.all(np.isfinite(raster_array2), axis=1)
    nan_mask = np.logical_not(finite_mask)
    _verbose_print("{} elements will be calculated and {} let as NaN".format(sum(finite_mask), sum(nan_mask)))
    if scale is None:
        finite_array = raster_array2[finite_mask]
        _verbose_print("Data not normalized")
    else:
        finite_array = scale.transform(raster_array2[finite_mask])
        MESSAGES.AddMessage("Data normalized")
    responses = classifier.predict_proba(finite_array)[:, classifier.classes_ == 1]

    response_vector = np.empty(n_rows * n_cols)
    response_vector[finite_mask] = responses
    response_vector[nan_mask] = -9
    response_array = np.reshape(response_vector, [n_rows, n_cols])
    # Known bug: There is a displacement between the input layers and the response
    response_raster = arcpy.NumPyArrayToRaster(response_array, lower_left_corner=lower_left_corner,
                                               x_cell_size=x_cell_size, y_cell_size=y_cell_size, value_to_nodata=-9)
    response_raster.save(output)
    MESSAGES.AddMessage("Raster file created in " + output)
    arcpy.DefineProjection_management(output, spatial_reference)

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


def execute(self, parameters, messages):

    global MESSAGES
    MESSAGES = messages

    print_parameters(parameters)

    _input_validation(parameters)

    parameter_dic = {par.name: par for par in parameters}
    input_model = parameter_dic["input_model"].valueAsText
    info_rasters = parameter_dic["info_rasters"].valueAsText
    output_map = parameter_dic["output_map"].valueAsText

    classifier = joblib.load(input_model)
    try:
        scale = joblib.load(input_model.replace(".pkl", "_scale.pkl"))
    except:
        scale = None

    create_response_raster(classifier, info_rasters, output_map, scale)

    return