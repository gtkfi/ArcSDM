from scipy import ndimage, signal
import arcpy
import numpy as np
import sys

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

    return


def execute(self, parameters, messages):

    global MESSAGES
    MESSAGES = messages

    print_parameters(parameters)

    _input_validation(parameters)

    parameter_dic = {par.name: par for par in parameters}
    input_raster_name = parameter_dic["input_raster"].valueAsText
    filter_type = parameter_dic["filter_type"].valueAsText
    output_raster_name = parameter_dic["output_raster"].valueAsText
    filter_size = parameter_dic["filter_size"].value

    raster = arcpy.Raster(input_raster_name)

    lower_left_corner = arcpy.Point(raster.extent.XMin, raster.extent.YMin)
    x_cell_size = raster.meanCellWidth
    y_cell_size = raster.meanCellHeight

    try:
        raster_array = arcpy.RasterToNumPyArray(input_raster_name, nodata_to_value=np.NaN)
    except ValueError:
        _verbose_print("Integer type raster, changed to float")
        raster_array = 1.0 * arcpy.RasterToNumPyArray(input_raster_name, nodata_to_value=sys.maxint)
        raster_array[raster_array == sys.maxint] = np.NaN

    if filter_type == "Gaussian":
        out_array = ndimage.filters.gaussian_filter(raster_array, filter_size/3.0, truncate=3.0)
    elif filter_type == "Mean":
        kernel = np.ones([int(filter_size/2), int(filter_size/2)])
        kernel = kernel / np.sum(kernel)
        out_array = signal.convolve2d(raster_array, kernel, mode='same', boundary='symm')
    elif filter_type == "Median":
        if filter_size % 2 == 0:
            filter_size += 1
        out_array = signal.medfilt(raster_array, kernel_size=filter_size)
    else:
        raise ValueError("Filter not recognized ({})".format(filter_type))

    mask = np.logical_or(np.isnan(out_array),  np.isnan(raster_array))
    out_array[mask] = raster_array[mask]

    out_raster = arcpy.NumPyArrayToRaster(out_array, lower_left_corner=lower_left_corner, x_cell_size=x_cell_size,
                                          y_cell_size=y_cell_size, value_to_nodata=-9)
    out_raster.save(output_raster_name)
    MESSAGES.AddMessage("Raster file created in " + output_raster_name)
    arcpy.DefineProjection_management(output_raster_name, arcpy.Describe(input_raster_name).spatialReference)

    return
