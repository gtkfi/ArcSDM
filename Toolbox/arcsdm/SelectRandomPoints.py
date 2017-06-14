# TODO: Add documentation
# TODO: Licence Check
# Todo: alert about scratch space

import arcpy

VERBOSE = False
if VERBOSE:
    def _verbose_print(text, messages):
        messages.AddMessage("Verbose: " + text)
else:
    _verbose_print = lambda *a: None


def print_parameters(parameters, messages):
    for var, par in enumerate(parameters):
        _verbose_print("parameters[" + str(var) + "](" + str(par.name) + "): " + str(par.valueAsText), messages)


def _constrain_from_points(constrain_area, excluding_points, excluding_distance, messages):

    messages.AddMessage("Constraining Area from points...")
    _verbose_print("Constrain Area: {}".format(constrain_area), messages)
    _verbose_print("Excluding points : {}".format(excluding_points), messages)
    _verbose_print("Excluding distance : {}".format(excluding_distance), messages)

    scratch_files = []
    try:
        buffer_scratch = arcpy.CreateScratchName("temp", workspace=arcpy.env.scratchWorkspace)
        arcpy.Buffer_analysis(excluding_points, buffer_scratch, excluding_distance, dissolve_option="ALL")
        scratch_files.append(buffer_scratch)
        _verbose_print("Scratch file created (buffer): {}".format(buffer_scratch), messages)

        deleted_scratch = arcpy.CreateScratchName("temp", workspace=arcpy.env.scratchWorkspace)
        arcpy.Erase_analysis(constrain_area, buffer_scratch, deleted_scratch)
        _verbose_print("Scratch file created (erase): {}".format(deleted_scratch), messages)
    except:
        _verbose_print("Error constraining from points", messages)
        raise
    finally:
        for s_file in scratch_files:
            _verbose_print("Scratch file deleted: {}".format(s_file), messages)
            arcpy.Delete_management(s_file)
    _verbose_print("Constrain from points finished", messages)
    return deleted_scratch


def _constrain_from_raster(constrain_area, rasters, messages):

    messages.AddMessage("Constraining Area from rasters...")
    _verbose_print("Constrain Area: {}".format(constrain_area), messages)
    _verbose_print("rasters: {}".format(rasters), messages)

    scratch_files = []
    rasters = [x.strip("'") for x in rasters.split(";")]

    arcpy.SetProgressor("step", "Restricting area from missings", min_range=0, max_range=len(rasters),
                        step_value=1)

    try:
        final_raster = arcpy.sa.IsNull(arcpy.sa.Raster(rasters[0]))
        arcpy.SetProgressorPosition()
        if len(rasters) > 1:
            for raster in rasters[1:]:
                final_raster = arcpy.sa.BooleanOr(final_raster, arcpy.sa.IsNull(arcpy.sa.Raster(raster)))
                arcpy.SetProgressorPosition()
                _verbose_print("Area reduced with nulls from {}".format(raster), messages)
        final_raster = arcpy.sa.SetNull(final_raster, final_raster)
        arcpy.SetProgressorLabel("Executing Select Random Points")
        arcpy.ResetProgressor()

        domain_scratch = arcpy.CreateScratchName("temp", workspace=arcpy.env.scratchWorkspace)
        arcpy.RasterToPolygon_conversion(final_raster, domain_scratch, "SIMPLIFY")
        scratch_files.append(domain_scratch)
        _verbose_print("Scratch file created (domain): {}".format(domain_scratch), messages)

        intersect_scratch = arcpy.CreateScratchName("temp", workspace=arcpy.env.scratchWorkspace)
        arcpy.Intersect_analysis([domain_scratch, constrain_area], intersect_scratch)
        _verbose_print("Scratch file created (intersect): {}".format(domain_scratch), messages)

    except:
        _verbose_print("Error constraining from rasters", messages)
        raise

    finally:
        for s_file in scratch_files:
            arcpy.Delete_management(s_file)
            _verbose_print("Scratch file deleted: {}".format(s_file), messages)

    _verbose_print("Constrain from rasters finished", messages)
    return intersect_scratch


def execute(self, parameters, messages):

    print_parameters(parameters, messages)
    parameter_dic = {par.name: par for par in parameters}

    out_ws = parameter_dic["output_workspace"].valueAsText
    if out_ws is None:
        messages.AddMessage("Using Scratch Space")
        out_ws = arcpy.env.scratchWorkspace
    output = parameter_dic["output_point"].valueAsText.strip("'")
    n_points = parameter_dic["number_points"].value
    constrain_area = parameter_dic["constraining_area"].valueAsText.strip("'")
    rasters = parameter_dic["data_rasters"].valueAsText
    excluding_points = parameter_dic["excluding_points"].valueAsText
    excluding_distance = parameter_dic["excluding_distance"].valueAsText
    min_distance = parameter_dic["minimum_distance"].valueAsText

    scratch_files = []
    try:
        constrain_scratch = arcpy.CreateScratchName("temp", workspace=arcpy.env.scratchWorkspace)
        arcpy.CopyFeatures_management(arcpy.Describe(constrain_area).catalogPath, constrain_scratch)
        scratch_files.append(constrain_scratch)
        _verbose_print("Scratch file created (constrain): {}".format(constrain_scratch), messages)
        if excluding_points is None or excluding_distance is None:
            _verbose_print("Exclude from points omitted", messages)
            points_scratch = constrain_scratch
        else:
            points_scratch = _constrain_from_points(constrain_scratch, excluding_points, excluding_distance, messages)
            scratch_files.append(points_scratch)
        if rasters is None:
            _verbose_print("Exclude from rasters omitted", messages)
            rasters_scratch = points_scratch
        else:
            rasters_scratch = _constrain_from_raster(points_scratch, rasters, messages)
            scratch_files.append(rasters_scratch)
        result = arcpy.CreateRandomPoints_management(out_ws, output, rasters_scratch,
                                                     number_of_points_or_field=n_points,
                                                     minimum_allowed_distance=min_distance)
        arcpy.DefineProjection_management(result, arcpy.Describe(constrain_area).spatialReference)
        messages.AddMessage("Random points saved in {}".format(result))
    except:
        raise
    finally:
        for s_file in scratch_files:
            arcpy.Delete_management(s_file)
            _verbose_print("Scratch file deleted: {}".format(s_file), messages)

    return
