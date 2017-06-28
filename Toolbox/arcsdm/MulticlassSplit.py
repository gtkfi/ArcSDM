import arcpy


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


def execute(self, parameters, messages):

    global MESSAGES
    MESSAGES = messages

    print_parameters(parameters)

    _input_validation(parameters)

    parameter_dic = {par.name: par for par in parameters}
    input_feature = parameter_dic["input_feature"].valueAsText
    class_field = parameter_dic["class_field"].valueAsText
    output_prefix = parameter_dic["output_prefix"].valueAsText
    transformation = parameter_dic["transformation"].valueAsText

    scratch_files = []

    scratch_base = arcpy.CreateScratchName("temp", data_type="FeatureClass", workspace=arcpy.env.scratchWorkspace)
    try:
        try:
            arcpy.Copy_management(arcpy.Describe(input_feature).catalogPath, scratch_base)
        except arcpy.ExecuteError:
            arcpy.Delete_management(scratch_base)
            arcpy.CopyFeatures_management(arcpy.Describe(input_feature).catalogPath, scratch_base)
        _verbose_print("Scratch file created {}".format(scratch_base))
        scratch_files.append(scratch_base)

        values = []
        with arcpy.da.SearchCursor(scratch_base, class_field) as cursor:
            for row in cursor:
                values.append(row[0])
        unique_values = list(set(values))
        _verbose_print("Unique values: {}".format(str(unique_values)))

        layer_scratch = "layer_scratch"
        arcpy.MakeFeatureLayer_management(scratch_base, layer_scratch)
        arcpy.SelectLayerByAttribute_management(layer_scratch, "CLEAR_SELECTION")

        for orig_val in unique_values:
            unicode_val = unicode(orig_val)
            fullname = output_prefix + unicode_val.encode("ascii",'ignore').replace(" ","_")
            if " " in unicode_val:
                arcpy.SelectLayerByAttribute_management(layer_scratch, "NEW_SELECTION",
                                                        u"{} = '{}'".format(class_field, orig_val))
            else:
                arcpy.SelectLayerByAttribute_management(layer_scratch, "NEW_SELECTION",
                                                        u"{} = {}".format(class_field, orig_val))
            raster = arcpy.sa.EucDistance(layer_scratch)
            if transformation == "Inverse Linear Distance":
                max_val = float(arcpy.GetRasterProperties_management(raster,"MAXIMUM").getOutput(0))
                raster = 1 - (raster * 1.0) / max_val
            elif transformation == "Inverse Distance": # TODO : do it in python to increase precision
                raster = 1 / (raster + 1.0)
            elif transformation == "Binary":
                raster = raster == 0

            raster.save(fullname)
            _verbose_print("Output file created: {}".format(fullname))
    except:
        raise
    finally:
        for s_f in scratch_files:
            arcpy.Delete_management(s_f)
            _verbose_print("Scratch file deleted {}".format(s_f))

