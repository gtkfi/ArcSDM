import arcpy
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


def _add_calculate_field(base, field_name, copy_data, value):
    scratch = arcpy.CreateScratchName("temp", data_type="FeatureClass", workspace=arcpy.env.scratchWorkspace)
    try:
        arcpy.Copy_management(arcpy.Describe(base).catalogPath, scratch)
    except arcpy.ExecuteError:
        arcpy.Delete_management(scratch)
        arcpy.CopyFeatures_management(arcpy.Describe(base).catalogPath, scratch)

    if not copy_data:
        drop_field = [f.name for f in arcpy.ListFields(scratch) if not f.required]
        _verbose_print("Fields to delete: " + str(drop_field))
        arcpy.DeleteField_management (scratch, drop_field)

    arcpy.AddField_management(scratch, field_name, "LONG", "", "", "", "", "NULLABLE", "NON_REQUIRED", "")
    arcpy.CalculateField_management(scratch, field_name, str(value), "PYTHON", "")
    return scratch


def _merge_fields(deposits_name, non_deposits_name, field_name, copy_data):
    global MESSAGES
    _verbose_print("Merge fields...")
    _verbose_print("Deposits: {}".format(deposits_name))
    _verbose_print("Non deposits: {}".format(non_deposits_name))
    scratch_files = []

    try:
        if deposits_name is None:
            merged_name = _add_calculate_field(non_deposits_name, field_name, copy_data, -1)
            _verbose_print("Scratch file created (add): {}".format(merged_name))
            _verbose_print("Number of non deposits {}".format(arcpy.GetCount_management(merged_name).getOutput(0)))
        elif non_deposits_name is None:
            merged_name = _add_calculate_field(deposits_name, field_name, copy_data, 1)
            _verbose_print("Scratch file created (add): {}".format(merged_name))
            _verbose_print("Number of deposits {}".format(arcpy.GetCount_management(merged_name).getOutput(0)))
        else:
            MESSAGES.AddMessage("Merging Data Points...")

            deposit_scratch = _add_calculate_field(deposits_name, field_name, copy_data, 1)
            scratch_files.append(deposit_scratch)
            _verbose_print("Scratch file created (add): {}".format(deposit_scratch))
            _verbose_print("Number of non deposits {}".format(arcpy.GetCount_management(deposit_scratch).getOutput(0)))
            non_deposit_scratch = _add_calculate_field(non_deposits_name, field_name, copy_data, -1)
            scratch_files.append(non_deposit_scratch)
            _verbose_print("Scratch file created (add): {}".format(non_deposit_scratch))
            _verbose_print("Number of non deposits {}".format(arcpy.GetCount_management
                                                              (non_deposit_scratch).getOutput(0)))
            merged_name = arcpy.CreateScratchName("temp", data_type="FeatureClass",
                                                  workspace=arcpy.env.scratchWorkspace)
            arcpy.Merge_management([deposit_scratch, non_deposit_scratch], merged_name)
            _verbose_print("Scratch file created (merge): {}".format(merged_name))
            _verbose_print("Total Number of points {}".format(arcpy.GetCount_management(merged_name).getOutput(0)))
    except:
        raise
    finally:
        for s_file in scratch_files:
            arcpy.Delete_management(s_file)
            _verbose_print("Scratch file deleted: {}".format(s_file))
    return merged_name


def _extract_fields(base, rasters):
    global MESSAGES
    MESSAGES.AddMessage("Assigning Raster information...")
    _verbose_print("Base: {}".format(base))
    _verbose_print("Rasters: {}".format(rasters))

    old_fields = arcpy.ListFields(base)

    arcpy.sa.ExtractMultiValuesToPoints (base, rasters)

    regressor_names = [n.name for n in arcpy.ListFields(base) if n not in old_fields]

    return base, regressor_names


def _clean_data(base, regressor_names, missing_value):
    global MESSAGES
    MESSAGES.AddMessage("Clearing Missing Data...")
    _verbose_print("Base: {}".format(base))
    _verbose_print("Regressor names: {}".format(regressor_names))
    _verbose_print("Missing value: {}".format(missing_value))

    layer_scratch = "layer_scratch"
    arcpy.MakeFeatureLayer_management(base, layer_scratch)
    arcpy.SelectLayerByAttribute_management(layer_scratch, "CLEAR_SELECTION")

    if missing_value is None:
        text = "Number of deleted elements {}"

        def deal_missings():
            arcpy.DeleteFeatures_management(layer_scratch)
    else:
        text = "number of modified fields {}"

        def deal_missings():
            arcpy.CalculateField_management(layer_scratch, regressor, str(missing_value), "PYTHON")
    tot_missings = 0
    for regressor in regressor_names:
        arcpy.SelectLayerByAttribute_management(layer_scratch, "NEW_SELECTION", "{} IS NULL".format(regressor))
        n_missings = int(arcpy.GetCount_management(layer_scratch).getOutput(0))
        _verbose_print("Regressor {} number of missings {}".format(regressor, n_missings))
        if n_missings > 0:
            tot_missings += n_missings
            deal_missings()
    _verbose_print(text.format(tot_missings))


def execute(self, parameters, messages):
    global MESSAGES
    MESSAGES = messages

    print_parameters(parameters)
    parameter_dic = {par.name: par for par in parameters}

    class1_points = parameter_dic["class1_points"].valueAsText
    class2_points = parameter_dic["class2_points"].valueAsText
    rasters_name = parameter_dic["info_rasters"].valueAsText
    missing_value = parameter_dic["missing_value"].valueAsText
    output = parameter_dic["output"].valueAsText
    field_name = parameter_dic["field_name"].valueAsText
    copy_data = parameter_dic["copy_data"].value

    if class1_points is None and class2_points is None:
        MESSAGES.addErrorMessage("Error with {} and {} \n At least one of the fields must be filled".format(
            parameter_dic["deposit_points"].displayName, parameter_dic["non_deposit_points"].displayName
        ))
        raise ValueError

    scratch_files = []
    try:

        merged_name = _merge_fields(class1_points, class2_points, field_name, copy_data)
        scratch_files.append(merged_name)

        if rasters_name is None:
            extracted_name = merged_name
        else:
            extracted_name, regressors = _extract_fields(merged_name, rasters_name)
            # scratch_files.append(extracted_name)
            _clean_data(extracted_name, regressors, missing_value)

        arcpy.CopyFeatures_management(extracted_name, output)
        _verbose_print("Output file created: {}".format(output))
    except:
        raise
    finally:
        for s_file in scratch_files:
            arcpy.Delete_management(s_file)
            _verbose_print("Scratch file deleted: {}".format(s_file))
    return output
