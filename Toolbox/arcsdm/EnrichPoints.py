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


def _add_calculate_field(base, value):
    scratch = arcpy.CreateScratchName("temp", data_type="FeatureClass", workspace=arcpy.env.scratchWorkspace)
    try:
        arcpy.Copy_management(arcpy.Describe(base).catalogPath, scratch)
    except arcpy.ExecuteError:
        arcpy.Delete_management(scratch)
        arcpy.CopyFeatures_management(arcpy.Describe(base).catalogPath, scratch)

    arcpy.AddField_management(scratch, "Prospect", "LONG", "", "", "", "", "NULLABLE", "NON_REQUIRED", "")
    arcpy.CalculateField_management(scratch, "Prospect", str(value), "PYTHON", "")
    return scratch


def _merge_fields(deposits_name, non_deposits_name, messages):
    _verbose_print("Merge fields...", messages)
    _verbose_print("Deposits: {}".format(deposits_name), messages)
    _verbose_print("Non deposits: {}".format(non_deposits_name), messages)
    scratch_files = []

    try:
        if deposits_name is None:
            merged_name = _add_calculate_field(non_deposits_name, -1)
            _verbose_print("Scratch file created (add): {}".format(merged_name), messages)
            _verbose_print("Number of non deposits {}".format(arcpy.GetCount_management(merged_name).getOutput(0)),
                           messages)
        elif non_deposits_name is None:
            merged_name = _add_calculate_field(deposits_name, 1)
            _verbose_print("Scratch file created (add): {}".format(merged_name), messages)
            _verbose_print("Number of deposits {}".format(arcpy.GetCount_management(merged_name).getOutput(0)),
                           messages)
        else:
            messages.AddMessage("Merging Data Points...")

            deposit_scratch = _add_calculate_field(deposits_name, 1)
            scratch_files.append(deposit_scratch)
            _verbose_print("Scratch file created (add): {}".format(deposit_scratch), messages)
            _verbose_print("Number of non deposits {}".format(arcpy.GetCount_management(deposit_scratch).getOutput(0)),
                           messages)
            non_deposit_scratch = _add_calculate_field(non_deposits_name, -1)
            scratch_files.append(non_deposit_scratch)
            _verbose_print("Scratch file created (add): {}".format(non_deposit_scratch), messages)
            _verbose_print("Number of non deposits {}".format(arcpy.GetCount_management
                                                              (non_deposit_scratch).getOutput(0)), messages)
            merged_name = arcpy.CreateScratchName("temp", data_type="FeatureClass",
                                                  workspace=arcpy.env.scratchWorkspace)
            arcpy.Merge_management([deposit_scratch, non_deposit_scratch], merged_name)
            _verbose_print("Scratch file created (merge): {}".format(merged_name), messages)
            _verbose_print("Total Number of points {}".format(arcpy.GetCount_management(merged_name).getOutput(0)),
                           messages)
            return merged_name
    except:
        raise
    finally:
        for s_file in scratch_files:
            arcpy.Delete_management(s_file)
            _verbose_print("Scratch file deleted: {}".format(s_file), messages)
    return merged_name


def _extract_fields(base, rasters, messages):
    messages.AddMessage("Assigning Raster information...")
    _verbose_print("Base: {}".format(base), messages)
    _verbose_print("Rasters: {}".format(rasters), messages)

    rasters = [x.strip("'") for x in rasters.split(";")]
    scratch_files = []

    try:
        regressor_names = []
        arcpy.SetProgressor("step", "Adding raster values to the points", min_range=0, max_range=len(rasters),
                            step_value=1)
        _verbose_print("Adding raster values to the points", messages)
        for raster in rasters:
            try:
                _verbose_print("Adding information of {}".format(raster), messages)
                extracted_name = arcpy.CreateScratchName("temp", data_type="FeatureClass",
                                                         workspace=arcpy.env.scratchWorkspace)
                arcpy.gp.ExtractValuesToPoints(base, raster, extracted_name, "INTERPOLATE",
                                               "VALUE_ONLY")
                _verbose_print("Scratch file created (merge): {}".format(extracted_name), messages)
                scratch_files.append(extracted_name)
                arcpy.AlterField_management(extracted_name, "RASTERVALU", arcpy.Describe(raster).baseName)
                regressor_names.append(arcpy.Describe(raster).baseName)
                base = extracted_name
                arcpy.SetProgressorPosition()
            except:
                messages.addErrorMessage("Problem with raster {}".format(raster))
                raise
        scratch_files.remove(extracted_name)
        arcpy.SetProgressorLabel("Executing Enrich Points")
        arcpy.ResetProgressor()
    except:
        raise
    finally:
        for s_file in scratch_files:
            arcpy.Delete_management(s_file)
            _verbose_print("Scratch file deleted: {}".format(s_file), messages)
    return extracted_name, regressor_names


def _clean_data(base, regressor_names, missing_value, messages):
    messages.AddMessage("Clearing Missing Data...")
    _verbose_print("Base: {}".format(base), messages)
    _verbose_print("Regressor names: {}".format(regressor_names), messages)
    _verbose_print("Missing value: {}".format(missing_value), messages)

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
        _verbose_print("Regressor {} number of missings {}".format(regressor, n_missings), messages)
        if n_missings > 0:
            tot_missings += n_missings
            deal_missings()
    _verbose_print(text.format(tot_missings), messages)


def execute(self, parameters, messages):
    print_parameters(parameters, messages)
    parameter_dic = {par.name: par for par in parameters}

    deposits_name = parameter_dic["deposit_points"].valueAsText
    non_deposits_name = parameter_dic["non_deposit_points"].valueAsText
    rasters_name = parameter_dic["info_rasters"].valueAsText
    missing_value = parameter_dic["missing_value"].valueAsText
    output = parameter_dic["output"].valueAsText

    if deposits_name is None and non_deposits_name is None:
        messages.addErrorMessage("Error with {} and {} \n At least one of the fields must be filled".format(
            parameter_dic["deposit_points"].displayName, parameter_dic["non_deposit_points"].displayName
        ))
        raise ValueError

    scratch_files = []
    try:

        merged_name = _merge_fields(deposits_name, non_deposits_name, messages)
        scratch_files.append(merged_name)

        if rasters_name is None:
            extracted_name = merged_name
        else:
            extracted_name, regressors = _extract_fields(merged_name, rasters_name, messages)
            scratch_files.append(extracted_name)
            _clean_data(extracted_name, regressors, missing_value, messages)

        arcpy.CopyFeatures_management(extracted_name, output)
        _verbose_print("Output file created: {}".format(output), messages)
    except:
        raise
    finally:
        for s_file in scratch_files:
            arcpy.Delete_management(s_file)
            _verbose_print("Scratch file deleted: {}".format(s_file), messages)
    return output
