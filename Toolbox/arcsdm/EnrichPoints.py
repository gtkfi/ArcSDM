"""
    Enrich Points tool

    This module contains the execution code for the tool Apply Model
    - Just the execute function should be called from the exterior, all other functions are called by this.
    - This module makes use of the non-standard modules:
        arcpy: for GIS operations and communication with ArcGIS. Is distributed along with ArcGIS desktop software


    Authors: Irving Cabrera <irvcaza@gmail.com>
"""
import arcpy

# Global Name for the messages object and be called from any function
MESSAGES = None

# Verbosity switch, True to print more detailed information
VERBOSE = True

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


def _add_calculate_field(base, field_name, copy_data, value):
    """
    _add_calculate_field
        Copies a feature, optionally erases all non-required fields and create one with a given value
          
    :param base: Name of the feature to be copied  
    :param field_name: Name of the new field
    :param copy_data: Boolean that selects if the previous data is kept (true) or deleted 
    :param value: Value to be assigned to the new field
    
    :return: Name of the copied feature
    """
    # Copy the feature to scratch
    scratch = arcpy.CreateScratchName("temp", data_type="FeatureClass", workspace=arcpy.env.scratchWorkspace)
    try:
        arcpy.Copy_management(arcpy.Describe(base).catalogPath, scratch)
    except arcpy.ExecuteError:
        arcpy.Delete_management(scratch)
        arcpy.CopyFeatures_management(arcpy.Describe(base).catalogPath, scratch)

    # Delete de data if indicated
    if not copy_data:
        drop_field = [f.name for f in arcpy.ListFields(scratch) if not f.required]
        _verbose_print("Fields to delete: " + str(drop_field))
        arcpy.DeleteField_management (scratch, drop_field)

    # Add and calculate the field
    arcpy.AddField_management(scratch, field_name, "LONG", "", "", "", "", "NULLABLE", "NON_REQUIRED", "")
    arcpy.CalculateField_management(scratch, field_name, str(value), "PYTHON", "")
    return scratch


def _merge_fields(deposits_name, non_deposits_name, field_name, copy_data):
    """
        _merge_fields
            Merges two feature classes into one labelling each one 
            
    :param deposits_name: Name of the feature labeled as 1
    :param non_deposits_name: Name of the feature labeled as -1 
    :param field_name: Name of the label field
    :param copy_data: Boolean that selects if the previous data is kept (true) or deleted 
    
    :return: Name with the merged feature 
    """
    global MESSAGES
    _verbose_print("Merge fields...")
    _verbose_print("Deposits: {}".format(deposits_name))
    _verbose_print("Non deposits: {}".format(non_deposits_name))
    scratch_files = []

    try:
        # If jus t one feature is given, the merging is not necessary
        if deposits_name is None:
            merged_name = _add_calculate_field(non_deposits_name, field_name, copy_data, -1)
            # TODO: change next two statgementes inside _add_calculate_field
            _verbose_print("Scratch file created (add): {}".format(merged_name))
            _verbose_print("Number of non deposits {}".format(arcpy.GetCount_management(merged_name).getOutput(0)))
        elif non_deposits_name is None:
            merged_name = _add_calculate_field(deposits_name, field_name, copy_data, 1)
            _verbose_print("Scratch file created (add): {}".format(merged_name))
            _verbose_print("Number of deposits {}".format(arcpy.GetCount_management(merged_name).getOutput(0)))
        else:
            MESSAGES.AddMessage("Merging Data Points...")
            #Make a copy of the first feature
            deposit_scratch = _add_calculate_field(deposits_name, field_name, copy_data, 1)
            scratch_files.append(deposit_scratch)
            _verbose_print("Scratch file created (add): {}".format(deposit_scratch))
            _verbose_print("Number of deposits {}".format(arcpy.GetCount_management(deposit_scratch).getOutput(0)))
            #Make a copy of the second feature
            non_deposit_scratch = _add_calculate_field(non_deposits_name, field_name, copy_data, -1)
            scratch_files.append(non_deposit_scratch)
            _verbose_print("Scratch file created (add): {}".format(non_deposit_scratch))
            _verbose_print("Number of non deposits {}".format(arcpy.GetCount_management
                                                              (non_deposit_scratch).getOutput(0)))
            # Merge the copied features
            merged_name = arcpy.CreateScratchName("temp", data_type="FeatureClass",
                                                  workspace=arcpy.env.scratchWorkspace)
            arcpy.Merge_management([deposit_scratch, non_deposit_scratch], merged_name)
            _verbose_print("Scratch file created (merge): {}".format(merged_name))
            _verbose_print("Total Number of points {}".format(arcpy.GetCount_management(merged_name).getOutput(0)))
    except:
        raise
    finally:
        # Clean up temporary files
        for s_file in scratch_files:
            arcpy.Delete_management(s_file)
            _verbose_print("Scratch file deleted: {}".format(s_file))
    return merged_name


def _extract_fields(base, rasters):
    """
    _extract_fields
        Extracts the value of the rasters at points' position and writes it in the points 
    :param base: feature with the points that will receive information 
    :param rasters: Multiband raster that will give information to the points 
    :return: feature with the points with the data and list with the name of fields of the the raster
    """
    global MESSAGES
    MESSAGES.AddMessage("Assigning Raster information...")
    _verbose_print("Base: {}".format(base))
    _verbose_print("Rasters: {}".format(rasters))

    old_fields = arcpy.ListFields(base)

    # TODO: rename fields to meaningful names
    arcpy.sa.ExtractMultiValuesToPoints (base, rasters)

    regressor_names = [n.name for n in arcpy.ListFields(base) if n not in old_fields]

    return base, regressor_names


def _clean_data(base, regressor_names, missing_value):
    """
    _clean_data
        Eliminates Missing from the data, imputing a pre-established value or deleting the observation 
    
    :param base: Feature to be cleaned  
    :param regressor_names: Name of the fields that need to be cleaned 
    :param missing_value: Value to be imputed, if none the observation is deleted
    :return: 
    """
    global MESSAGES
    MESSAGES.AddMessage("Clearing Missing Data...")
    _verbose_print("Base: {}".format(base))
    _verbose_print("Regressor names: {}".format(regressor_names))
    _verbose_print("Missing value: {}".format(missing_value))

    layer_scratch = "layer_scratch"
    arcpy.MakeFeatureLayer_management(base, layer_scratch)
    arcpy.SelectLayerByAttribute_management(layer_scratch, "CLEAR_SELECTION")

    # Selects how to deal with missings
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
        _verbose_print("regressor: {}".format(regressor))
        if regressor == 'Shape':
            continue
        arcpy.SelectLayerByAttribute_management(layer_scratch, "NEW_SELECTION", "{} IS NULL".format(regressor))
        n_missings = int(arcpy.GetCount_management(layer_scratch).getOutput(0))
        _verbose_print("Regressor {} number of missings {}".format(regressor, n_missings))
        if n_missings > 0:
            tot_missings += n_missings
            deal_missings()
    _verbose_print(text.format(tot_missings))


def execute(self, parameters, messages):
    """
        Enrich Point tools
            Takes a set of deposits and non-deposits, labels them and adds the information of the rasters in a 
                single feature 
        :param parameters: parameters object with all the parameters from the python-tool. It contains:
            class1_points: Name of the feature labelled as 1
            class2_points: Name of the feature labelled as 1
            rasters_name: Multiband raster with the information to be included in the points  
            missing_value: Imputed value for missings. If none then the point gets deleted 
            output: Name of the unique feature with the data 
            field_name: Name of the field With the class label 
            copy_data: Boolean that selects if the previous data is kept (true) or deleted
        :param messages: messages object to print in the console, must implement AddMessage
         
        :return: Name of the output feature 
    """
    global MESSAGES
    MESSAGES = messages

    # Print parameters for debugging purposes
    print_parameters(parameters)
    parameter_dic = {par.name: par for par in parameters}

    class1_points = parameter_dic["class1_points"].valueAsText
    class2_points = parameter_dic["class2_points"].valueAsText
    rasters_name = parameter_dic["info_rasters"].valueAsText
    missing_value = parameter_dic["missing_value"].valueAsText
    output = parameter_dic["output"].valueAsText
    field_name = parameter_dic["field_name"].valueAsText
    copy_data = parameter_dic["copy_data"].value

    # If both deposits and non deposits are epty display an error
    if class1_points is None and class2_points is None:
        MESSAGES.addErrorMessage("Error with {} and {} \n At least one of the fields must be filled".format(
            parameter_dic["deposit_points"].displayName, parameter_dic["non_deposit_points"].displayName
        ))
        raise ValueError

    scratch_files = []
    try:

        # Merge both fields into one
        merged_name = _merge_fields(class1_points, class2_points, field_name, copy_data)
        scratch_files.append(merged_name)

        # If a raster is given, add the information to the points
        if rasters_name is None:
            extracted_name = merged_name
        else:
            extracted_name, regressors = _extract_fields(merged_name, rasters_name)
            scratch_files.append(extracted_name)
            _clean_data(extracted_name, regressors, missing_value)

        # Create the final feature
        arcpy.CopyFeatures_management(extracted_name, output)
        _verbose_print("Output file created: {}".format(output))
    except:
        raise
    finally:
        # Delete intermediate files
        for s_file in scratch_files:
            arcpy.Delete_management(s_file)
            _verbose_print("Scratch file deleted: {}".format(s_file))
    return output
