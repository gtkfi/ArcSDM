"""
    Apply Model tool

    This module contains the execution code for the tool Multiclass Split
    - Just the execute function should be called from the exterior, all other functions are called by this.
    - This module makes use of the non-standard modules:
        arcpy: for GIS operations and communication with ArcGIS. Is distributed along with ArcGIS desktop software


    Authors: Irving Cabrera <irvcaza@gmail.com>
"""

import arcpy


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
    #TODO: Implement this
    return


def execute(self, parameters, messages):
    """
        Multiclass Split tool 
            Create a map with the distance the closest polygon of a certain class of a feature class.
            It creates as many rasters as unique values in the given field 
        :param parameters: parameters object with all the parameters from the python-tool. It contains:
            input_feature = Feature class to be split 
            class_field = Name of the field that will group the geometries
            output_prefix = Prefix of the maps to be output, the final name will be [prefix][field value]
            transformation = Transformation to be made to the distance measure 
            max_distance = Threshold value, values greater than this wil be cut down to this 
        :param messages: messages object to print in the console, must implement AddMessage

        :return: None
    """
    global MESSAGES
    MESSAGES = messages

    # Print parameters for debugging purposes
    print_parameters(parameters)

    # Check for correctness in the parameters
    _input_validation(parameters)

    parameter_dic = {par.name: par for par in parameters}
    input_feature = parameter_dic["input_feature"].valueAsText
    class_field = parameter_dic["class_field"].valueAsText
    output_prefix = parameter_dic["output_prefix"].valueAsText
    transformation = parameter_dic["transformation"].valueAsText
    max_distance = parameter_dic["max_distance"].value

    scratch_files = []

    scratch_base = arcpy.CreateScratchName("temp", data_type="FeatureClass", workspace=arcpy.env.scratchWorkspace)
    try:
        # Create a copy of the feature class
        try:
            arcpy.Copy_management(arcpy.Describe(input_feature).catalogPath, scratch_base)
        except arcpy.ExecuteError:
            arcpy.Delete_management(scratch_base)
            arcpy.CopyFeatures_management(arcpy.Describe(input_feature).catalogPath, scratch_base)
        _verbose_print("Scratch file created {}".format(scratch_base))
        scratch_files.append(scratch_base)

        # Look for unique values
        values = []
        with arcpy.da.SearchCursor(scratch_base, class_field) as cursor:
            for row in cursor:
                values.append(row[0])
        unique_values = list(set(values))
        _verbose_print("Unique values: {}".format(str(unique_values)))

        # Create a layer to be able to select
        layer_scratch = "layer_scratch"
        arcpy.MakeFeatureLayer_management(scratch_base, layer_scratch)
        arcpy.SelectLayerByAttribute_management(layer_scratch, "CLEAR_SELECTION")

        # If the spliting field is text o numbers the expression needs to be different
        for field in arcpy.ListFields(scratch_base):
            if field.name == class_field:
                if field.type in ("Double", "Integer", "Single", "SmallInteger"):
                    expression_base = u"{} = {}"
                else:
                    expression_base = u"{} = '{}'"
                _verbose_print("{} is of type {}".format(class_field, field.type))
                break

        for orig_val in unique_values:
            # File names just accept unicode names, for values not in unicode a transformation is done
            try:
                unicode_val = unicode(orig_val)
                fullname = output_prefix + unicode_val.encode("ascii",'ignore').replace(" ","_")
            except NameError:
                unicode_val = str(orig_val)
                fullname = output_prefix + unicode_val.replace(" ","_")
            if " " in unicode_val:
                arcpy.SelectLayerByAttribute_management(layer_scratch, "NEW_SELECTION",
                                                        u"{} = '{}'".format(class_field, orig_val))
            else:
                arcpy.SelectLayerByAttribute_management(layer_scratch, "NEW_SELECTION",
                                                        expression_base.format(class_field, orig_val))
            # Calculate distance to selected feature
            raster = arcpy.sa.EucDistance(layer_scratch)
            # Apply the threshold
            # TODO: Maxbe move this after transformation and rename it max value
            if max_distance is not None:
                raster = ( raster + max_distance - arcpy.sa.Abs(raster - max_distance)) / 2

            # Apply corresponding transformation
            if transformation == "Inverse Linear Distance":
                max_val = float(arcpy.GetRasterProperties_management(raster,"MAXIMUM").getOutput(0))
                raster = 1 - (raster * 1.0) / max_val
            elif transformation == "Inverse Distance": # TODO : do it in python to increase precision
                raster = 1 / (raster + 1.0)
            elif transformation == "Binary":
                raster = raster == 0
            elif transformation == "Logarithmic":
                raster = arcpy.sa.Ln(raster +1 )

            # Save calculated raster
            raster.save(fullname)
            _verbose_print("Output file created: {}".format(fullname))
    except:
        raise
    finally:
        # Clean temporary files
        for s_f in scratch_files:
            arcpy.Delete_management(s_f)
            _verbose_print("Scratch file deleted {}".format(s_f))

