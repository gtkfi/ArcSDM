"""
    Apply Model tool

    This module contains the execution code for the tool Apply Model
    - Just the execute function should be called from the exterior, all other functions are called by this.
    - This module makes use of the non-standard modules:
        arcpy: for GIS operations and communication with ArcGIS. Is distributed along with ArcGIS desktop software


    Authors: Irving Cabrera <irvcaza@gmail.com>
"""

import arcpy
import os

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


def _constrain_from_points(constrain_area, excluding_points, excluding_distance, select_inside):
    """
    _constrain_from_points
        Constrains an area to intersect/exclude zones around given points
         
    :param constrain_area: General area to be constrained 
    :param excluding_points: Seed points to create the areas 
    :param excluding_distance: Radius of the areas 
    :param select_inside: Boolean to select if the area should be intersected (True) or excluded (false)
    
    :return: Area after be intersected/excluded  
    """
    global MESSAGES
    MESSAGES.AddMessage("Constraining Area from points...")
    _verbose_print("Constrain Area: {}".format(constrain_area))
    _verbose_print("Excluding points : {}".format(excluding_points))
    _verbose_print("Excluding distance : {}".format(excluding_distance))
    _verbose_print("select inside : {}".format(select_inside))

    scratch_files = []
    try:
        # Create the buffer area from the points
        buffer_scratch = arcpy.CreateScratchName("buff_sct.shp", workspace=arcpy.env.scratchWorkspace)
        arcpy.Buffer_analysis(excluding_points, buffer_scratch, excluding_distance, dissolve_option="ALL")
        scratch_files.append(buffer_scratch)
        _verbose_print("Scratch file created (buffer): {}".format(buffer_scratch))
        combined_scratch = arcpy.CreateScratchName("comb_sct.shp", workspace=arcpy.env.scratchWorkspace)
        # Intersect/Delete from the original area
        if select_inside:
            _verbose_print("Intersect selected")
            arcpy.Intersect_analysis([buffer_scratch, constrain_area], combined_scratch)
        else:
            _verbose_print("Erase selected")
            arcpy.Erase_analysis(constrain_area, buffer_scratch, combined_scratch)
        _verbose_print("Scratch file created (erase): {}".format(combined_scratch))
    except:
        _verbose_print("Error constraining from points")
        raise
    finally:
        # This process creates files that will not be needed and are erased at the end of the execution or when  an error is found
        for s_file in scratch_files:
            _verbose_print("Scratch file deleted: {}".format(s_file))
            arcpy.Delete_management(s_file)

    _verbose_print("Constrain from points finished")
    return combined_scratch


def _constrain_from_raster(constrain_area, rasters):
    """
    _constrain_from_raster
        Constrains an area to zones where all bands of the raster contain information 
        
    :param constrain_area: Original area  
    :param rasters: Multiband rasters to be used as information source
    
    :return: Constrained area 
    """

    global MESSAGES
    MESSAGES.AddMessage("Constraining Area from rasters...")
    _verbose_print("Constrain Area: {}".format(constrain_area))
    _verbose_print("rasters: {}".format(rasters))

    scratch_files = []

    # Obtain the name of the bands
    oldws = arcpy.env.workspace  # Save previous workspace
    raster_path = arcpy.Describe(rasters.strip("'")).catalogPath
    arcpy.env.workspace = raster_path
    rasters = [os.path.join(raster_path, b) for b in arcpy.ListRasters()]
    arcpy.env.workspace = oldws     # Restore previous workspace
    _verbose_print("Rasters list: {}".format(str(rasters)))

    # Start a progression bar to feedback for the user
    arcpy.SetProgressor("step", "Restricting area from missings", min_range=0, max_range=len(rasters),
                        step_value=1)

    try:
        # TODO: Maybe this is faster if is transform to numpy arrays, make calculations and the back to raster
        # Initialize raster with all the Null points
        final_raster = arcpy.sa.IsNull(arcpy.sa.Raster(rasters[0]))
        arcpy.SetProgressorPosition()
        # loop trough all the remaining rasters adding the points where other bands have missings
        if len(rasters) > 1:
            for raster in rasters[1:]:
                final_raster = arcpy.sa.BooleanOr(final_raster, arcpy.sa.IsNull(arcpy.sa.Raster(raster)))
                arcpy.SetProgressorPosition()
                _verbose_print("Area reduced with nulls from {}".format(raster))
        # Set null the positions where it was found at least one null
        final_raster = arcpy.sa.SetNull(final_raster, final_raster)
        # reset the Progressor to previous state
        arcpy.SetProgressorLabel("Executing Select Random Points")
        arcpy.ResetProgressor()

        # Transform the raster to polygon
        domain_scratch = arcpy.CreateScratchName("temp", workspace=arcpy.env.scratchWorkspace)
        arcpy.RasterToPolygon_conversion(final_raster, domain_scratch, "SIMPLIFY")
        scratch_files.append(domain_scratch)
        _verbose_print("Scratch file created (domain): {}".format(domain_scratch))

        # Intersect the polygon created with the original area
        intersect_scratch = arcpy.CreateScratchName("temp", workspace=arcpy.env.scratchWorkspace)
        arcpy.Intersect_analysis([domain_scratch, constrain_area], intersect_scratch)
        _verbose_print("Scratch file created (intersect): {}".format(domain_scratch))

    except:
        raise

    finally:
        # Clean up intermediate files
        for s_file in scratch_files:
            arcpy.Delete_management(s_file)
            _verbose_print("Scratch file deleted: {}".format(s_file))

    _verbose_print("Constrain from rasters finished")
    return intersect_scratch


def execute(self, parameters, messages):
    """
        Create random points tool 
            Create a set of random points constrained to a specific area, maintaining a minimum maximum distance to a 
                set of points and remaining in areas with full information 
        :param parameters: parameters object with all the parameters from the python-tool. It contains:
            output: Name of the file where the points will be stored 
            n_points: number of random points to be created 
            constrain_area: Original constraining area used as baseline for further constraints
            rasters: Information rasters that will be used to restrict to areas with full information 
            buffer_points: The random points will maintain a minimal/maximal distance to these points
            buffer_distance: Distance away buffer points
            min_distance: Minimal distance along created points 
            select_inside: Boolean to  create the points inside the area (True) or outside the area (False) 
        :param messages: messages object to print in the console, must implement AddMessage
         
        :return: None
    """

    global MESSAGES
    MESSAGES = messages

    # Print parameters for debugging purposes
    print_parameters(parameters)
    parameter_dic = {par.name: par for par in parameters}

    output = parameter_dic["output_points"].valueAsText.strip("'")
    n_points = parameter_dic["number_points"].value
    constrain_area = parameter_dic["constraining_area"].valueAsText.strip("'")
    rasters = parameter_dic["constraining_rasters"].valueAsText
    buffer_points = parameter_dic["buffer_points"].valueAsText
    buffer_distance = parameter_dic["buffer_distance"].valueAsText
    min_distance = parameter_dic["minimum_distance"].valueAsText
    select_inside = parameter_dic["select_inside"].value

    # Split the path of the output file in file and database, necesarry for
    out_ws , out_f = os.path.split(output)

    scratch_files = []
    try:
        # constrain area to avoid modifications to the original
        constrain_scratch = arcpy.CreateScratchName("const_sct.shp", workspace=arcpy.env.scratchWorkspace)
        arcpy.CopyFeatures_management(arcpy.Describe(constrain_area).catalogPath, constrain_scratch)
        scratch_files.append(constrain_scratch)
        _verbose_print("Scratch file created (constrain): {}".format(constrain_scratch))
        # Constrain to the points only if they exist, otherwise is not constrained
        if buffer_points is None or buffer_distance is None:
            _verbose_print("Exclude from points omitted")
            points_scratch = constrain_scratch
        else:
            points_scratch = _constrain_from_points(constrain_scratch, buffer_points, buffer_distance, select_inside)
            scratch_files.append(points_scratch)
        # Constrain to the information raster only if is specified, otherwise do not constrain
        if rasters is None:
            _verbose_print("Exclude from rasters omitted")
            rasters_scratch = points_scratch
        else:
            rasters_scratch = _constrain_from_raster(points_scratch, rasters)
            scratch_files.append(rasters_scratch)
        # Dissolve the polygon into a single object to make the selection
        dissolve_scratch = arcpy.CreateScratchName("diss_sct.shp", workspace=arcpy.env.scratchWorkspace)
        arcpy.Dissolve_management(in_features=rasters_scratch, out_feature_class=dissolve_scratch,
                                  multi_part="MULTI_PART")
        scratch_files.append(dissolve_scratch)

        # Select the random points
        # TODO: Sometimes, random points fall right in the border of the rasters and therefore they show null information, an erosion needs to be added to avoid this
        result = arcpy.CreateRandomPoints_management(out_ws, out_f, dissolve_scratch, number_of_points_or_field=n_points, minimum_allowed_distance=min_distance)
        arcpy.DefineProjection_management(result, arcpy.Describe(constrain_area).spatialReference)
        MESSAGES.AddMessage("Random points saved in {}".format(result))
    except:
        raise
    finally:
        # Delete intermediate files
        for s_file in scratch_files:
            arcpy.Delete_management(s_file)
            _verbose_print("Scratch file deleted: {}".format(s_file))

    return
