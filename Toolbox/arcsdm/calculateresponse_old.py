"""
    ArcSDM 6 ToolBox for ArcGIS Pro

    Conversion and tool development for ArcGIS Pro by Geological Survey of Finland (GTK), 2024.

    History:
    4/2016 Conversion started - TR
    9/2016 Conversion started to Python toolbox TR
    01/2018 Bug fixes for 10.x - Arianne Ford
    27.4.2020 added Input Weights table file type checking /Arto Laiho, GTK/GFS
    18.5.2020 added Input Raster coordinate system checking /Arto Laiho
    15.6.2020 added "arcsdm." to import missingdatavar_func /Arto Laiho
    20-23.7.2020 combined with Unicamp fixes (made 19.7.2018) /Arto Laiho
    6.10.2020 If using GDB database, remove numbers and underscore from the beginning of all output table names /Arto Laiho

    Spatial Data Modeller for ESRI* ArcGIS 9.2
    Copyright 2007
    Gary L Raines, Reno, NV, USA: production and certification
    Don L Sawatzky, Spokane, WA, USA: Python software development
    
# ---------------------------------------------------------------------------
# NewCalcResp3.py
#  Modifications for use of Lookup_sa by Ryan Bryn, ESRI
# ---------------------------------------------------------------------------
"""

import arcgisscripting
import arcpy
import gc
import math
import os
import sys
import traceback

from arcsdm.common import log_arcsdm_details
from arcsdm.missingdatavar_func import create_missing_data_variance_raster
from arcsdm.wofe_common import check_wofe_inputs, get_study_area_parameters


debuglevel = 0


def testdebugfile():
    returnvalue = 0 # This because python sucks in detecting outputs from functions
    import sys
    import os
    if (debuglevel > 0):
        return 1
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if os.path.isfile(dir_path + "/DEBUG"):
        return 1
    return returnvalue


def dwrite(message):
    debug = testdebugfile()
    if (debuglevel > 0 or debug > 0):
        arcpy.AddMessage("Debug: " + message)


def Execute(self, parameters, messages):
    try:
        # TODO: Refactor to arcpy.
        gp = arcgisscripting.create()

        arcpy.CheckOutExtension("Spatial")

        arcpy.env.overwriteOutput = True
        arcpy.AddMessage("Setting overwriteOutput to True")

        arcpy.SetLogHistory(True)
        arcpy.AddMessage("Setting LogHistory to True")

        input_evidence_rasters = parameters[0].valueAsText
        input_weights_tables = parameters[1].valueAsText
        training_points_feature = parameters[2].valueAsText
        is_ignore_missing_data_selected = parameters[3].value
        missing_data_value = parameters[4].value
        unit_cell_area_sq_km = parameters[5].value
        output_pprb_raster = parameters[6].valueAsText
        output_std_raster = parameters[7].valueAsText
        output_mdvar_raster = parameters[8].valueAsText
        output_total_std_raster = parameters[9].valueAsText
        output_confidence_raster = parameters[10].valueAsText
        
        if is_ignore_missing_data_selected: # for nodata argument to CopyRaster tool
            nodata_value = missing_data_value
        else:
            nodata_value = '#'

        input_rasters = input_evidence_rasters.split(";")
        weights_tables = input_weights_tables.split(";")

        if len(input_rasters) != len(weights_tables):
            raise ValueError("The number of evidence rasters should equal the number of weights tables!")

        check_wofe_inputs(input_rasters, training_points_feature)

        log_arcsdm_details()
        total_area_sq_km_from_mask, training_point_count = get_study_area_parameters(unit_cell_area_sq_km, training_points_feature)
        
        area_cell_count = total_area_sq_km_from_mask / unit_cell_area_sq_km
        arcpy.AddMessage(("%-20s %s" % ("Study Area:", str(area_cell_count))))
        arcpy.AddMessage("%-20s %s" % ("# training points:", str(training_point_count)))

        prior_probability = training_point_count / area_cell_count
        arcpy.AddMessage("%-20s %s"% ("Prior probability:" , str(prior_probability)))



        arcpy.AddMessage(f"Input rasters: {input_rasters}")
        
        # Create weight raster from raster's associated weights table
        tmp_weights_rasters = []
        tmp_std_rasters = []
        # Create a list for the Missing Data Variance tool.
        rasters_with_missing_data = []
        # Evidence rasters should have missing data values, where necessary, for
        # NoData cell values within study area.
        # For each input_raster create a weights raster from the raster and its weights table.

        workspace_type = arcpy.Describe(arcpy.env.workspace).workspaceType
        
        arcpy.AddMessage("\nCreating tmp weight and STD rasters...")
        arcpy.AddMessage("=" * 41)

        i = 0
        while i < len(input_rasters):
            input_raster = input_rasters[i]
            weights_table = weights_tables[i]

            # Check each Input Raster datatype and coordinate system
            inputDescr = arcpy.Describe(input_raster)
            inputCoord = inputDescr.spatialReference.name
            arcpy.AddMessage(f"{input_raster}, Data type: {inputDescr.dataType}, Coordinate System: {inputCoord}")
            
            trainingDescr = arcpy.Describe(training_points_feature)
            trainingCoord = trainingDescr.spatialReference.name

            if inputCoord != trainingCoord:
                arcpy.AddError(f"ERROR: Coordinate System of Input Raster is {inputCoord} and Training points it is {trainingCoord}. These must be same.")
                raise

            # When workspace type is File System, Input Weight Table also must end with .dbf
            # If using GDB database, remove numbers and underscore from the beginning of the name (else block)
            if workspace_type == "FileSystem":
                if not(weights_table.endswith(".dbf")):
                    weights_table += ".dbf"
            else:
                wtsbase = os.path.basename(weights_table)
                while len(wtsbase) > 0 and (wtsbase[:1] <= "9" or wtsbase[:1] == "_"):
                    wtsbase = wtsbase[1:]
                weights_table = os.path.dirname(weights_table) + "\\" + wtsbase

            # Compare workspaces to make sure they match
            
            arcpy.AddMessage(f"Processing {input_raster}")
            
            w_raster_name = input_raster.replace(".", "_")
            w_raster_name = w_raster_name[:10] + "W"
            # Needs to be able to extract input raster name from full path.
            # Can't assume only a layer from ArcMap.
            std_raster_name = os.path.basename(input_raster[:9]).replace(".", "_") + "S" # No . allowed in filegeodatgabases

            # If using GDB database, remove numbers and underscore from the beginning of the names
            if workspace_type != "FileSystem":
                while len(w_raster_name) > 0 and (w_raster_name[:1] <= "9" or w_raster_name[:1] == "_"):
                    w_raster_name = w_raster_name[1:]
                while len(std_raster_name) > 0 and (std_raster_name[:1] <= "9" or std_raster_name[:1] == "_"):
                    std_raster_name = std_raster_name[1:]

            # Create the _W & _S rasters that will be used to calculate output rasters
            output_tmp_w_raster = arcpy.CreateScratchName(w_raster_name, "", "rst", arcpy.env.scratchWorkspace)
            output_tmp_std_raster = arcpy.CreateScratchName(std_raster_name, "", "rst", arcpy.env.scratchWorkspace)
            
            # Increase the count for next round
            i += 1
     
            # Need to create in-memory Raster Layer for Join
            # Check for unsigned integer raster; cannot have negative missing data
            if nodata_value != "#" and arcpy.Describe(input_raster).pixelType.upper().startswith("U"):
                NoDataArg2 = "#"
            else:
                NoDataArg2 = nodata_value

            # Create new rasterlayer from input raster for both the weights and the std raster -> Result RasterLayer
            # These will be in-memory only
            # AddJoin requires an input layer or tableview not Input Raster Dataset.
            tmp_raster_layer = "OutRas_lyr"

            arcpy.management.MakeRasterLayer(input_raster, tmp_raster_layer)
            
            # Join result layer with weights table
            arcpy.management.AddJoin(tmp_raster_layer, "VALUE", weights_table, "CLASS")

            temp_raster = arcpy.CreateScratchName("tmp_rst", "", "rst", arcpy.env.scratchWorkspace)
            
            # Delete existing temp_raster
            if arcpy.Exists(temp_raster):
                arcpy.management.Delete(temp_raster)
                gc.collect()
                arcpy.management.ClearWorkspaceCache()
                arcpy.AddMessage("Deleted tempraster")
            
            # Copy created and joined in-memory raster to temp_raster
            arcpy.CopyRaster_management(tmp_raster_layer, temp_raster, "#", "#", NoDataArg2)

            arcpy.AddMessage(f"Output tmp weights raster: {output_tmp_w_raster}")
            
            # Save weights raster
            weight_lookup = arcpy.sa.Lookup(temp_raster, "WEIGHT")
            weight_lookup.save(output_tmp_w_raster)
            
            if not arcpy.Exists(output_tmp_w_raster):
                arcpy.AddError(f"{output_tmp_w_raster} does not exist.")
                raise
            tmp_weights_rasters.append(output_tmp_w_raster)

            std_lookup = arcpy.sa.Lookup(temp_raster, "W_STD")
            std_lookup.save(output_tmp_std_raster)
            
            if not arcpy.Exists(output_tmp_std_raster):
                arcpy.AddError(f"{output_tmp_std_raster} does not exist.")
                raise
            tmp_std_rasters.append(output_tmp_std_raster)

            # Check for Missing Data in raster's Wts table
            if not is_ignore_missing_data_selected:
                # Update the list for Missing Data Variance Calculation
                tblrows = gp.SearchCursor(weights_table, "Class = %s" % missing_data_value)
                tblrow = tblrows.Next()
                if tblrow:
                    rasters_with_missing_data.append(arcpy.Describe(output_tmp_w_raster).catalogPath)
            
            arcpy.management.Delete(temp_raster)

            arcpy.AddMessage(" ") # Cycle done - add ONE linefeed
        
        # Get Post Logit Raster
        arcpy.AddMessage("\nGetting Post Logit raster...\n" + "=" * 41)

        Input_Data_Str = ' + '.join('"{0}"'.format(w) for w in tmp_weights_rasters)
        arcpy.AddMessage(f"Input_data_str: {Input_Data_Str}")

        Constant = math.log(prior_probability / (1.0 - prior_probability))
        
        if len(tmp_weights_rasters) == 1:
            InExpressionPLOG = "%s + %s" % (Constant, Input_Data_Str)
        else:
            InExpressionPLOG = "%s + (%s)" % (Constant, Input_Data_Str)

        arcpy.AddMessage(f"InexpressionPlog: {InExpressionPLOG}")

        # Get Post Probability Raster
        arcpy.AddMessage("\nCreating Post Probability Raster...\n" + "=" * 41)

        try:
            InExpression = "Exp(%s) / (1.0 + Exp(%s))" % (InExpressionPLOG, InExpressionPLOG)
            arcpy.AddMessage(f"InExpression = {InExpression}")
            arcpy.AddMessage(f"Postprob: {output_pprb_raster}")

            gp.SingleOutputMapAlgebra_sa(InExpression, output_pprb_raster)
            
            arcpy.SetParameterAsText(6, output_pprb_raster)
        except:
            arcpy.AddError(arcpy.GetMessages(2))
            raise
        else:
            arcpy.AddWarning(arcpy.GetMessages(1))
            arcpy.AddMessage(arcpy.GetMessages(0))

        arcpy.AddMessage("\nCreating Post Probability STD Raster...\n" + "=" * 41)
        
        #TODO: Figure out what this does!? TR
        #TODO: This is always false now
        if len(tmp_std_rasters) == 1:
            InExpression = '"%s"' % (tmp_std_rasters[0])
        else:
            SUM_args_list = []
            for Std_Raster in tmp_std_rasters:
                SUM_args_list.append("SQR(\"%s\")" % Std_Raster)
            
            SUM_args = " + ".join(SUM_args_list)

            arcpy.AddMessage("Sum_args: " + SUM_args + "\n" + "=" * 41)
       
            Constant = 1.0 / float(training_point_count)

            InExpression = "SQRT(SQR(%s) * (%s + SUM(%s)))" % (output_pprb_raster, Constant, SUM_args)
            arcpy.AddMessage(f"InExpression = {InExpression}")

        try:
            arcpy.AddMessage(f"InExpression 2 ====> {InExpression}")

            gp.SingleOutputMapAlgebra_sa(InExpression, output_std_raster)

            arcpy.SetParameterAsText(7, output_std_raster)
        except:
            arcpy.AddError(arcpy.GetMessages(2))
            raise
        else:
            arcpy.AddWarning(arcpy.GetMessages(1))
            arcpy.AddMessage(arcpy.GetMessages(0))
        
        # Create Variance of missing data here and create totVar = VarMD + SQR(VarWts)
        if not is_ignore_missing_data_selected:
            # Calculate Missing Data Variance
            if len(rasters_with_missing_data) > 0:
                arcpy.AddMessage("Calculating Missing Data Variance...")
                try:
                    if arcpy.Exists(output_mdvar_raster):
                        arcpy.management.Delete(output_mdvar_raster)

                    create_missing_data_variance_raster(gp, rasters_with_missing_data, output_pprb_raster, output_mdvar_raster)

                    InExpression = 'SQRT(SUM(SQR(%s),%s))' % (output_std_raster, output_mdvar_raster)

                    arcpy.AddMessage("Calculating Total STD...")
                    arcpy.AddMessage(f"InExpression 3 ====> {InExpression}")

                    gp.SingleOutputMapAlgebra_sa(InExpression, output_total_std_raster)

                    arcpy.SetParameterAsText(9, output_total_std_raster)
                except:
                    arcpy.AddError(arcpy.GetMessages(2))
                    raise
            else:
                arcpy.AddWarning("No evidence with missing data. Missing Data Variance not calculated.")
                output_mdvar_raster = None
                output_total_std_raster = output_std_raster
        else:
            arcpy.AddWarning("Missing Data Ignored. Missing Data Variance not calculated.")
            output_mdvar_raster = None
            output_total_std_raster = output_std_raster
        
        # Confidence is PP / sqrt(totVar)
        arcpy.AddMessage("\nCalculating Confidence...\n" + "=" * 41)

        InExpression = "%s / %s" % (output_pprb_raster, output_std_raster)
        arcpy.AddMessage(f"InExpression 4====> {InExpression}")
        try: 
            gp.SingleOutputMapAlgebra_sa(InExpression, output_confidence_raster)
            arcpy.SetParameterAsText(10, output_confidence_raster)
        except:
            arcpy.AddError(arcpy.GetMessages(2))
            raise
        else:
            arcpy.AddWarning(arcpy.GetMessages(1))
            arcpy.AddMessage(arcpy.GetMessages(0))
        
        # Set derived output parameters
        arcpy.SetParameterAsText(6, output_pprb_raster)
        arcpy.SetParameterAsText(7, output_std_raster)

        if output_mdvar_raster and (not is_ignore_missing_data_selected):
            arcpy.SetParameterAsText(8, output_mdvar_raster)
        else:
            arcpy.AddWarning("No Missing Data Variance.")

        if not (output_total_std_raster == output_std_raster):
            arcpy.SetParameterAsText(9, output_total_std_raster)
        else:
            arcpy.AddWarning("Total STD same as Post Probability STD.")

        arcpy.SetParameterAsText(10, output_confidence_raster)

        arcpy.AddMessage("Deleting tmp rasters...")
        for raster in tmp_weights_rasters:
            arcpy.management.Delete(raster)
        
        for raster in tmp_std_rasters:
            arcpy.management.Delete(raster)

        arcpy.AddMessage("Done\n" + "=" * 41)
    except arcpy.ExecuteError as e:
        arcpy.AddError("\n")
        arcpy.AddMessage("Calculate Response caught arcpy.ExecuteError ")
        arcpy.AddError(arcpy.GetMessages(2))
        if len(e.args) > 0:
            args = e.args[0]
            args.split("\n")
                    
        arcpy.AddMessage("-------------- END EXECUTION ---------------")
        raise 
    except:
        tb = sys.exc_info()[2]
        tbinfo = traceback.format_tb(tb)[0]

        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
            str(sys.exc_info()) + "\n"
        msgs = "GP ERRORS:\n" + arcpy.GetMessages(2) + "\n"
        
        arcpy.AddError(msgs)
        arcpy.AddError(pymsg)

        raise
