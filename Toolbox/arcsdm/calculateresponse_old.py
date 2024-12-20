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
from arcsdm.missingdatavar_func import MissingDataVariance
from arcsdm.wofe_common import get_study_area_parameters


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

        gp.CheckOutExtension("spatial")

        Evidence = parameters[0].valueAsText # gp.GetParameterAsText(0)
        Wts_Tables = parameters[1].valueAsText # gp.GetParameterAsText(1)
        Training_Points = parameters[2].valueAsText # gp.GetParameterAsText(2)
        trainingDescr = arcpy.Describe(Training_Points)
        trainingCoord = trainingDescr.spatialReference.name
        IgnoreMsgData = parameters[3].value # gp.GetParameter(3)
        MissingDataValue = parameters[4].value # gp.GetParameter(4)
        
        if IgnoreMsgData: # for nodata argument to CopyRaster tool
            NoDataArg = MissingDataValue
        else:
            NoDataArg = '#'
        UnitArea = parameters[5].value # gp.GetParameter(5)

        log_arcsdm_details()
        total_area_sq_km_from_mask, numTPs = get_study_area_parameters(UnitArea, Training_Points)
        
        area_cell_count = total_area_sq_km_from_mask / UnitArea
        arcpy.AddMessage(("%-20s %s" % ("Study Area:", str(area_cell_count))))
        arcpy.AddMessage("%-20s %s" % ("# training points:", str(numTPs)))
        
        # Prior probability
        prior_probability = numTPs / area_cell_count
        arcpy.AddMessage("%-20s %s"% ("Prior probability:" , str(prior_probability)))

        # Get input evidence rasters
        Input_Rasters = Evidence.split(";")

        arcpy.AddMessage(f"Input rasters: {Input_Rasters}")
        
        # Get input weight tables
        Wts_Tables = Wts_Tables.split(";")

        arcpy.env.overwriteOutput = True
        arcpy.AddMessage("Setting overwriteOutput to True")

        arcpy.SetLogHistory(True)
        arcpy.AddMessage("Setting LogHistory to True")
        
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
        while i < len(Input_Rasters):
            input_raster = Input_Rasters[i]
            weights_table = Wts_Tables[i]

            # Check each Input Raster datatype and coordinate system
            inputDescr = arcpy.Describe(input_raster)
            inputCoord = inputDescr.spatialReference.name
            arcpy.AddMessage(f"{input_raster}, Data type: {inputDescr.datatype}, Coordinate System: {inputCoord}")
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
            std_raster_name = os.path.basename(input_raster[:9]).replace(".","_") + "S" # No . allowed in filegeodatgabases

            # If using GDB database, remove numbers and underscore from the beginning of the names
            if workspace_type != "FileSystem":
                while len(w_raster_name) > 0 and (w_raster_name[:1] <= "9" or w_raster_name[:1] == "_"):
                    w_raster_name = w_raster_name[1:]
                while len(std_raster_name) > 0 and (std_raster_name[:1] <= "9" or std_raster_name[:1] == "_"):
                    std_raster_name = std_raster_name[1:]

            # Create the _W & _S rasters that will be used to calculate output rasters
            output_w_raster = gp.CreateScratchName(w_raster_name, "", "rst", arcpy.env.scratchWorkspace)
            output_std_raster = gp.CreateScratchName(std_raster_name, "", "rst", arcpy.env.scratchWorkspace)
            
            # Increase the count for next round
            i += 1
     
            # Need to create in-memory Raster Layer for Join
            # Check for unsigned integer raster; cannot have negative missing data
            if NoDataArg != "#" and arcpy.Describe(input_raster).pixelType.upper().startswith("U"):
                NoDataArg2 = "#"
            else:
                NoDataArg2 = NoDataArg

            # Create new rasterlayer from input raster for both the weights and the std raster -> Result RasterLayer
            # These will be in-memory only
            # AddJoin requires an input layer or tableview not Input Raster Dataset.
            tmp_raster_layer = "OutRas_lyr"

            arcpy.MakeRasterLayer_management(input_raster, tmp_raster_layer)
            
            # Join result layer with weights table
            arcpy.AddJoin_management(tmp_raster_layer, "VALUE", weights_table, "CLASS")

            temp_raster = gp.CreateScratchName("tmp_rst", "", "rst", arcpy.env.scratchWorkspace)
            
            # Delete existing temp_raster
            if arcpy.Exists(temp_raster):
                arcpy.management.Delete(temp_raster)
                gc.collect()
                arcpy.ClearWorkspaceCache_management()
                arcpy.AddMessage("Deleted tempraster")
            
            # Copy created and joined in-memory raster to temp_raster
            arcpy.CopyRaster_management(tmp_raster_layer, temp_raster, "#", "#", NoDataArg2)

            arcpy.AddMessage(f"Output_Raster: {output_w_raster}")
            
            # Save weights raster
            weight_lookup = arcpy.sa.Lookup(temp_raster, "WEIGHT")
            weight_lookup.save(output_w_raster)
            
            if not arcpy.Exists(output_w_raster):
                arcpy.AddError(f"{output_w_raster} does not exist.")
                raise
            tmp_weights_rasters.append(output_w_raster)

            gp.Lookup_sa(temp_raster, "W_STD", output_std_raster)
            
            if not arcpy.Exists(output_std_raster):
                arcpy.AddError(f"{output_std_raster} does not exist.")
                raise
            tmp_std_rasters.append(output_std_raster)

            # Check for Missing Data in raster's Wts table
            if not IgnoreMsgData:
                # Update the list for Missing Data Variance Calculation
                tblrows = gp.SearchCursor(weights_table, "Class = %s" % MissingDataValue)
                tblrow = tblrows.Next()
                if tblrow:
                    rasters_with_missing_data.append(arcpy.Describe(output_w_raster).catalogPath)
            
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
            PostProb = parameters[6].valueAsText #gp.GetParameterAsText(6)
            
            InExpression = "Exp(%s) / (1.0 + Exp(%s))" % (InExpressionPLOG, InExpressionPLOG)
            arcpy.AddMessage(f"InExpression = {InExpression}")
            # Fix: This is obsolete
            #gp.MultiOutputMapAlgebra_sa(InExpression)
            arcpy.AddMessage(f"Postprob: {PostProb}")
            #output_raster = gp.RasterCalculator(InExpression, PostProb)
            #output_raster.save(postprob)
            gp.SingleOutputMapAlgebra_sa(InExpression, PostProb)
            # Pro/10 this needs to be done differently....
            #output_raster = arcpy.sa.RasterCalculator(InExpression, PostProb)
            #output_raster.save(postprob)
            
            arcpy.SetParameterAsText(6, PostProb)
        except:
            arcpy.AddError(arcpy.GetMessages(2))
            raise
        else:
            arcpy.AddWarning(arcpy.GetMessages(1))
            arcpy.AddMessage(arcpy.GetMessages(0))

        arcpy.AddMessage("\nCreating Post Probability STD Raster...\n" + "=" * 41)

        PostProb_Std = parameters[7].valueAsText # gp.GetParameterAsText(7)
        
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
       
            Constant = 1.0 / float(numTPs)

            InExpression = "SQRT(SQR(%s) * (%s + SUM(%s)))" % (PostProb, Constant, SUM_args)
            arcpy.AddMessage(f"InExpression = {InExpression}")

        try:
            arcpy.AddMessage(f"InExpression 2 ====> {InExpression}")

            gp.SingleOutputMapAlgebra_sa(InExpression, PostProb_Std)

            arcpy.SetParameterAsText(7, PostProb_Std)
        except:
            arcpy.AddError(arcpy.GetMessages(2))
            raise
        else:
            arcpy.AddWarning(arcpy.GetMessages(1))
            arcpy.AddMessage(arcpy.GetMessages(0))
        
        # Create Variance of missing data here and create totVar = VarMD + SQR(VarWts)
        if not IgnoreMsgData:
            # Calculate Missing Data Variance
            if len(rasters_with_missing_data) > 0:
                arcpy.AddMessage("Calculating Missing Data Variance...")
                try:
                    MDVariance = parameters[8].valueAsText # gp.GetParameterAsText(8)
                    if arcpy.Exists(MDVariance):
                        arcpy.management.Delete(MDVariance)

                    MissingDataVariance(gp, rasters_with_missing_data, PostProb, MDVariance)

                    Total_Std = parameters[9].valueAsText # gp.GetParameterAsText(9)
                    InExpression = 'SQRT(SUM(SQR(%s),%s))' % (PostProb_Std, MDVariance)

                    arcpy.AddMessage("Calculating Total STD...")
                    arcpy.AddMessage(f"InExpression 3 ====> {InExpression}")

                    gp.SingleOutputMapAlgebra_sa(InExpression, Total_Std)

                    arcpy.SetParameterAsText(9, Total_Std)
                except:
                    arcpy.AddError(arcpy.GetMessages(2))
                    raise
            else:
                arcpy.AddWarning("No evidence with missing data. Missing Data Variance not calculated.")
                MDVariance = None
                Total_Std = PostProb_Std
        else:
            arcpy.AddWarning("Missing Data Ignored. Missing Data Variance not calculated.")
            MDVariance = None
            Total_Std = PostProb_Std
        
        # Confidence is PP / sqrt(totVar)
        arcpy.AddMessage("\nCalculating Confidence...\n" + "=" * 41)

        Confidence = parameters[10].valueAsText # gp.GetParameterAsText(10)
        InExpression = "%s / %s" % (PostProb, PostProb_Std)
        arcpy.AddMessage(f"InExpression 4====> {InExpression}")
        try: 
            gp.SingleOutputMapAlgebra_sa(InExpression, Confidence)
            arcpy.SetParameterAsText(10, Confidence)
        except:
            arcpy.AddError(arcpy.GetMessages(2))
            raise
        else:
            arcpy.AddWarning(arcpy.GetMessages(1))
            arcpy.AddMessage(arcpy.GetMessages(0))
        
        # Set derived output parameters
        arcpy.SetParameterAsText(6, PostProb)
        arcpy.SetParameterAsText(7, PostProb_Std)

        if MDVariance and (not IgnoreMsgData):
            arcpy.SetParameterAsText(8, MDVariance)
        else:
            arcpy.AddWarning("No Missing Data Variance.")

        if not (Total_Std == PostProb_Std):
            arcpy.SetParameterAsText(9, Total_Std)
        else:
            arcpy.AddWarning("Total STD same as Post Probability STD.")

        arcpy.SetParameterAsText(10, Confidence)

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
