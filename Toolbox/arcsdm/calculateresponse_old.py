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

import arcpy
import gc
import importlib

from arcsdm.common import log_arcsdm_details
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
        import sys, os, math, traceback
        import arcsdm.sdmvalues
        import arcsdm.workarounds_93
        try:
            importlib.reload(arcsdm.sdmvalues)
            importlib.reload(arcsdm.workarounds_93)
        except:
            importlib.reload(arcsdm.sdmvalues)
            importlib.reload(arcsdm.workarounds_93)

        # Todo: Refactor to arcpy.
        import arcgisscripting
        gp = arcgisscripting.create()

        # Check out any necessary licenses
        gp.CheckOutExtension("spatial")
    
        ''' Parameters '''
        
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
        gp.AddMessage(("%-20s %s" % ("Study Area:", str(area_cell_count))))
        gp.AddMessage("%-20s %s" % ("# training points:", str(numTPs)))
        
        # Prior probability
        prior_probability = numTPs / area_cell_count
        arcpy.AddMessage("%-20s %s"% ("Prior probability:" , str(prior_probability)))

        # Get input evidence rasters
        Input_Rasters = Evidence.split(";")

        gp.AddMessage("Input rasters: " + str(Input_Rasters))
        
        # Get input weight tables
        Wts_Tables = Wts_Tables.split(";")
        
        # Create weight raster from raster's associated weights table
        gp.OverwriteOutput = 1
        gp.LogHistory = 1
        tmp_weights_rasters = []
        tmp_std_rasters = []
        # Create a list for the Missing Data Variance tool.
        rasters_with_missing_data = []
        # Evidence rasters should have missing data values, where necessary, for
        # NoData cell values within study area.
        # For each input_raster create a weights raster from the raster and its weights table.

        wsdesc = arcpy.Describe(gp.workspace)

        ''' Weight rasters '''
        
        # gp.AddMessage("\nCreating weight rasters ")
        gp.AddMessage("\nCreating tmp weight and STD rasters...")
        arcpy.AddMessage("=" * 41)

        i = 0
        while i < len(Input_Rasters):
            Input_Raster = Input_Rasters[i]
            Wts_Table = Wts_Tables[i]

            # Check each Input Raster datatype and coordinate system
            inputDescr = arcpy.Describe(Input_Raster)
            inputCoord = inputDescr.spatialReference.name
            arcpy.AddMessage(Input_Raster + ", Data type: " + inputDescr.datatype + ", Coordinate System: " + inputCoord)
            if inputCoord != trainingCoord:
                arcpy.AddError("ERROR: Coordinate System of Input Raster is " + inputCoord + " and Training points it is " + trainingCoord + ". These must be same.")
                raise

            # When workspace type is File System, Input Weight Table also must end with .dbf
            # If using GDB database, remove numbers and underscore from the beginning of the name (else block)
            if wsdesc.workspaceType == "FileSystem":
                if not(Wts_Table.endswith('.dbf')):
                    Wts_Table += ".dbf"
            else:
                wtsbase = os.path.basename(Wts_Table)
                while len(wtsbase) > 0 and (wtsbase[:1] <= "9" or wtsbase[:1] == "_"):
                    wtsbase = wtsbase[1:]
                Wts_Table = os.path.dirname(Wts_Table) + "\\" + wtsbase

            # Compare workspaces to make sure they match
            
            arcpy.AddMessage("Processing " + Input_Raster)
            
            outputrastername = Input_Raster.replace(".", "_")

            # If using GDB database, remove numbers and underscore from the beginning of the outputrastername
            outputrastername = outputrastername[:10] + "W"
            if wsdesc.workspaceType != "FileSystem":
                while len(outputrastername) > 0 and (outputrastername[:1] <= "9" or outputrastername[:1] == "_"):
                    outputrastername = outputrastername[1:]

            # Needs to be able to extract input raster name from full path.
            # Can't assume only a layer from ArcMap.
            stdoutputrastername = os.path.basename(Input_Raster[:9]).replace(".","_") + "S" # No . allowed in filegeodatgabases
            # If using GDB database, remove numbers and underscore from the beginning of the name (else block)
            if (wsdesc.workspaceType != "FileSystem"):
                while len(stdoutputrastername) > 0 and (stdoutputrastername[:1] <= "9" or stdoutputrastername[:1] == "_"):
                    stdoutputrastername = stdoutputrastername[1:]

            # Create _W raster
            Output_Raster = gp.CreateScratchName(outputrastername, '', 'rst', gp.scratchworkspace)

            # Create _S raster
            Output_Std_Raster = gp.CreateScratchName(stdoutputrastername, '', 'rst', gp.scratchworkspace)
            
            # Increase the count for next round
            i += 1
            
            dwrite("WtsTable is " + Wts_Table)
     
            # Need to create in-memory Raster Layer for Join
            # Check for unsigned integer raster; cannot have negative missing data
            if NoDataArg != '#' and gp.describe(Input_Raster).pixeltype.upper().startswith('U'):
                NoDataArg2 = '#'
            else:
                NoDataArg2 = NoDataArg

            # Create new rasterlayer from input raster for both the weights and the std raster -> Result RasterLayer
            RasterLayer = "OutRas_lyr"
            RasterLayer2 = "OutRas_lyr2"

            arcpy.MakeRasterLayer_management(Input_Raster, RasterLayer)
            gp.makerasterlayer(Input_Raster, RasterLayer2)

            # AddJoin requires and input layer or tableview not Input Raster Dataset.
            # Join result layer with weights table
            dwrite("Layer and Rasterlayer: " + Input_Raster + ", " + RasterLayer)

            arcpy.AddJoin_management(RasterLayer, "VALUE", Wts_Table, "CLASS")
            gp.AddJoin_management(RasterLayer2, "Value", Wts_Table, "CLASS")

            Temp_Raster = gp.CreateScratchName('tmp_rst', '', 'rst', gp.scratchworkspace)
            Temp_Std_Raster = gp.CreateScratchName('tmp_rst', '', 'rst', gp.scratchworkspace)
            dwrite("Temp_Raster=" + Temp_Raster)
            dwrite("Temp_Std_Raster=" + Temp_Raster)
            dwrite("Wts_Table=" + Wts_Table)
            
            # Delete old temp_raster
            if gp.exists(Temp_Raster):
                arcpy.Delete_management(Temp_Raster)
                gc.collect()
                arcpy.ClearWorkspaceCache_management()
                gp.AddMessage("Deleted tempraster")

            if gp.exists(Temp_Std_Raster): 
                arcpy.Delete_management(Temp_Std_Raster)
                gc.collect()
                arcpy.ClearWorkspaceCache_management()
                gp.AddMessage("Tmprst deleted.")
            
            # Copy created and joined raster to temp_raster
            arcpy.CopyRaster_management(RasterLayer, Temp_Raster, '#', '#', NoDataArg2)
            arcpy.CopyRaster_management(RasterLayer2, Temp_Std_Raster, "#", "#", NoDataArg2)

            arcpy.AddMessage("Output_Raster: " + Output_Raster)
            
            outras = arcpy.sa.Lookup(Temp_Raster, "WEIGHT")
            outras.save(Output_Raster)
            
            if not gp.Exists(Output_Raster):
                gp.AddError( " " + Output_Raster + " does not exist.")
                raise
            tmp_weights_rasters.append(Output_Raster)

            gp.Lookup_sa(Temp_Std_Raster, "W_STD", Output_Std_Raster)
            
            if not gp.Exists(Output_Std_Raster):
                gp.AddError(Output_Std_Raster + " does not exist.")
                raise
            # Output_Raster = gp.Describe(Output_Std_Raster).CatalogPath
            tmp_std_rasters.append(Output_Std_Raster)

            # Check for Missing Data in raster's Wts table
            if not IgnoreMsgData:
                # Update the list for Missing Data Variance Calculation
                tblrows = gp.SearchCursor(Wts_Table, "Class = %s" % MissingDataValue)
                tblrow = tblrows.Next()
                if tblrow:
                    rasters_with_missing_data.append(gp.Describe(Output_Raster).CatalogPath)
            arcpy.AddMessage(" ") # Cycle done - add ONE linefeed
        
        # Get Post Logit Raster
        
        gp.AddMessage("\nGetting Post Logit raster...\n" + "=" * 41)
        
        # This used to be comma separated, now +
        Input_Data_Str = ' + '.join('"{0}"'.format(w) for w in tmp_weights_rasters)
        arcpy.AddMessage("Input_data_str: " + Input_Data_Str)
        Constant = math.log(prior_probability / (1.0 - prior_probability))
        
        if len(tmp_weights_rasters) == 1:
            InExpressionPLOG = "%s + %s" % (Constant, Input_Data_Str)
        else:
            InExpressionPLOG = "%s + (%s)" % (Constant, Input_Data_Str)
        gp.AddMessage("InexpressionPlog: " + InExpressionPLOG)

        # Get Post Probability Raster
        
        gp.AddMessage("\nCreating Post Probability Raster...\n" + "=" * 41)
        try:
            #pass
            #PostLogitRL = os.path.join(gp.Workspace, "PostLogitRL")
            #gp.MakeRasterLayer_management(PostLogit, PostLogitRL)
            #InExpression = "EXP(%s) / ( 1.0 + EXP(%s))" %(PostLogitRL, PostLogitRL)
            PostProb = parameters[6].valueAsText #gp.GetParameterAsText(6)
            
            InExpression = "Exp(%s) / (1.0 + Exp(%s))" % (InExpressionPLOG, InExpressionPLOG)
            gp.AddMessage("InExpression = " + str(InExpression))
            # Fix: This is obsolete
            #gp.MultiOutputMapAlgebra_sa(InExpression)
            gp.AddMessage("Postprob: " + PostProb)
            #output_raster = gp.RasterCalculator(InExpression, PostProb)
            #output_raster.save(postprob)
            gp.SingleOutputMapAlgebra_sa(InExpression, PostProb)
            # Pro/10 this needs to be done differently....
            #output_raster = arcpy.sa.RasterCalculator(InExpression, PostProb)
            #output_raster.save(postprob)
            
            gp.SetParameterAsText(6, PostProb)
        except:
            gp.AddError(gp.getMessages(2))
            raise
        else:
            gp.AddWarning(gp.getMessages(1))
            gp.AddMessage(gp.getMessages(0))

        # Create STD raster from raster's associated weights table
        
        gp.AddMessage("\nCreating Post Probability STD Raster...\n" + "=" * 41)
        PostProb_Std = parameters[7].valueAsText # gp.GetParameterAsText(7)
        
        #TODO: Figure out what this does!? TR
        #TODO: This is always false now
        if len(tmp_std_rasters) == 1: # If there is only one input... ??? TR 
            InExpression = '"%s"' % (tmp_std_rasters[0])
        else:
            SUM_args_list = []
            for Std_Raster in tmp_std_rasters:
                SUM_args_list.append("SQR(\"%s\")" % Std_Raster)
            SUM_args = " + ".join(SUM_args_list)
            gp.AddMessage("Sum_args: " + SUM_args + "\n" + "=" * 41)
       
            Constant = 1.0 / float(numTPs)

            InExpression = "SQRT(SQR(%s) * (%s + SUM(%s)))" % (PostProb, Constant, SUM_args)
            gp.AddMessage("InExpression = " + str(InExpression))

        try:
            gp.addmessage("InExpression 2 ====> " + InExpression)
            # gp.MultiOutputMapAlgebra_sa(InExpression)
            # output_raster = gp.RasterCalculator(InExpression, PostProb_Std)
            gp.SingleOutputMapAlgebra_sa(InExpression, PostProb_Std)
            gp.SetParameterAsText(7, PostProb_Std)
        except:
            gp.AddError(gp.getMessages(2))
            raise
        else:
            gp.AddWarning(gp.getMessages(1))
            gp.AddMessage(gp.getMessages(0))
        
        # Create Variance of missing data here and create totVar = VarMD + SQR(VarWts)
        if not IgnoreMsgData:
            # Calculate Missing Data Variance
            if len(rasters_with_missing_data) > 0:
                import arcsdm.missingdatavar_func
                gp.AddMessage("Calculating Missing Data Variance...")
                try:
                    MDVariance = parameters[8].valueAsText # gp.GetParameterAsText(8)
                    if gp.exists(MDVariance):
                        arcpy.Delete_management(MDVariance)

                    arcsdm.missingdatavar_func.MissingDataVariance(gp, rasters_with_missing_data, PostProb, MDVariance)
                    Total_Std = parameters[9].valueAsText # gp.GetParameterAsText(9)
                    InExpression = 'SQRT(SUM(SQR(%s),%s))' % (PostProb_Std, MDVariance)

                    gp.AddMessage("Calculating Total STD...")
                    gp.addmessage("InExpression 3 ====> " + InExpression)

                    gp.SingleOutputMapAlgebra_sa(InExpression, Total_Std)
                    gp.SetParameterAsText(9, Total_Std)
                except:
                    gp.AddError(gp.getMessages(2))
                    raise
            else:
                gp.AddWarning("No evidence with missing data. Missing Data Variance not calculated.")
                MDVariance = None
                Total_Std = PostProb_Std
        else:
            gp.AddWarning("Missing Data Ignored. Missing Data Variance not calculated.")
            MDVariance = None
            Total_Std = PostProb_Std
        
        # Confidence is PP / sqrt(totVar)
        gp.AddMessage("\nCalculating Confidence...\n" + "=" * 41)

        Confidence = parameters[10].valueAsText # gp.GetParameterAsText(10)
        InExpression = "%s / %s" % (PostProb, PostProb_Std)
        gp.AddMessage("InExpression 4====> " + InExpression)
        try: 
            gp.SingleOutputMapAlgebra_sa(InExpression, Confidence)
            gp.SetParameterAsText(10, Confidence)
        except:
            gp.AddError(gp.getMessages(2))
            raise
        else:
            gp.AddWarning(gp.getMessages(1))
            gp.AddMessage(gp.getMessages(0))
        
        # Set derived output parameters
        gp.SetParameterAsText(6, PostProb)
        gp.SetParameterAsText(7, PostProb_Std)
        if MDVariance and (not IgnoreMsgData):
            gp.SetParameterAsText(8, MDVariance)
        else:
            gp.AddWarning('No Missing Data Variance.')
        if not (Total_Std == PostProb_Std):
            gp.SetParameterAsText(9, Total_Std)
        else:
            gp.AddWarning('Total STD same as Post Probability STD.')

        gp.SetParameterAsText(10, Confidence)
        gp.AddMessage("done\n" + "=" * 41)
    except arcpy.ExecuteError as e:
        arcpy.AddError("\n")
        arcpy.AddMessage("Calculate Response caught arcpy.ExecuteError ")
        gp.AddError(arcpy.GetMessages())
        if len(e.args) > 0:
            args = e.args[0]
            args.split('\n')
                    
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

        print(pymsg)
        print(msgs)

        raise
