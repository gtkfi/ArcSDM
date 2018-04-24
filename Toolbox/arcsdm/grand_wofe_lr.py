

#
# ArcSDM 5 for ArcGis pro
# Converted by Tero Ronkko, GTK 2017
# Updated by Arianne Ford, Kenex Ltd. 2018
#

"""Gets all valid weights tables for each evidence raster, generates all
    combinations of rasters and their tables, and runs each combination
    in Calculcate Response and Logistic Regression tools.  Produces
    probability, standard deviation, missing data variance, total variance,
    and confidence rasters.
"""
"""
    Spatial Data Modeller for ESRI* ArcGIS 9.2
    Copyright 2007
    Gary L Raines, Reno, NV, USA: production and certification
    Don L Sawatzky, Spokane, WA, USA: Python software development

    01/2018 Bug fixes for 10.x, allowing ascending (use a) and descending (use d) data types - Arianne Ford
    
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Grand_WofE_LR.py
# Created on: Fri Feb 29 2008 12:00:01 AM
# ---------------------------------------------------------------------------
"""

# Import system modules
import sys, os, traceback, arcgisscripting, string, operator,arcsdm
import arcpy

debuglevel = 0;
#Debug write
def dwrite(message):
    if (debuglevel > 0):
        arcpy.AddMessage("Debug: " + message) 

def execute(self, parameters, messages):


    # Create the Geoprocessor object
    gp = arcgisscripting.create()

    # Check out any necessary licenses
    gp.CheckOutExtension("spatial")

    gp.OverwriteOutput = 1
    gp.LogHistory = 1

    # Load required toolboxes...
    try:
        parentfolder = os.path.dirname(sys.path[0])
        #
        #tbxpath = os.path.join(parentfolder,"arcsdm.pyt")
        tbxpath = os.path.join(parentfolder,"toolbox\\arcsdm.pyt")
        dwrite(tbxpath)
        gp.AddToolbox(tbxpath)
        #gp.addmessage('getting arguments...')
        Grand_WOFE_Name = '_'+ parameters[0].valueAsText; #gp.GetParameterAsText(0)
        Evidence_Rasters = parameters[1].valueAsText.split(';'); #gp.GetParameterAsText(1).split(';')
        Evidence_Data_Types = parameters[2].valueAsText.lower().split(';'); #gp.GetParameterAsText(2).lower().split(';')
        Input_Training_Sites_Feature_Class = parameters[3].valueAsText; #gp.GetParameterAsText(3)
        Ignore_Missing_Data = parameters[4].value; #gp.GetParameter(4)
        Confidence_Level_of_Studentized_Contrast = parameters[5].value; #gp.GetParameter(5)
        Unit_Area__sq_km_ = parameters[6].value #gp.GetParameter(6)
        Missing_Data_Value = -99;
        #gp.addmessage('got arguments')
        #import SDMValues
        arcsdm.sdmvalues.appendSDMValues(gp, Unit_Area__sq_km_, Input_Training_Sites_Feature_Class)

        # Local variables...
        List_Wts_Tables = []
        suffixes = {'Ascending':'_CA','Descending':'_CD','Categorical':'_CT'}
        Missing_Data_Value = -99
        Evidence_Raster_Code_Field = ''
        OutSet = [] #List of output datasets
        dwrite('set local variables')

        #Processing...
        # Test for proper table data types:
        if len(Evidence_Data_Types) != len(Evidence_Rasters):
            gp.adderror('Number of evidence layers and weights data types do not match')
            raise
        for evtype in Evidence_Data_Types:       
            if not evtype[0] in 'ofcad':
                gp.adderror('Evidence data type %s not of %s'%(Evidence_Data_Type, ['free', 'categorical', 'ordered','ascending','descending']))
                raise TypeError         
        # Process: Calculate Weights of Evidence...
        dwrite(str(Evidence_Data_Types));
        dwrite(str(Evidence_Rasters));
        arcpy.AddMessage("========== Starting GrandWofe ====================" );
            
        for Evidence_Raster_Layer, Evidence_Data_Type in zip(Evidence_Rasters, Evidence_Data_Types):            
            prefix = Evidence_Raster_Layer + Grand_WOFE_Name
            arcpy.AddMessage("Calculating weights for %s (%s)..."%(Evidence_Raster_Layer,Evidence_Data_Type  ));
            if Evidence_Data_Type.startswith('o'):
                Wts_Table_Types = ['Ascending','Descending']
            elif Evidence_Data_Type.startswith('a'):
                Wts_Table_Types = ['Ascending']
            elif Evidence_Data_Type.startswith('d'):
                Wts_Table_Types = ['Descending']
            else: Wts_Table_Types = ['Categorical']
                
            for Wts_Table_Type in Wts_Table_Types:
                suffix = suffixes[Wts_Table_Type]
                filename = prefix + suffix; # + '.dbf' NO DBF anymore
                unique_name = gp.createuniquename(filename, gp.workspace)
                Output_Weights_Table = unique_name
                dwrite(gp.ValidateTablename(prefix + suffix) )
                
                dwrite('%s Exists: %s'%(Output_Weights_Table,gp.exists(Output_Weights_Table)))
                arcpy.ImportToolbox(tbxpath)
                
                # Temporarily print directory
                #gp.addmessage(dir(arcpy));
                #gp.addmessage("Calling calculate weights...")
                dwrite( " raster layer name: " + Evidence_Raster_Layer);
                
                result = arcpy.CalculateWeightsTool_ArcSDM ( Evidence_Raster_Layer, Evidence_Raster_Code_Field, \
                                               Input_Training_Sites_Feature_Class, Wts_Table_Type, Output_Weights_Table, \
                                               Confidence_Level_of_Studentized_Contrast, \
                                               Unit_Area__sq_km_, Missing_Data_Value)
                arcpy.AddMessage("     ...done");        
                gp.addwarning('Result: %s\n'%result)
                #gp.addmessage("Done...")
                #gp.addmessage(result);
                
                #Output, Success = result.split(';')
                Success = "True" # horrible fix...
                Output = result.getOutput(0) + ".dbf";
                if Success.strip().lower() == 'true':
                    List_Wts_Tables.append((Evidence_Raster_Layer, Output))
                    #gp.addmessage('Valid Wts Table: %s'%Output_Weights_Table)
                    OutSet.append(str(Output)) # Save name of output table for display kluge
                else:
                    gp.addmessage('Invalid Wts Table: %s'%Output.strip())
                #arcpy.AddMessage("\n")
                
        #Get list of valid tables for each input raster
        raster_tables = {}
        #arcpy.AddMessage("     ...done");
        
        for Evidence_Raster_Layer, Output_Weights_Table in List_Wts_Tables:
            #gp.addmessage(str((evidence_layer, wts_table)))
            if Evidence_Raster_Layer in raster_tables:
                raster_tables[Evidence_Raster_Layer].append(Output_Weights_Table)
            else:
                raster_tables[Evidence_Raster_Layer] = [Output_Weights_Table]
                
        if len(raster_tables) > 0:
            #Function to do nested "for" statements by recursion
            def nested_fors(ranges, tables, N, tables_out = [], tables_all = []):
                for n in ranges[0]:
                    tables_out.append(tables[0][n])
                    if len(ranges) > 1:
                        nested_fors(ranges[1:], tables[1:], N, tables_out, tables_all)
                    if len(tables_out) == N:
                        tables_all.append(tables_out[:])
                    del tables_out[-1]
                return tables_all

            #Get per-test lists of tables; each table in a list is in input raster order
            #tables = [raster_tables[Evidence_Raster_Layer] for Evidence_Raster_Layer in Evidence_Rasters]
            tables = []
            valid_rasters = []
            valid_raster_datatypes = []
            for Evidence_Raster_Layer, Evidence_Data_Type in zip(Evidence_Rasters, Evidence_Data_Types):
                if Evidence_Raster_Layer in raster_tables:
                    valid_rasters.append(Evidence_Raster_Layer)
                    valid_raster_datatypes.append(Evidence_Data_Type)
                    tables.append(raster_tables[Evidence_Raster_Layer])
            #Get ranges for number of tables for each evidence layer (in input evidence order)
            ranges = map(range,map(len, tables))
            #gp.addmessage(str(ranges))
            #Get combinations of valid wts table for input evidence_rasters
            Weights_Tables_Per_Test = nested_fors(ranges, tables, len(tables))
            for Testnum, Weights_Tables in enumerate(Weights_Tables_Per_Test):
                gp.addmessage("------ Running tests... (%s) ------" %(Testnum))
                # Process: Calculate Response...
                Test = Testnum
                dwrite (str(Weights_Tables));
                Weights_Tables =  ";".join(Weights_Tables)
                prefix = Grand_WOFE_Name[1:] + str(Test)
                gp.addMessage("%s: Response & Logistic Regression: %s,%s\n"%(Test, ";".join(valid_rasters), Weights_Tables))
                Output_Post_Prob_Raster = gp.createuniquename(prefix + "_pprb", gp.workspace)
                Output_Prob_Std_Dev_Raster = gp.createuniquename(prefix + "_pstd", gp.workspace)
                Output_MD_Variance_Raster = gp.createuniquename(prefix + "_mvar", gp.workspace)
                Output_Total_Std_Dev_Raster = gp.createuniquename(prefix + "_tstd", gp.workspace)
                Output_Confidence_Raster = gp.createuniquename(prefix + "_conf", gp.workspace)
                gp.AddToolbox(tbxpath)
                #dwrite (str(dir(arcpy)))
                gp.addMessage(" Calculating response... ");
                out_paths = arcpy.CalculateResponse_ArcSDM(";".join(valid_rasters), Weights_Tables, Input_Training_Sites_Feature_Class, \
                                 Ignore_Missing_Data, Missing_Data_Value, Unit_Area__sq_km_, Output_Post_Prob_Raster, Output_Prob_Std_Dev_Raster, \
                                 Output_MD_Variance_Raster, Output_Total_Std_Dev_Raster, Output_Confidence_Raster)
                # Set the actual output parameters
                gp.addMessage("       ...done");
                
                actualoutput = []
                dwrite (str(out_paths));
                dwrite ("Outputcount: " + str(out_paths.outputCount))
                dwrite ("Output0: " + str(out_paths.getOutput(0)))
                paths = "";
                for i in range(0,out_paths.outputCount):
                        dwrite ("Output: " + str(out_paths.getOutput(i)))               
                        paths = out_paths.getOutput(i) + ";"
                
                #for raspath in out_paths.split(';'):
                for raspath in paths.split(';'):
                    if gp.exists(raspath.strip()):
                        actualoutput.append(raspath.strip())
                out_paths = ';'.join(actualoutput)
                #Append delimited string to list
                OutSet.append(out_paths) # Save name of output raster dataset for kluge
                dwrite (" Outset: " + str(OutSet));
            
                # Process: Logistic Regression...
                Output_Polynomial_Table = gp.createuniquename(prefix + "_lrpoly.dbf", gp.workspace)
                Output_Coefficients_Table = gp.createuniquename(prefix + "_lrcoef.dbf", gp.workspace)
                Output_Post_Probability_raster = gp.createuniquename(prefix + "_lrpprb", gp.workspace)
                Output_Standard_Deviation_raster = gp.createuniquename(prefix + "_lrstd", gp.workspace)
                Output_LR_Confidence_raster = gp.createuniquename(prefix + "_lrconf", gp.workspace)
                #gp.AddToolbox(tbxpath)
                gp.addMessage(" Running logistic regression...");
                
                out_paths = arcpy.LogisticRegressionTool_ArcSDM(";".join(valid_rasters), ";".join(valid_raster_datatypes), Weights_Tables, Input_Training_Sites_Feature_Class,
                                 Missing_Data_Value, Unit_Area__sq_km_, Output_Polynomial_Table, Output_Coefficients_Table,
                                 Output_Post_Probability_raster, Output_Standard_Deviation_raster, Output_LR_Confidence_raster)
                dwrite(str(out_paths.status))                
                gp.addMessage("     ...done ");
                
                # Set the output parameters
                #Append delimited string to list
                for i in range(0,out_paths.outputCount):
                    dwrite ("Output: " + str(out_paths.getOutput(i)))               
                    OutSet.append(out_paths.getOutput(i))
                #OutSet.append(out_paths) # Save name of output raster dataset for kluge
                #Set output parameters
                #gp.addmessage("==== ====")
                dwrite(str(out_paths.status))
                    
            """Kluge because Geoprocessor can't handle variable number of ouputs"""
            dwrite (" Outset: " + str(OutSet));
            OutSet = ';'.join(OutSet)
            
            gp.addwarning("Copy the following line with ControlC,")
            gp.addwarning("then paste in the Name box of the Add Data command,")
            gp.addwarning("then click Add button")
            dwrite(str(OutSet));
            #stoppitalle();       
            gp.addwarning(OutSet)

        else:
            #Stop processing
            gp.AddError('No Valid Weights Tables: Stopped.')

    except:
        # get the traceback object
        tb = sys.exc_info()[2]
        # tbinfo contains the line number that the code failed on and the code from that line
        tbinfo = traceback.format_tb(tb)[0]
        # concatenate information together concerning the error into a message string
        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
            str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
        # generate a message string for any geoprocessing tool errors
        if len(gp.GetMessages(2)) > 0:
            msgs = "GP ERRORS:\n" + gp.GetMessages(2) + "\n"
            gp.AddError(msgs)

        # return gp messages for use with a script tool
        gp.AddError(pymsg)

        # print messages for use in Python/PythonWin
        print (msgs)
        raise
