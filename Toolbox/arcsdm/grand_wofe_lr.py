"""
    ArcSDM 6 ToolBox for ArcGIS Pro

    Conversion and tool development for ArcGIS Pro by Geological Survey of Finland (GTK), 2024.

    Grand WofE

    Gets all valid weights tables for each evidence raster, generates all
    combinations of rasters and their tables, and runs each combination
    in Calculcate Response and Logistic Regression tools.  Produces
    probability, standard deviation, missing data variance, total variance,
    and confidence rasters.

    ArcSDM 5 for ArcGis pro
    Converted by Tero Ronkko, GTK 2017
    Updated by Arianne Ford, Kenex Ltd. 2018
    Updated by Arto Laiho, Geological survey of Finland 4.5-12.6.2020:
    - "Invalid Wts Table" changed from message to warning.
    - sys.exc_type and exc_value are deprecated, replaced by sys.exc_info()
    - Grand Wofe Name cannot be longer than 7 characters
    - Weights table prefix changed
    Logistic Regression don't work on ArcGIS Pro 2.5 with File System workspace but works on V2.6 /AL 140820
    If using GDB database, remove numbers and underscore from the beginning of Weights Table name /AL 061020


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

from arcsdm.common import log_arcsdm_details
from arcsdm.wofe_common import get_study_area_parameters



#Set to one while debugging (Or add DEBUG file to arcsdm directory)
#Debug write
debuglevel = 0;

def  testdebugfile():
    returnvalue = 0;
    import sys;
    import os;
    if (debuglevel > 0):
        returnvalue = 1;
    else:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if (os.path.isfile(dir_path + "/DEBUG")):
            returnvalue =  1;            
    return returnvalue;

def dwrite(message):
    debug = testdebugfile();
    if (debuglevel > 0 or debug > 0):
        arcpy.AddMessage("Debug: " + message)



def execute(self, parameters, messages):


    
    # Create the Geoprocessor object
    gp = arcgisscripting.create()

    # Check out any necessary licenses
    gp.CheckOutExtension("spatial")

    gp.OverwriteOutput = 1
    gp.LogHistory = 1

    # Logistic Regression don't work on ArcGIS Pro 2.5 when workspace is File System but works on V2.6! #AL 140820
    desc = arcpy.Describe(gp.workspace)
    install_version=str(arcpy.GetInstallInfo()['Version'])
    if str(arcpy.GetInstallInfo()['ProductName']) == "ArcGISPro" and install_version <= "2.5"  and desc.workspaceType == "FileSystem":
        arcpy.AddError ("ERROR: Logistic Regression don't work on ArcGIS Pro " + install_version + " when workspace is File System!")
        raise ValueError()

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
        # Grand Wofe Name cannot be longer than 7 characters #AL 090620
        if (len(Grand_WOFE_Name) > 7):
            arcpy.AddError("ERROR: Grand Wofe Name cannot be longer than 7 characters.")
            raise ValueError()
        Evidence_Rasters = parameters[1].valueAsText.split(';'); #gp.GetParameterAsText(1).split(';')
        Evidence_Data_Types = parameters[2].valueAsText.lower().split(';'); #gp.GetParameterAsText(2).lower().split(';')
        Input_Training_Sites_Feature_Class = parameters[3].valueAsText; #gp.GetParameterAsText(3)
        trainingDescr = arcpy.Describe(Input_Training_Sites_Feature_Class) #AL 180520
        trainingCoord = trainingDescr.spatialReference.name                #AL 180520
        Ignore_Missing_Data = parameters[4].value; #gp.GetParameter(4)
        Confidence_Level_of_Studentized_Contrast = parameters[5].value; #gp.GetParameter(5)
        Unit_Area__sq_km_ = parameters[6].value #gp.GetParameter(6)
        Missing_Data_Value = -99;

        log_arcsdm_details()
        _, _ = get_study_area_parameters(Unit_Area__sq_km_, Input_Training_Sites_Feature_Class)

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
            raise ValueError()
        for evtype in Evidence_Data_Types:       
            if not evtype[0] in 'ofcad':
                arcpy.AddError('ERROR: Evidence data type %s not of %s'%(Evidence_Data_Type, ['free', 'categorical', 'ordered','ascending','descending']))
                raise TypeError()         
        # Process: Calculate Weights of Evidence...
        dwrite(str(Evidence_Data_Types));
        dwrite(str(Evidence_Rasters));
        arcpy.AddMessage("========== Starting GrandWofe ====================" );
            
        for Evidence_Raster_Layer, Evidence_Data_Type in zip(Evidence_Rasters, Evidence_Data_Types):
            # Check Evidence Raster datatype and Coordinate System #AL 180520 
            evidenceDescr = arcpy.Describe(Evidence_Raster_Layer) 
            evidenceCoord = evidenceDescr.spatialReference.name
            arcpy.AddMessage("Data type of Evidence Layer " + Evidence_Raster_Layer + " is " + evidenceDescr.datatype + " and Coordinate System " + evidenceCoord)
            if (evidenceCoord != trainingCoord):
                arcpy.AddError("ERROR: Coordinate System of Evidence Layer is " + evidenceCoord + " and Training points it is " + trainingCoord + ". These must be same.")
                raise

            splitted_evidence = os.path.split(Evidence_Raster_Layer)   #AL 090620
            eviname = os.path.splitext(splitted_evidence[1])           #AL 090620
            #prefix = Evidence_Raster_Layer + Grand_WOFE_Name
            prefix = gp.workspace + "\\" + eviname[0] + Grand_WOFE_Name	#AL 090620
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
                desc = arcpy.Describe(gp.workspace)

                # If using non gdb database, lets add .dbf
                # If using GDB database, remove numbers and underscore from the beginning of the name (else block) #AL 061020                
                if  desc.workspaceType == "FileSystem":
                    if not(filename.endswith('.dbf')):
                        filename = filename + ".dbf";
                    dwrite ("Filename is a file - adding dbf")
                else:
                    wtsbase = os.path.basename(filename)
                    while len(wtsbase) > 0 and (wtsbase[:1] <= "9" or wtsbase[:1] == "_"):
                        wtsbase = wtsbase[1:]
                    filename = os.path.dirname(filename) + "\\" + wtsbase
                
                unique_name = gp.createuniquename(filename, gp.workspace)
                Output_Weights_Table = unique_name
                #dwrite("Validate: " + gp.ValidateTablename(prefix + suffix) )
                
                arcpy.ImportToolbox(tbxpath)
                
                # Temporarily print directory
                #gp.addmessage(dir(arcpy));
                gp.addmessage("Calling calculate weights...")
                gp.addmessage(' Output table name: %s Exists already: %s'%(Output_Weights_Table,gp.exists(Output_Weights_Table)))

                result = arcpy.CalculateWeightsTool_ArcSDM ( Evidence_Raster_Layer, Evidence_Raster_Code_Field, \
                                               Input_Training_Sites_Feature_Class, Wts_Table_Type, Output_Weights_Table, \
                                               Confidence_Level_of_Studentized_Contrast, \
                                               Unit_Area__sq_km_, Missing_Data_Value)
                arcpy.AddMessage("     ...done");        
                gp.AddMessage('Result: %s\n'%result)
                #gp.addmessage("Done...")
                #gp.addmessage(result);
                
                #Output, Success = result.split(';')
                Success = "True" # horrible fix...
                outputfilename = result.getOutput(0);
                tmp = result.getOutput(1);
                dwrite ("Result: " + str(tmp));
                warning = result.getMessages(1);
                
                dwrite(warning);
                if(len(warning)>0):
                    arcpy.AddWarning(warning);
                    #Success = "False"; #AL 180520 removed
                    #Should stop here?
              
                
                
                #TODO: filegeodatabase support! No .dbf there.
                #dbf file-extension fix
                # Testing workspace
                
                Output = outputfilename;
                
                
                if not(outputfilename.endswith('.dbf')):
                    Output = outputfilename #+ ".dbf";
                    #Geodatabase....
                if  desc.workspaceType == "FileSystem":
                    if not(outputfilename.endswith('.dbf')):
                        Output = outputfilename + ".dbf";
                    dwrite ("Workspace is filesystem - adding dbf")
                    
            
                if Success.strip().lower() == 'true':
                    List_Wts_Tables.append((Evidence_Raster_Layer, Output))
                    #gp.addmessage('Valid Wts Table: %s'%Output_Weights_Table)
                    OutSet.append(str(Output)) # Save name of output table for display kluge
                else:
                    #gp.addmessage('Invalid Wts Table: %s'%Output.strip())
                    gp.AddWarning('Invalid Wts Table: %s'%Output.strip())     #AL 040520
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
            # Py 3.4 fixes here:
            ranges = list(map(range,list(map(len, tables))))
            dwrite("Tables: " + str(tables));
            dwrite("Ranges: " + str(ranges))
            #ranges = map(range,map(len, tables))
            #gp.addmessage(str(ranges))
            #Get combinations of valid wts table for input evidence_rasters
            Weights_Tables_Per_Test = nested_fors(ranges, tables, len(tables))
            for Testnum, Weights_Tables in enumerate(Weights_Tables_Per_Test):
                Test = Testnum + 1
                gp.addmessage("------ Running tests... (%s) ------" %(Test))
                # Process: Calculate Response...
                dwrite ("Weight tables: " + str(Weights_Tables));
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
                dwrite ("valid_rasters = " + str(valid_rasters))
                dwrite ("valid_raster_datatypes = " + str(valid_raster_datatypes))
                dwrite ("0 Input Raster Layer(s) (GPValueTable: GPRasterLayer) = ';'.join(valid_rasters)")
                dwrite ("1 Evidence type (GPValueTable: GPString) = ';'.join(valid_raster_datatypes)")
                dwrite ("w Input weights tables (GPValueTable: DETable) = " + str(Weights_Tables))
                dwrite ("2 Training sites (GPFeatureLayer) = " + str(Input_Training_Sites_Feature_Class))
                dwrite ("3 Missing data value (GPLong) = " + str(Missing_Data_Value))
                dwrite ("4 Unit area (km^2) (GPDouble) = " + str(Unit_Area__sq_km_))
                dwrite ("5 Output polynomial table (DEDbaseTable) = " + str(Output_Polynomial_Table))
                dwrite ("52 Output coefficients table (DEDbaseTable) = " + str(Output_Coefficients_Table))
                dwrite ("6 Output post probablity raster (DERasterDataset) = " + str(Output_Post_Probability_raster))
                dwrite ("62 Output standard deviation raster (DERasterDataset) = " + str(Output_Standard_Deviation_raster))
                dwrite ("63 Output confidence raster (DERasterDataset) = " + str(Output_LR_Confidence_raster))

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

    except ValueError():
        raise
    except:
        # get the traceback object
        tb = sys.exc_info()[2]
        #e = sys.exc_info()[1]
        #dwrite(e.args[0])
    
        # If using this code within a script tool, AddError can be used to return messages 
        #   back to a script tool.  If not, AddError will have no effect.
        #arcpy.AddError(e.args[0])
        ## Begin old.
        
        
        # tbinfo contains the line number that the code failed on and the code from that line
        tbinfo = traceback.format_tb(tb)[0]
        # concatenate information together concerning the error into a message string
        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
            str(sys.exc_info()) + "\n"    #AL 040520
        #    str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
        # generate a message string for any geoprocessing tool errors
        if len(gp.GetMessages(2)) > 0:
            msgs = "GP ERRORS:\n" + gp.GetMessages(2) + "\n"
            gp.AddError(msgs)
            gp.AddMessage(msgs)

        # return gp messages for use with a script tool
        gp.AddError(pymsg)

        raise
