"""

    ArcSDM 5 - ArcSDM for ArcGis pro

    History:
    4/2016 Conversion started - TR
    9/2016 Conversion started to Python toolbox TR
    01/2018 Bug fixes for 10.x - Arianne Ford
    27.4.2020 added Input Weights table file type checking / Arto Laiho, GTK/GFS
    18.5.2020 added Input Raster coordinate system checking / Arto Laiho, GTK/GFS
    15.6.2020 added "arcsdm." to import missingdatavar_func / Arto Laiho, GTK/GFS
    20-23.7.2020 combined with Unicamp fixes (made 19.7.2018) / Arto Laiho, GTK/GFS

    Spatial Data Modeller for ESRI* ArcGIS 9.2
    Copyright 2007
    Gary L Raines, Reno, NV, USA: production and certification
    Don L Sawatzky, Spokane, WA, USA: Python software development
    
# ---------------------------------------------------------------------------
# NewCalcResp3.py
#  Modifications for use of Lookup_sa by Ryan Bryn, ESRI
# ---------------------------------------------------------------------------
"""

import arcsdm.sdmvalues;
import arcpy;
import gc;
import importlib;
import sys

debuglevel = 0;
#Debug write

def testdebugfile():
    returnvalue = 0; #This because python sucks in detecting outputs from functions
    import sys;
    import os;
    if (debuglevel > 0):
        return 1;
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if (os.path.isfile(dir_path + "/DEBUG")):
        return 1;            
    return returnvalue;

def dwrite(message):
    debug = testdebugfile();
    if (debuglevel > 0 or debug > 0):
        arcpy.AddMessage("Debug: " + message)
        #arcpy.AddError("DebugError:" + message)


def Execute(self, parameters, messages):
    
    try:
        # Import system modules
        import sys, os, math, traceback;
        import arcsdm.sdmvalues;
        import arcsdm.workarounds_93;
        try:
            importlib.reload (arcsdm.sdmvalues)
            importlib.reload (arcsdm.workarounds_93);
        except :
            reload(arcsdm.sdmvalues);
            reload(arcsdm.workarounds_93);        
        # Create the Geoprocessor object
        #Todo: Refactor to arcpy.
        import arcgisscripting
        gp = arcgisscripting.create()

        # Check out any necessary licenses
        gp.CheckOutExtension("spatial")
    
        ''' Parameters '''
        
        Evidence = parameters[0].valueAsText #gp.GetParameterAsText(0)
        Wts_Tables = parameters[1].valueAsText #gp.GetParameterAsText(1)
        Training_Points = parameters[2].valueAsText #gp.GetParameterAsText(2)
        trainingDescr = arcpy.Describe(Training_Points)     #AL 180520
        trainingCoord = trainingDescr.spatialReference.name #AL 180520
        IgnoreMsgData = parameters[3].value #gp.GetParameter(3)
        MissingDataValue = parameters[4].value #gp.GetParameter(4)
        #Cleanup extramessages after stuff
        #gp.AddMessage('Got arguments' )
        #arcsdm.common.testandwarn_arcgispro();
        
        if IgnoreMsgData: # for nodata argument to CopyRaster tool
            NoDataArg = MissingDataValue
        else:
            NoDataArg = '#'
        UnitArea = parameters[5].value #gp.GetParameter(5)
        
        arcsdm.sdmvalues.appendSDMValues(gp, UnitArea, Training_Points)
    # Local variables...

        
    #Getting Study Area in counts and sq. kilometers
   
        Counts = arcsdm.sdmvalues.getMaskSize(arcsdm.sdmvalues.getMapUnits(True));
        gp.AddMessage("\n"+"="*21+" Starting calculate response "+"="*21)
        #gp.AddMessage(str(gp.CellSize))
        #CellSize = float(gp.CellSize)
        Study_Area = Counts / UnitArea # getMaskSize returns mask size in sqkm now - TODO: WHy is this divided with UnitArea? (Counts * CellSize * CellSize / 1000000.0) / UnitArea
        gp.AddMessage( ("%-20s %s" % ("Study Area:",  str(Study_Area))))

        #Get number of training points
        numTPs = gp.GetCount_management(Training_Points)
        gp.AddMessage("%-20s %s" % ("# training points:", str(numTPs)))
        
        #Prior probability
        Prior_prob = float(numTPs) / Study_Area 
        gp.AddMessage("%-20s %s"% ("Prior_prob:" , str(Prior_prob)))

        #Get input evidence rasters
        #dwrite ("Evidence = " + Evidence)
        Input_Rasters = Evidence.split(";")
        
        # Process things and removve grouplayer names including EXTRA ' ' symbols around spaced grouplayer name
        gp.AddMessage("Input rasters: " + str(Input_Rasters))
        # These lines causing BUG. Commented because these are not nocessary (Unicamp 190718 / AL 200720) 
        #for i, s in enumerate(Input_Rasters):
        #    Input_Rasters[i] = s.strip("'"); #arcpy.Describe( s.strip("'")).file;
        gp.AddMessage("Input rasters: " + str(Input_Rasters))
        
        #Get input weight tables
        Wts_Tables = Wts_Tables.split(";")
        #dwrite ("Wts_Tables = " + str(Wts_Tables))
        
        #Create weight raster from raster's associated weights table
        #dwrite ("Getting Weights rasters...")
        gp.OverwriteOutput = 1
        gp.LogHistory = 1
        Wts_Rasters = []
        i = 0
        # Create a list for the Missing Data Variance tool.
        rasterList = []
        # Evidence rasters should have missing data values, where necessary, for
        # NoData cell values within study area.
        # For each input_raster create a weights raster from the raster and its weights table.
        mdidx = 0

        # Method selection: 0 = ArcGIS Pro & FileSystem, 1 = all other #AL 220720       
        wsdesc = arcpy.Describe(gp.workspace) #AL 030620
        method = 1
        if str(arcpy.GetInstallInfo()['ProductName']) == "ArcGISPro" and wsdesc.workspaceType == "FileSystem": method = 0;
        arcpy.AddMessage("Method = " + str(method))

        ''' Weight rasters '''
        
        gp.AddMessage("\nCreating weight rasters ")
        arcpy.AddMessage("=" * 41);
        for Input_Raster in Input_Rasters:
            # Check each Input Raster datatype and coordinate system #AL 180520,030620
            inputDescr = arcpy.Describe(Input_Raster)
            inputCoord = inputDescr.spatialReference.name
            arcpy.AddMessage(Input_Raster + ", Data type: " + inputDescr.datatype + ", Coordinate System: " + inputCoord)
            if (inputCoord != trainingCoord):
                arcpy.AddError("ERROR: Coordinate System of Input Raster is " + inputCoord + " and Training points it is " + trainingCoord + ". These must be same.")
                raise

            #<== RDB
            #++ Needs to be able to extract input raster name from full path.
            #++ Can't assume only a layer from ArcMap.
    ##        Output_Raster = os.path.join(gp.ScratchWorkspace,Input_Raster[:11] + "_W")  
            ##Output_Raster = os.path.basename(Input_Raster)[:11] + "_W"
            #outputrastername = (Input_Raster[:9]) + "_W"; 
            #TODO: Do we need to consider if the file names collide with shapes? We got collision with featureclasses
            #desc = arcpy.Describe(Input_Raster); #AL 160620 unnecessary

            Wts_Table = Wts_Tables[i]           

            # When workspace type is File System, Input Weight Table also must end with .dbf #AL
            if (wsdesc.workspaceType == "FileSystem"):
                if not(Wts_Table.endswith('.dbf')):
                    Wts_Table += ".dbf";   #AL 060520
            
            #Compare workspaces to make sure they match
            #desc2 = arcpy.Describe(Wts_Table); #AL removed (unnecessary) 220720
            
            arcpy.AddMessage("Processing " + Input_Raster);
            
            outputrastername = Input_Raster.replace(".","_");
            
            outputrastername = outputrastername[:10] + "W";
            #outputrastername = desc.nameString + "_W2";
            # Create _W raster
            Output_Raster = gp.CreateScratchName(outputrastername, '', 'rst', gp.scratchworkspace)
            
            #Increase the count for next round
            i += 1
            
            dwrite("WtsTable is " + Wts_Table);
            #Wts_Table = gp.Describe(Wts_Table).CatalogPath
            ## >>>>> Section replaced by join and lookup below >>>>>
            ##        try:
            ##            gp.CreateRaster_sdm(Input_Raster, Wts_Table, "CLASS", "WEIGHT", Output_Raster, IgnoreMsgData , MissingDataValue)
            ##        except:
            ##            gp.AddError(gp.getMessages(2))
            ##            raise
            ##        else:
            ##            gp.AddWarning(gp.getMessages(1))
            ##            gp.AddMessage(gp.getMessages(0))
            ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            #<==RDB Updated code
            #Same as CreateRaster above
            #++ Removed try: finally: statment as logic did not create any added effect.
            #++ only forced the remove join but what happens if join fails?        
            #++ Need to create in-memory Raster Layer for Join
            #Check for unsigned integer raster; cannot have negative missing data
            if NoDataArg != '#' and gp.describe(Input_Raster).pixeltype.upper().startswith('U'):
                NoDataArg2 = '#'
            else:
                NoDataArg2 = NoDataArg
            #Create new rasterlayer from input raster -> Result RasterLayer
            RasterLayer = "OutRas_lyr"

            if method == 0:
                arcpy.MakeRasterLayer_management(Input_Raster,RasterLayer)       # Unicamp removed 190718  (AL 200720) # AL added selection 220720

            #++ AddJoin requires and input layer or tableview not Input Raster Dataset.     
            #Join result layer with weights table
            dwrite ("Layer and Rasterlayer: " + Input_Raster + ", " + RasterLayer);

            if method == 0:
                arcpy.AddJoin_management(RasterLayer,"VALUE",Wts_Table,"CLASS")  # Unicamp removed 190718  (AL 200720) # AL added selection 220720
            # This is where it crashes on ISsue 44!https://github.com/gtkfi/ArcSDM/issues/44
            #return;
            
            # These are born in wrong place when the scratch workspace is filegeodatabase
            #Temp_Raster = os.path.join(arcpy.env.scratchFolder,'temp_raster')
            # Note! ScratchFolder doesn't seem to work            
            #Note Scratch these:
            #Temp_Raster = os.path.join(arcpy.env.scratchWorkspace,'temp_raster')
            Temp_Raster = gp.CreateScratchName('tmp_rst', '', 'rst', gp.scratchworkspace)
            #gp.AddMessage(" RasterLayer=" + RasterLayer);
            dwrite (" Temp_Raster=" + Temp_Raster);
            dwrite (" Wts_Table=" + Wts_Table);
            
            #Delete old temp_raster
            if gp.exists(Temp_Raster):
                arcpy.Delete_management(Temp_Raster)
                gc.collect();
                arcpy.ClearWorkspaceCache_management()

                gp.AddMessage("Deleted tempraster");
            
            #Copy created and joined raster to temp_raster
            if method == 0:
                arcpy.CopyRaster_management(RasterLayer,Temp_Raster,'#','#',NoDataArg2) # Unicamp removed this and added two new lines 190718  (AL 200720) # AL added selection 220720
            else:
                arcpy.CopyRaster_management(Input_Raster,Temp_Raster,'#','#',NoDataArg2)
                gp.JoinField_management(Temp_Raster, 'Value', Wts_Table, 'CLASS')
            arcpy.AddMessage(" Output_Raster: " + Output_Raster);
            
            #gp.Lookup_sa(Temp_Raster,"WEIGHT",Output_Raster)
            #This doesn't work in ArcGis Pro
            
            outras = arcpy.sa.Lookup(Temp_Raster,"WEIGHT")
            outras.save(Output_Raster);
            
            #return;
            #gp.addwarning(gp.getmessages())
            # ISsue 44 fix
            #arcpy.ClearWorkspaceCache_management()
            #arcpy.Delete_management(RasterLayer)
            
            #++ Optionally you can remove join but not necessary because join is on the layer
            #++ Better to just delete layer
    ##        #++ get name of join from the input table (without extention)
    ##        join = os.path.splitext(os.path.basename(Wts_Table))
    ##        join_name = join[0]
    ##        gp.RemoveJoin_management(RasterLayer,join_name)
            #<==
            
            #gp.AddMessage(Output_Raster + " exists: " + str(gp.Exists(Output_Raster)))
            if not gp.Exists(Output_Raster):
                gp.AddError( " " + Output_Raster + " does not exist.")
                raise
            #Output_Raster = gp.Describe(Output_Raster).CatalogPath
            Wts_Rasters.append(Output_Raster)
            #Check for Missing Data in raster's Wts table
            if not IgnoreMsgData:
                # Update the list for Missing Data Variance Calculation
                #dwrite ("Wts_Table = " + Wts_Table);
                tblrows = gp.SearchCursor(Wts_Table,"Class = %s" % MissingDataValue)
                tblrow = tblrows.Next()
                if tblrow: rasterList.append(gp.Describe(Output_Raster).CatalogPath)
            arcpy.AddMessage(" ") #Cycle done - add ONE linefeed
        #Get Post Logit Raster
        
        
        ''' Post Logit Raster '''
        
        gp.AddMessage( "\n" + "Getting Post Logit raster...\n" + "=" * 41)
        
        # This used to be comma separated, now +
        
        
        Input_Data_Str = ' + '.join('"{0}"'.format(w) for w in Wts_Rasters) #must be comma delimited string list
        arcpy.AddMessage(" Input_data_str: " + Input_Data_Str)
        Constant = math.log(Prior_prob/(1.0 - Prior_prob))
        
        if len(Wts_Rasters) == 1:
            InExpressionPLOG = "%s + %s" %(Constant,Input_Data_Str)
        else:
            InExpressionPLOG = "%s + (%s)" %(Constant,Input_Data_Str)
        #gp.AddMessage("="*41);
        gp.AddMessage(" InexpressionPlog: " + InExpressionPLOG);
        #gp.AddMessage("InExpression = " + str(InExpression))
    ##    PostLogit = os.path.join(gp.Workspace, OutputPrefix + "_PLOG")
    ##    try:
    ##        pass
    ##        gp.SingleOutputMapAlgebra_sa(InExpression, PostLogit)
    ##    except:
    ##        gp.AddError(gp.getMessages(2))
    ##        raise
    ##    else:
    ##        gp.AddWarning(gp.getMessages(1))
    ##        gp.AddMessage(gp.getMessages(0))
            
        
        #gp.AddMessage(" Wts_rasters: " + str(Wts_Rasters));
            
    
        #Get Post Probability Raster
        #gp.AddMessage("Exists(PostLogit) = " + str(gp.Exists(PostLogit)))
        
        gp.AddMessage("\nCreating Post Probability Raster...\n"+"="*41)
        try:
            #pass
            #PostLogitRL = os.path.join( gp.Workspace, "PostLogitRL")
            #gp.MakeRasterLayer_management(PostLogit,PostLogitRL)
            #InExpression = "EXP(%s) / ( 1.0 + EXP(%s))" %(PostLogitRL,PostLogitRL)
            PostProb = parameters[6].valueAsText #gp.GetParameterAsText(6)
            ##InExpression = "EXP(%s) / (1.0 + EXP(%s))" %(InExpressionPLOG,InExpressionPLOG)
            
            #Pre arcgis pro expression
            #InExpression = "%s = Exp(%s) / (1.0 + Exp(%s))" %(PostProb,InExpressionPLOG,InExpressionPLOG)  # <==RDB update to MOMA  07/01/2010
            InExpression = "Exp(%s) / (1.0 + Exp(%s))" %(InExpressionPLOG,InExpressionPLOG)  
            gp.AddMessage("InExpression = " + str(InExpression))
            #gp.addmessage("InExpression 1 ====> "  + InExpression) # <==RDB  07/01/2010
            # Fix: This is obsolete
            #gp.MultiOutputMapAlgebra_sa(InExpression)  # <==RDB  07/01/2010
            gp.AddMessage("Postprob: " + PostProb);
            #output_raster = gp.RasterCalculator(InExpression, PostProb);
            #output_raster.save(postprob)            
            gp.SingleOutputMapAlgebra_sa(InExpression, PostProb)
            #Pro/10 this needs to be done differently....
            #output_raster = arcpy.sa.RasterCalculator(InExpression, PostProb);
            #output_raster.save(postprob)            
            
            gp.SetParameterAsText(6, PostProb)
        except:
            gp.AddError(gp.getMessages(2))
            raise
        else:
            gp.AddWarning(gp.getMessages(1))
            gp.AddMessage(gp.getMessages(0))
        #gp.AddMessage("Exists(PostProb) = " + str(gp.Exists(PostProb)))

        #Create STD raster from raster's associated weights table
        gp.AddMessage("\nCreating STD rasters...\n"+"="*41)
        Std_Rasters = []
        i = 0
        mdidx = 0
        for Input_Raster in Input_Rasters:
            arcpy.AddMessage(" Processing " + Input_Raster);
            #<== RDB
            #++ Needs to be able to extract input raster name from full path.
            #++ Can't assume only a layer from ArcMap.
            ##Output_Raster = Input_Raster[:11] + "_S"
            ##Output_Raster = os.path.basename(Input_Raster)[:11] + "_S"  
            stdoutputrastername = os.path.basename(Input_Raster[:9]).replace(".","_") + "S"; # No . allowed in filegeodatgabases           

            #arcpy.AddMessage("Debug:" + stdoutputrastername);
            
            Output_Raster = gp.CreateScratchName(stdoutputrastername, '', 'rst', gp.scratchworkspace)
            #print ("DEBUG STD1");
            Wts_Table = Wts_Tables[i]
            if (wsdesc.workspaceType == "FileSystem"): #AL 060520
                if not(Wts_Table.endswith('.dbf')): #AL 060520
                    Wts_Table += ".dbf";            #AL 060520
            
            i += 1
            #Wts_Table = gp.Describe(Wts_Table).CatalogPath
    ##        gp.CreateRaster_sdm(Input_Raster, Wts_Table, "CLASS", "W_STD", Output_Raster, IgnoreMsgData, MissingDataValue)
            gp.AddMessage("OutputRaster:" + Output_Raster + " exists: " + str(gp.Exists(Output_Raster)))
            
            #<== Updated RDB
            #++ Same as calculate weight rasters above        
            #++ Need to create in-memory Raster Layer for Join
            #Check for unsigned integer raster; cannot have negative missing data
            
            if NoDataArg != '#' and gp.describe(Input_Raster).pixeltype.upper().startswith('U'):
                NoDataArg2 = '#'
            else:
                NoDataArg2 = NoDataArg
            dwrite ("NoDataArg = " + str(NoDataArg));
            RasterLayer = "OutRas_lyr2";
            if method == 0:
                gp.makerasterlayer(Input_Raster,RasterLayer)                  # Unicamp removed 190718  (AL 200720) # AL added selection 220720
                #++ Input to AddJoin must be a Layer or TableView
                gp.AddJoin_management(RasterLayer,"Value",Wts_Table,"CLASS")  # Unicamp removed 190718  (AL 200720) # AL added selection 220720
            # Folder doesn't seem to do the trick...
            #Temp_Raster = os.path.join(arcpy.env.scratchFolder,'temp_raster')
            #Temp_Raster = os.path.join(arcpy.env.scratchWorkspace,'temp_raster2')
            Temp_Raster = gp.CreateScratchName('tmp_rst', '', 'rst', gp.scratchworkspace)
            
            if gp.exists(Temp_Raster): 
                arcpy.Delete_management(Temp_Raster)
                gc.collect();
                arcpy.ClearWorkspaceCache_management()
                gp.AddMessage("Tmprst deleted.");
            dwrite ("RasterLayer=" + RasterLayer);
            dwrite ("Temp_Raster=" + Temp_Raster);
            
            if method == 0:
                arcpy.CopyRaster_management(RasterLayer, Temp_Raster,"#","#",NoDataArg2)  # Unicamp removed this and added two new lines 190718  (AL 200720) # AL added selection 220720
            else:
                arcpy.CopyRaster_management(Input_Raster, Temp_Raster,"#","#",NoDataArg2)
                gp.JoinField_management(Temp_Raster, 'Value', Wts_Table, 'CLASS')
            #gp.AddMessage("DEBUG STD1");
            gp.Lookup_sa(Temp_Raster,"W_STD",Output_Raster)
            # Issue 44 fix - no delete on temprasters
            #arcpy.Delete_management(RasterLayer)
            #gp.AddMessage("DEBUG STD1");
            #++ Optionally you can remove join but not necessary because join is on the layer
            #++ Better to just delete layer
    ##        #get name of join from the input table (without extenstion)
    ##        join = os.path.splitext(os.path.basename(Wts_Table))
    ##        join_name = join[0]
    ##        gp.RemoveJoin_management(RasterLayer,join_name)
            #<==
            
            if not gp.Exists(Output_Raster):
                gp.AddError(Output_Raster + " does not exist.")
                raise
            #Output_Raster = gp.Describe(Output_Raster).CatalogPath
            Std_Rasters.append(Output_Raster)
            gp.AddMessage(Output_Raster)  # <==RDB 07/01/2010 
        #gp.AddMessage("Created Std_Rasters = " + str(Std_Rasters))   
           
        gp.AddMessage("\nCreating Post Probability STD Raster...\n"+"="*41)
        #SQRT(SUM(SQR(kbgeol2_STD), SQR(kjenks_Std), SQR(rclssb2_Std)))
        PostProb_Std = parameters[7].valueAsText #gp.GetParameterAsText(7)
        
        
        #TODO: Figure out what this does!? TR
        #TODO: This is always false now
        if len(Std_Rasters) == 1: #If there is only one input... ??? TR 
            InExpression = '"%s"' % (Std_Rasters[0])
        else:
            SUM_args_list = []
            for Std_Raster in Std_Rasters:
                SUM_args_list.append("SQR(\"%s\")" % Std_Raster)
            #SUM_args = ",".join(SUM_args_list)
            SUM_args = " + ".join(SUM_args_list);
            gp.AddMessage("Sum_args: " + SUM_args + "\n" + "="*41);
            
            #Input_Data_Str = ' + '.join('"{0}"'.format(w) for w in Wts_Rasters) #must be comma delimited string list
       
            Constant = 1.0 / float(numTPs)
            ##InExpression = "SQRT(SQR(%s) * (%s + SUM(%s)))" %(PostProb,Constant,SUM_args)
            InExpression = "SQRT(SQR(%s) * (%s + SUM(%s)))" %(PostProb,Constant,SUM_args)  # PRe ARcGis pro
            #InExpression = "SquareRoot(Square(\"%s\") * (%s +(%s)))" %(PostProb,Constant,SUM_args)  
            gp.AddMessage("InExpression = " + str(InExpression))
        #SQRT(SUM(SQR(rclssb2_md_S),SQR(kbgeol2_md_S)))
        try:
            gp.addmessage("InExpression 2 ====> " + InExpression) # <==RDB
            #gp.MultiOutputMapAlgebra_sa(InExpression)   # <==RDB  07/01/2010
            #output_raster = gp.RasterCalculator(InExpression, PostProb_Std);
            gp.SingleOutputMapAlgebra_sa(InExpression, PostProb_Std)
            gp.SetParameterAsText(7,PostProb_Std)
        except:
            gp.AddError(gp.getMessages(2))
            raise
        else:
            gp.AddWarning(gp.getMessages(1))
            gp.AddMessage(gp.getMessages(0))
        #gp.AddMessage("Exists(PostProb_Std) = " + str(gp.Exists(PostProb_Std)))
        
        #Create Variance of missing data here and create totVar = VarMD + SQR(VarWts)
        if not IgnoreMsgData:
            #Calculate Missing Data Variance
            #gp.AddMessage("RowCount=%i"%len(rasterList))
            if len(rasterList) > 0:
                import arcsdm.missingdatavar_func    #AL 150620 added arcsdm.
                gp.AddMessage("Calculating Missing Data Variance...")
    ##            MDRasters=[]
    ##            for i in range(len(rasterList)):
    ##                MDRasters.append(str(rasterList[i]))
                MDRasters = rasterList
                #gp.AddMessage("MissingDataRasters = " + str(MDRasters))
                try:
                    MDVariance = parameters[8].valueAsText #gp.GetParameterAsText(8)
                    if gp.exists(MDVariance): arcpy.Delete_management(MDVariance)
                    #<== Tool DOES NOT EXIST = FAIL
                    #gp.MissingDataVariance_sdm(rasterList,PostProb,MDVariance)
                    missingdatavar_func.MissingDataVariance(gp,rasterList,PostProb,MDVariance)
                    Total_Std = parameters[9].valueAsText #gp.GetParameterAsText(9)
                    InExpression = 'SQRT(SUM(SQR(%s),%s))' % (PostProb_Std, MDVariance)
                    # OBsolete, replaced with raster calc
                    #InExpression = "\"%s\" = SQRT(SUM(SQR(\"%s\"),\"%s\"))" % (Total_Std, PostProb_Std, MDVariance)  # <==RDB update to MOMA
                    #InExpression = "SquareRoot(SUM ( Square (\"%s\"),\"%s\"))" % ( PostProb_Std, MDVariance)  # <==RDB update to MOMA
                    #InExpression = "SquareRoot( Square (\"%s\") + \"%s\")" % ( PostProb_Std, MDVariance)  # <==RDB update to MOMA
                    #gp.SetParameterAsText(8,MDVariance)
                    #gp.AddMessage(InExpression)
                    gp.AddMessage("Calculating Total STD...")
                    gp.addmessage("InExpression 3 ====> " + InExpression) # <==RDB
                    #gp.MultiOutputMapAlgebra_sa(InExpression)  # <==RDB
                    #output_raster = gp.RasterCalculator(InExpression, Total_Std);
                    gp.SingleOutputMapAlgebra_sa(InExpression, Total_Std)
                    gp.SetParameterAsText(9,Total_Std)
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
        #Confidence is PP / sqrt(totVar)
        gp.AddMessage("\nCalculating Confidence...\n"+"="*41)
        #PostProb1 / PP_Std
    ##    PostProbRL = os.path.join( gp.Workspace, "PostProbRL")
    ##    gp.MakeRasterLayer_management(PostProb,PostProbRL)
        #PostProbRL = gp.describe(PostProb).catalogpath
    ##    PostProb_StdRL = os.path.join( gp.Workspace, "PostProb_StdRL")
    ##    gp.MakeRasterLayer_management(Total_Std, PostProb_StdRL)
        #PostProb_StdRL = gp.describe(Total_Std).catalogpath
        Confidence = parameters[10].valueAsText #gp.GetParameterAsText(10)
        #InExpression = PostProbRL + " / " + PostProb_StdRL
        InExpression = "%s / %s" %(PostProb,PostProb_Std)  # PreARcGis pro
        #InExpression = '"%s" / "%s"' %(PostProbRL,PostProb_StdRL)  # <==RDB update to MOMA
        #gp.AddMessage("InExpression = " + str(InExpression))
        gp.addmessage("InExpression 4====> " + InExpression) # <==RDB
        try: 
            #gp.MultiOutputMapAlgebra_sa(InExpression)  # <==RDB
            #output_raster = arcpy.gp.RasterCalculator_sa(InExpression, Confidence);
            gp.SingleOutputMapAlgebra_sa(InExpression, Confidence)
            gp.SetParameterAsText(10,Confidence)
        except:
            gp.AddError(gp.getMessages(2))
            raise
        else:
            gp.AddWarning(gp.getMessages(1))
            gp.AddMessage(gp.getMessages(0))
        #Set derived output parameters
        gp.SetParameterAsText(6,PostProb)
        gp.SetParameterAsText(7,PostProb_Std)
        if MDVariance and (not IgnoreMsgData): gp.SetParameterAsText(8,MDVariance)
        else: gp.AddWarning('No Missing Data Variance.')
        if not (Total_Std == PostProb_Std): gp.SetParameterAsText(9,Total_Std)
        else: gp.AddWarning('Total STD same as Post Probability STD.')
        gp.SetParameterAsText(10,Confidence)
        
        gp.addmessage("done\n"+"="*41)
    except arcpy.ExecuteError as e:
        #TODO: Clean up all these execute errors in final version
        arcpy.AddError("\n");
        arcpy.AddMessage("Calculate Response caught arcpy.ExecuteError ");
        gp.AddError(arcpy.GetMessages())
        if len(e.args) > 0:
            #arcpy.AddMessage("Calculate Response caught arcpy.ExecuteError: ");
            args = e.args[0];
            args.split('\n')
            #arcpy.AddError(args);
                    
        arcpy.AddMessage("-------------- END EXECUTION ---------------");        
        raise 
    except:
        # get the traceback object
        tb = sys.exc_info()[2]
        # tbinfo contains the line number that the code failed on and the code from that line
        tbinfo = traceback.format_tb(tb)[0]
        # concatenate information together concerning the error into a message string
        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
            str(sys.exc_info()) + "\n"    #AL 050520
        #        str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
        # generate a message string for any geoprocessing tool errors
        msgs = "GP ERRORS:\n" + arcpy.GetMessages(2) + "\n"
        arcpy.AddError(msgs)

        # return gp messages for use with a script tool
        arcpy.AddError(pymsg)

        # print messages for use in Python/PythonWin
        print (pymsg)
        print (msgs)

        raise
