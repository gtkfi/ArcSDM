"""

    ArcSDM 5 - ArcSDM for ArcGis pro

    History:
    4/2016 Conversion started - TR
    9/2016 Conversion started to Python toolbox TR
    

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

def Execute(self, parameters, messages):


    try:
        # Import system modules
        import sys, os, math, traceback;
        import arcsdm.sdmvalues
        
        import arcsdm.sdmvalues;
        import arcsdm.workarounds_93;
        try:
            importlib.reload (arcsdm.sdmvalues)
            importlib.reload (arcsdm.workarounds_93);
        except :
            reload(arcsdm.sdmvalues);
            reload(arcsdm.workarounds_93);   

        # Create the Geoprocessor object

        import arcgisscripting


        gp = arcgisscripting.create()

        # Check out any necessary licenses
        gp.CheckOutExtension("spatial")

        # Load required toolboxes...
        ##gp.AddToolbox("C:/Program Files/ArcGIS/ArcToolbox/Toolboxes/Spatial Analyst Tools.tbx")
        ##gp.AddToolbox("C:/Program Files/ArcGIS/ArcToolbox/Toolboxes/Data Management Tools.tbx")

        #++remove hardcoded path for toolboxes... may not be only on c:\     #<== RDB
        ##sdm_toolbox = os.path.dirname(sys.path[0])+ os.sep + "Spatial Data Modeller Tools.tbx"
        ##gp.AddToolbox(sdm_toolbox)

        #gp.AddMessage("\n"+"="*41+"\n"+"="*41)
    # Script arguments...
        gp.AddMessage("\n"+"="*41)

        Evidence = parameters[0].valueAsText #gp.GetParameterAsText(0)
        Wts_Tables = parameters[1].valueAsText #gp.GetParameterAsText(1)
        Training_Points = parameters[2].valueAsText #gp.GetParameterAsText(2)
        IgnoreMsgData = parameters[3].value #gp.GetParameter(3)
        MissingDataValue = parameters[4].value #gp.GetParameter(4)
        #Cleanup extramessages after stuff
        #gp.AddMessage('Got arguments' )
        if IgnoreMsgData: # for nodata argument to CopyRaster tool
            NoDataArg = MissingDataValue
        else:
            NoDataArg = '#'
        UnitArea = parameters[5].value #gp.GetParameter(5)
        
        arcsdm.sdmvalues.appendSDMValues(gp, UnitArea, Training_Points)
    # Local variables...

        
    #Getting Study Area in counts and sq. kilometers
        Counts = 0
        desc = gp.Describe(gp.mask)
        #gp.AddMessage(desc.CatalogPath)
        
        rows = gp.SearchCursor(desc.catalogpath)
        row = rows.Next()
        while row:
            Counts += row.Count
            row = rows.Next()
            
        #gp.AddMessage(str(gp.CellSize))
        CellSize = float(gp.CellSize)
        Study_Area = (Counts * CellSize * CellSize / 1000000.0) / UnitArea
        #gp.AddMessage("Study Area: " + str(Study_Area))

        #Get number of training points
        numTPs = gp.GetCount_management(Training_Points)
        gp.AddMessage("numTPs: " + str(numTPs))
        
        #Prior probability
        Prior_prob = float(numTPs) / Study_Area 
        gp.AddMessage("Prior_prob = " + str(Prior_prob))

        #Get input evidence rasters
        Input_Rasters = Evidence.split(";")
        gp.AddMessage("Input rasters: " + str(Input_Rasters))

        #Get input weight tables
        Wts_Tables = Wts_Tables.split(";")
        gp.AddMessage("Wts_Tables = " + str(Wts_Tables))
        
        #Create weight raster from raster's associated weights table
        #gp.AddMessage("Getting Weights rasters...")
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
        gp.AddMessage("\n"+"="*41+" Starting "+"="*41)

        for Input_Raster in Input_Rasters:
            #<== RDB
            #++ Needs to be able to extract input raster name from full path.
            #++ Can't assume only a layer from ArcMap.
    ##        Output_Raster = os.path.join(gp.ScratchWorkspace,Input_Raster[:11] + "_W")  
            ##Output_Raster = os.path.basename(Input_Raster)[:11] + "_W"
            outputrastername = (Input_Raster[:9]) + "_W";
            
            # Create _W raster
            Output_Raster = gp.CreateScratchName(outputrastername, '', 'raster', gp.scratchworkspace)
            #gp.AddMessage("\n");            
            gp.AddMessage("\nOutputraster: " + outputrastername);
            
            Wts_Table = Wts_Tables[i]           
            
            #Increase the count for next round
            i += 1
            
            arcpy.AddMessage("WtsTable: " + Wts_Table);
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
            gp.MakeRasterLayer_management(Input_Raster,RasterLayer)
            
            #++ AddJoin requires and input layer or tableview not Input Raster Dataset.     
            #Join result layer with weights table
            gp.AddJoin_management(RasterLayer,"Value",Wts_Table,"CLASS")
            
           
            
            # These are born in wrong place when the scratch workspace is filegeodatabase
            #Temp_Raster = os.path.join(arcpy.env.scratchFolder,'temp_raster')
            # Note! ScratchFolder doesn't seem to work            
            Temp_Raster = os.path.join(arcpy.env.scratchWorkspace,'temp_raster')
            gp.AddMessage("RasterLayer=" + RasterLayer);
            gp.AddMessage("Temp_Raster=" + Temp_Raster);
            gp.AddMessage("Wts_Table=" + Wts_Table);
            
            #Delete old temp_raster
            if gp.exists(Temp_Raster):
                arcpy.Delete_management(Temp_Raster)
                gp.AddMessage("Deleted tempraster");
            
            #Copy created and joined raster to temp_raster
            gp.CopyRaster_management(RasterLayer,Temp_Raster,'#','#',NoDataArg2)
            gp.AddMessage("Output_Raster: " + Output_Raster);
            
            gp.Lookup_sa(Temp_Raster,"WEIGHT",Output_Raster)
            
            #gp.addwarning(gp.getmessages())
            arcpy.Delete_management(RasterLayer)
            #++ Optionally you can remove join but not necessary because join is on the layer
            #++ Better to just delete layer
    ##        #++ get name of join from the input table (without extention)
    ##        join = os.path.splitext(os.path.basename(Wts_Table))
    ##        join_name = join[0]
    ##        gp.RemoveJoin_management(RasterLayer,join_name)
            #<==
            
            #gp.AddMessage(Output_Raster + " exists: " + str(gp.Exists(Output_Raster)))
            if not gp.Exists(Output_Raster):
                gp.AddError(Output_Raster + " does not exist.")
                raise
            #Output_Raster = gp.Describe(Output_Raster).CatalogPath
            Wts_Rasters.append(Output_Raster)
            gp.AddMessage("Wts_rasters: " + str(Wts_Rasters));
            #Check for Missing Data in raster's Wts table
            print ("DEBUG");
            print ("DEBUG: IgnoreMsgData=" + str(IgnoreMsgData));
            if not IgnoreMsgData:
                # Update the list for Missing Data Variance Calculation
                gp.addMessage("Debug: Wts_Table = " + Wts_Table);
                tblrows = gp.SearchCursor(Wts_Table,"Class = %s" % MissingDataValue)
                tblrow = tblrows.Next()
                if tblrow: rasterList.append(gp.Describe(Output_Raster).CatalogPath)
        
        #Get Post Logit Raster
        #gp.AddMessage("Getting Post Logit raster...")
        # This used to be comma separated, now +
        Input_Data_Str = ' + '.join('"{0}"'.format(w) for w in Wts_Rasters) #must be comma delimited string list
        gp.AddMessage("Input_data_str: " + Input_Data_Str)
        Constant = math.log(Prior_prob/(1.0 - Prior_prob))
        if len(Wts_Rasters) == 1:
            InExpressionPLOG = "%s + %s" %(Constant,Input_Data_Str)
        else:
            InExpressionPLOG = "%s + (%s)" %(Constant,Input_Data_Str)
        gp.AddMessage("="*41);
        gp.AddMessage("InexpressionPlog: " + InExpressionPLOG);
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
            
        #Get Post Probability Raster
        #gp.AddMessage("Exists(PostLogit) = " + str(gp.Exists(PostLogit)))
        gp.AddMessage("Creating Post Probability Raster...\n"+"="*41)
        try:
            #pass
            #PostLogitRL = os.path.join( gp.Workspace, "PostLogitRL")
            #gp.MakeRasterLayer_management(PostLogit,PostLogitRL)
            #InExpression = "EXP(%s) / ( 1.0 + EXP(%s))" %(PostLogitRL,PostLogitRL)
            PostProb = parameters[6].valueAsText #gp.GetParameterAsText(6)
            ##InExpression = "EXP(%s) / (1.0 + EXP(%s))" %(InExpressionPLOG,InExpressionPLOG)
            
            #Pre arcgis pro expression
            #InExpression = "%s = EXP(%s) / (1.0 + EXP(%s))" %(PostProb,InExpressionPLOG,InExpressionPLOG)  # <==RDB update to MOMA  07/01/2010
            InExpression = "Exp(%s) / (1.0 + Exp(%s))" %(InExpressionPLOG,InExpressionPLOG)  
            #gp.AddMessage("InExpression = " + str(InExpression))
            gp.addmessage("InExpression 1 ====> "  + InExpression) # <==RDB  07/01/2010
            # Fix: This is obsolete
            #gp.MultiOutputMapAlgebra_sa(InExpression)  # <==RDB  07/01/2010
            gp.AddMessage("Postprob: " + PostProb);
            output_raster = gp.RasterCalculator(InExpression, PostProb);
            #output_raster.save(postprob)
            #gp.SingleOutputMapAlgebra_sa(InExpression, PostProb)
            #gp.SetParameterAsText(6, PostProb)
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
            #<== RDB
            #++ Needs to be able to extract input raster name from full path.
            #++ Can't assume only a layer from ArcMap.
            ##Output_Raster = Input_Raster[:11] + "_S"
            ##Output_Raster = os.path.basename(Input_Raster)[:11] + "_S"  
            Output_Raster = gp.CreateScratchName(os.path.basename(Input_Raster[:9]) + "_S", '', 'raster', gp.scratchworkspace)
            #print ("DEBUG STD1");
            Wts_Table = Wts_Tables[i]
            
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
            arcpy.AddMessage("Debug: " + str(NoDataArg));
            RasterLayer = "OutRas_lyr2";
            gp.makerasterlayer(Input_Raster,RasterLayer)
            #++ Input to AddJoin must be a Layer or TableView
            gp.AddJoin_management(RasterLayer,"Value",Wts_Table,"CLASS")
            # Folder doesn't seem to do the trick...
            #Temp_Raster = os.path.join(arcpy.env.scratchFolder,'temp_raster')
            Temp_Raster = os.path.join(arcpy.env.scratchWorkspace,'temp_raster')
            if gp.exists(Temp_Raster): 
                arcpy.Delete_management(Temp_Raster)
                gp.AddMessage("Tmpraster deleted.");
            gp.AddMessage("RasterLayer=" + RasterLayer);
            gp.AddMessage("Temp_Raster=" + Temp_Raster);
            
            arcpy.CopyRaster_management(RasterLayer, Temp_Raster,"#","#",NoDataArg2)
            #gp.AddMessage("DEBUG STD1");
            gp.Lookup_sa(Temp_Raster,"W_STD",Output_Raster)
            arcpy.Delete_management(RasterLayer)
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
        if len(Std_Rasters) == 1:
            InExpression = Std_Rasters[0]
        else:
            SUM_args_list = []
            for Std_Raster in Std_Rasters:
                SUM_args_list.append("Square(\"%s\")" % Std_Raster)
            #SUM_args = ",".join(SUM_args_list)
            SUM_args = " + ".join(SUM_args_list);
            gp.AddMessage("Sum_args: " + SUM_args + "\n" + "="*41);
            
            #Input_Data_Str = ' + '.join('"{0}"'.format(w) for w in Wts_Rasters) #must be comma delimited string list
       
            Constant = 1.0 / float(numTPs)
            PostProb_Std = parameters[2].valueAsText #gp.GetParameterAsText(7)
            ##InExpression = "SQRT(SQR(%s) * (%s + SUM(%s)))" %(PostProb,Constant,SUM_args)
            #InExpression = "SQRT(SQR(%s) * (%s + SUM(%s)))" %(PostProb,Constant,SUM_args)  # PRe ARcGis pro
            InExpression = "SquareRoot(Square(\"%s\") * (%s +(%s)))" %(PostProb,Constant,SUM_args)  
        #gp.AddMessage("InExpression = " + str(InExpression))
        #SQRT(SUM(SQR(rclssb2_md_S),SQR(kbgeol2_md_S)))
        try:
            gp.addmessage("InExpression 2 ====> " + InExpression) # <==RDB
            #gp.MultiOutputMapAlgebra_sa(InExpression)   # <==RDB  07/01/2010
            output_raster = gp.RasterCalculator(InExpression, PostProb_Std);
            #gp.SingleOutputMapAlgebra_sa(InExpression, PostProb_Std)
            #gp.SetParameterAsText(7,PostProb_Std)
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
                import MissingDataVar_Func
                gp.AddMessage("Calculating Missing Data Variance...")
    ##            MDRasters=[]
    ##            for i in range(len(rasterList)):
    ##                MDRasters.append(str(rasterList[i]))
                MDRasters = rasterList
                #gp.AddMessage("MissingDataRasters = " + str(MDRasters))
                try:
                    MDVariance = gp.GetParameterAsText(8)
                    if gp.exists(MDVariance): arcpy.Delete_management(MDVariance)
                    #<== Tool DOES NOT EXIST = FAIL
                    #gp.MissingDataVariance_sdm(rasterList,PostProb,MDVariance)
                    MissingDataVar_Func.MissingDataVariance(gp,rasterList,PostProb,MDVariance)
                    Total_Std = gp.GetParameterAsText(9)
                    ##InExpression = 'SQRT(SUM(SQR(%s),%s))' % (PostProb_Std, MDVariance)
                    InExpression = '%s = SQRT(SUM(SQR(%s),%s))' % (Total_Std, PostProb_Std, MDVariance)  # <==RDB update to MOMA
                    #gp.SetParameterAsText(8,MDVariance)
                    #gp.AddMessage(InExpression)
                    gp.AddMessage("Calculating Total STD...")
                    gp.addmessage("InExpression 3 ====> " + InExpression) # <==RDB
                    gp.MultiOutputMapAlgebra_sa(InExpression)  # <==RDB
                    #gp.SingleOutputMapAlgebra_sa(InExpression, Total_Std)
                    #gp.SetParameterAsText(9,Total_Std)
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
        gp.AddMessage("Calculating Confidence...\n"+"="*41)
        #PostProb1 / PP_Std
    ##    PostProbRL = os.path.join( gp.Workspace, "PostProbRL")
    ##    gp.MakeRasterLayer_management(PostProb,PostProbRL)
        PostProbRL = gp.describe(PostProb).catalogpath
    ##    PostProb_StdRL = os.path.join( gp.Workspace, "PostProb_StdRL")
    ##    gp.MakeRasterLayer_management(Total_Std, PostProb_StdRL)
        PostProb_StdRL = gp.describe(Total_Std).catalogpath
        Confidence = gp.GetParameterAsText(10)
        #InExpression = PostProbRL + " / " + PostProb_StdRL
        #InExpression = "%s = %s / %s" %(Confidence,PostProbRL,PostProb_StdRL)  # PreARcGis pro
        InExpression = '"%s" / "%s"' %(PostProbRL,PostProb_StdRL)  # <==RDB update to MOMA
        #gp.AddMessage("InExpression = " + str(InExpression))
        gp.addmessage("InExpression 4====> " + InExpression) # <==RDB
        try: 
            #gp.MultiOutputMapAlgebra_sa(InExpression)  # <==RDB
            output_raster = gp.RasterCalculator(InExpression, Confidence);
            #gp.SingleOutputMapAlgebra_sa(InExpression, Confidence)
            #gp.SetParameterAsText(10,Confidence)
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
        arcpy.AddMessage("Calculate weights caught arcpy.ExecuteError ");
        gp.AddError(arcpy.GetMessages())
        if len(e.args) > 0:
            #arcpy.AddMessage("Calculate weights caught arcpy.ExecuteError: ");
            args = e.args[0];
            args.split('\n')
            #arcpy.AddError(args);
                    
        arcpy.AddMessage("-------------- END EXECUTION ---------------");        
        raise arcpy.ExecuteError;   
    except:
        # get the traceback object
        tb = sys.exc_info()[2]
        # tbinfo contains the line number that the code failed on and the code from that line
        tbinfo = traceback.format_tb(tb)[0]
        # concatenate information together concerning the error into a message string
        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
                str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
        # generate a message string for any geoprocessing tool errors
        msgs = "GP ERRORS:\n" + gp.GetMessages(2) + "\n"
        gp.AddError(msgs)

        # return gp messages for use with a script tool
        gp.AddError(pymsg)

        # print messages for use in Python/PythonWin
        print (pymsg)
        print (msgs)

        raise
