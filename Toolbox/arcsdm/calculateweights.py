# -*- coding: utf-8 -*-
"""
    Calculate Weights - ArcSDM 5 for ArcGis pro 
    Recode from the original by Tero Rönkkö / Geological survey of Finland
    Update by Arianne Ford, Kenex Ltd. 2018
   
    History: 
    18.5.2020 Added changing Evidence Layer raster type from RasterBand or RasterLayer to RasterDataset / Arto Laiho, GTK/GSF
    15.5.2020 Added Evidence Layer and Training points coordinate system checking / Arto Laiho, GTK/GSF
    27.4.2020 Database table field name cannot be same as alias name when ArcGIS Pro with File System Workspace is used. / Arto Laiho, GTK/GSF
    09/01/2018 Bug fixes for 10.x, fixed perfect correlation issues, introduced patch for b-db<=0 - Arianne Ford, Kenex Ltd.
    3.11.2017 Updated categorical calculations when perfect correlation exists as described in issue 66
    27.9.2016 Calculate weights output cleaned
    23.9.2016 Goes through
    12.8.2016 First running version for pyt. Shapefile training points and output?
    1.8.2016 Python toolbox version started
    12.12.2016 Fixes
    
    
    
    Spatial Data Modeller for ESRI* ArcGIS 9.3
    Copyright 2009
    Gary L Raines, Reno, NV, USA: production and certification
    Don L Sawatzky, Spokane, WA, USA: Python software development
    Ascending or Descending:  Calculates accumulative Wts for accumulative row areas and num points
        for ascending or descending classes both with ascending counts.
    
    Categorical: Calculates Wts for each row, then reports those Wts for rows having >= Confidence.
        For rows having < Confidence, Wts are produced from sum of row areas and num points of
        classes < Confidence.
    Required Input(0): Integer raster dataset
    Optional Input(1): Attribute field name
    Required Input(2): Points feature class
    Required Input - Output type(3): Ascending, Descending, Categorical, Unique
    Required Output (4): Weights table
    Required Input(5): Confident_Contrast
    Required Input(6):  Unitarea
    Required Input(7): MissingDataValue
    Derived Output(8) - Success of calculation, whether Valid table: True or False
"""
# Import system modules
import sys, os, traceback
import math
import arcpy;
# TODO: Make these imports soem other way?
if __name__ == "__main__":
    import sys, string, os, math, traceback
    import sdmvalues, workarounds_93
    import sdmvalues;
    import arcgisscripting
    
    



# Create the Geoprocessor object
import arcgisscripting
gp = arcgisscripting.create()

# Check out any necessary licenses
gp.CheckOutExtension("spatial")

class ErrorExit(Exception): pass

def MakeWts(patternNTP, patternArea, unit, totalNTP, totalArea, Type):
    """
                    >>> Graeme's Fortran algorithm - Appendix II <<<
                    class(s=totalArea, b=patternArea,unit=unit,db=patternNTP,ds=totalNTP)
                    print *, � area of study region ?�
                    read *, s
                    print *, �area of binary map pattern?�
                    read *, b
                    print , � area of unit cell?�
                    read *, unit
                    print *, �no of deposits on pattern?�
                    read *, db
                    print , � total no of deposits?�
                    read *, ds
    """
    db = patternNTP
    ds = totalNTP
    s = totalArea/unit
    b = patternArea/unit
    #gp.addwarning("%s"%[db,ds,s,b])

    try:
       #>>>>>>>>>>>> Traps
        #Traps and fixes for various acceptable data anomalies
        if db > ds: #Graeme's trap
            gp.addwarning( 'Input error: More than one TP per Unitcell in study area.')
            return tuple([0.0]*7)
        if Type == 'Categorical': # Categorical generalization
            if db == 0:
                #db = 0.01
                return tuple([0.0]*7)
            elif db == ds:
                #As with issue #66 suggests - replaced with db -=.99:
                #db -= 0.01 # Won't work when s-b < ds-db 
                db -= 0.99
                #return tuple([0.0]*7)
            elif db == 0.001:
                db = ds
                db -= 0.99
        else: # Ascending and Descending generalization
            if db ==0: #no accumulation
                #db = 0.01
                return tuple([0.0]*7)
            elif db == ds: #Maximum accumulation
                #return tuple([0.0]*7)
                db -= 0.99 # Won't work when s-b < ds-db
        # Fix b so can compute W- when db = MaxTPs
        if (s - b) <= (ds - db):  b = s + db - ds - 0.99
        # Warning if cannot compute W+
        if (b-db) <= 0.0:
            #fix pattern area if area less than unit size
            b = db + 1
            #gp.addwarning( 'More than one TP per Unitcell in pattern.')
            #return tuple([0.0]*7)


        #<<<<<<<<<<<<<<<<<<End of traps
        
        db = float(db)
        ds = float(ds)

        #gp.addwarning( "db, ds, b, s %s"%([db, ds, b, s]))
        
        #Calculate W+
         # b-db can be negative or zero, but is trapped above
        pbd = db/ds
        pbdb = (b-db) / (s-ds)
        ls = pbd/pbdb
        wp = math.log(ls)
        #gp.addwarning("%s"%(wp))

        #Calculate vp and sp
         # b-db can be negative, but is trapped above
        vp = (1.0 / db) + (1.0 / (b-db))
        sp = math.sqrt(vp)
        #gp.addwarning("%s"%(sp))
        
        #Calculate W-
        #(s - b) <= (ds - db) creates negative arg to log
        #This inequality is trapped and fixed above
        pbbd = (ds-db) / ds
        pbbdb = (s-b-ds+db) / (s-ds)
        ln = pbbd / pbbdb
        wm = math.log(ln)
        #gp.addwarning("%s"%(wm))

        #Calculate vm and sm        
        vm = (1.0 / (ds-db)) + (1.0 / (s-b-ds+db))
        sm = math.sqrt(vm)
        #gp.addwarning("%s"%(sm))
        
        #Calculate Contrast
        c = wp - wm
        #gp.addwarning("%s"%(c))
        
        #Calculate Contrast Std Dev
        sc = math.sqrt(vp+vm)
        #gp.addwarning("%s"%(sc))

        #gp.addwarning("%s"%(c/sc))
        return (wp,sp,wm,sm,c,sc,c/sc)
    
    except Exception as msg:
        # get the traceback object
        tb = sys.exc_info()[2]
        # tbinfo contains the line number that the code failed on and the code from that line
        tbinfo = traceback.format_tb(tb)[0]
        # concatenate information together concerning the error into a message string
        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
            str(sys.exc_info()) + "\n"    #AL 050520
        #    str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
        # generate a message string for any geoprocessing tool errors
        msgs = "GP ERRORS:\n" + gp.GetMessages(2) + "\n"
        gp.AddError(msgs)

        # return gp messages for use with a script tool
        gp.AddError(pymsg)

        # print messages for use in Python/PythonWin
        print (pymsg)
        print (msgs)

        return None
            
# Load arguments...
def Calculate(self, parameters, messages):
    import importlib;
    try:
        import arcsdm.sdmvalues;
        import arcsdm.workarounds_93;
        try:
            importlib.reload (arcsdm.sdmvalues)
            importlib.reload (arcsdm.workarounds_93);
        except :
            reload(arcsdm.sdmvalues);
            reload(arcsdm.workarounds_93);        
        gp.OverwriteOutput = 1
        gp.LogHistory = 1
        EvidenceLayer = parameters[0].valueAsText

        # Test data type of Evidence Layer #AL 150520
        evidenceDescr = arcpy.Describe(EvidenceLayer)
        evidenceCoord = evidenceDescr.spatialReference.name
        arcpy.AddMessage("Evidence Layer is " + EvidenceLayer + " and its data type is " + evidenceDescr.datatype)
        if (evidenceDescr.datatype == "RasterBand" or evidenceDescr.datatype == "RasterLayer"):
            # Try to change RasterBand or RasterLayer to RasterDataset #AL 180520
            evidence1 = os.path.split(EvidenceLayer)
            evidence2 = os.path.split(evidence1[0])
            if (evidence1[1] == evidence2[1] or evidence1[1][:4] == "Band"):
                EvidenceLayer = evidence1[0]
                evidenceDescr = arcpy.Describe(EvidenceLayer)
                arcpy.AddMessage("Evidence Layer is now " + EvidenceLayer + " and its data type is " + evidenceDescr.datatype)
            else:
                arcpy.AddError("ERROR: Data Type of Evidence Layer cannot be RasterBand, use Raster Dataset.")
                raise

        valuetype = gp.GetRasterProperties (EvidenceLayer, 'VALUETYPE')
        valuetypes = {1:'Integer', 2:'Float'}
        #if valuetype != 1:
        if valuetype > 8:  # <==RDB  07/01/2010 - new  integer valuetype property value for arcgis version 10
            gp.adderror('ERROR: ' + EvidenceLayer + ' is not an integer-type raster because VALUETYPE is ' + str(valuetype)) #AL 040520
            raise ErrorExit
        CodeName =  parameters[1].valueAsText #gp.GetParameterAsText(1)
        TrainingSites =  parameters[2].valueAsText
        # Test coordinate system of Training sites and confirm it is same than Evidence Layer #AL 150520 
        trainingDescr = arcpy.Describe(TrainingSites)
        trainingCoord = trainingDescr.spatialReference.name
        if (evidenceCoord != trainingCoord):
            arcpy.AddError("ERROR: Coordinate System of Evidence Layer is " + evidenceCoord + " and Training points it is " + trainingCoord + ". These must be same.")
            raise
        Type =  parameters[3].valueAsText
        wtstable = parameters[4].valueAsText;

        # If using non gdb database, lets add .dbf
        wdesc = arcpy.Describe(gp.workspace)
        if (wdesc.workspaceType == "FileSystem"):
            if not(wtstable.endswith('.dbf')):
                wtstable += ".dbf";
        
        Confident_Contrast = float( parameters[5].valueAsText)
        #Unitarea = float( parameters[6].valueAsText)
        Unitarea = float( parameters[6].value)
        MissingDataValue = int( parameters[7].valueAsText) # Python 3 fix, long -> int
        #gp.AddMessage("Debug step 12");
        arcsdm.sdmvalues.appendSDMValues(gp,  Unitarea, TrainingSites)
        arcpy.AddMessage("="*10 + " Calculate weights " + "="*10)
    # Process: ExtractValuesToPoints
        arcpy.AddMessage ("%-20s %s (%s)" %("Creating table:" , wtstable, Type ));

        #tempTrainingPoints = gp.createscratchname("OutPoints", "FC", "shapefile", gp.scratchworkspace)
        #gp.ExtractValuesToPoints_sa(TrainingSites, EvidenceLayer, tempTrainingPoints, "NONE", "VALUE_ONLY")
        assert isinstance(EvidenceLayer, object)
        tempTrainingPoints = arcsdm.workarounds_93.ExtractValuesToPoints(gp, EvidenceLayer, TrainingSites, "TPFID")
    # Process: Summarize Frequency and manage fields
    
        #Statistics = gp.createuniquename("WtsStatistics.dbf")
        
        Statistics = gp.createuniquename("WtsStatistics")
        if gp.exists(Statistics): gp.Delete_management(Statistics)
        gp.Statistics_analysis(tempTrainingPoints, Statistics, "rastervalu sum" ,"rastervalu")
    # Process: Create the table
            
        gp.CreateTable_management(os.path.dirname(wtstable), os.path.basename(wtstable), Statistics)
        
        gp.AddField_management (wtstable, "Count", "long") 
        gp.AddField_management (wtstable, "Area", 'double')
        gp.AddField_management (wtstable, "AreaUnits", 'double')
        gp.AddField_management (wtstable, "CLASS", "long") 
        if CodeName != None and len(CodeName) > 0:
            gp.AddField_management(wtstable,"CODE","text","5","#","#","Symbol")
        gp.AddField_management (wtstable, "AREA_SQ_KM", "double") 
        gp.AddField_management (wtstable, "AREA_UNITS", "double")
        gp.AddField_management (wtstable, "NO_POINTS", "long")
        gp.AddField_management(wtstable,"WPLUS","double","10","4","#","W+")
        gp.AddField_management(wtstable,"S_WPLUS","double","10","4","#","W+ Std")
        gp.AddField_management(wtstable,"WMINUS","double","10","4","#","W-")
        gp.AddField_management(wtstable,"S_WMINUS","double","10","4","#","W- Std")
        # Database table field name cannot be same as alias name when ArcGIS Pro with File System Workspace is used. #AL
        gp.AddField_management(wtstable,"CONTRAST","double","10","4","#","Contrast_")
        gp.AddField_management(wtstable,"S_CONTRAST","double","10","4","#","Contrast_Std")
        gp.AddField_management(wtstable,"STUD_CNT","double","10","4","#","Studentized_Contrast")
        gp.AddField_management(wtstable,"GEN_CLASS","long","#","#","#","Generalized_Class")
        gp.AddField_management(wtstable,"WEIGHT","double","10","6","#","Generalized_Weight")
        gp.AddField_management(wtstable,"W_STD","double","10","6","#","Generalized_Weight_Std")
        OIDName = gp.Describe(wtstable).OIDFieldName

        #Fill output table rows depending on Type    
        desc = gp.describe(EvidenceLayer)
        cellsize = desc.MeanCellWidth
        if desc.datatype == 'RasterLayer': EvidenceLayer =desc.catalogpath
        if Type == "Descending":
            wtsrows = gp.InsertCursor(wtstable)
            rows = gp.SearchCursor(EvidenceLayer,'','','','Value D')
            row = rows.Next()
            while row:
                #gp.AddMessage("Inserting row.")
                wtsrow = wtsrows.NewRow()
                wtsrow.rastervalu = row.Value
                wtsrow.SetValue('class',row.Value)
                if CodeName != None and len(CodeName) > 0: 
                    wtsrow.Code = row.GetValue(CodeName)
                #This related to Access Personal geodatabase bug
                #arcpy.AddMessage("DEBUG: Rowcount:%s"%(str(row.Count)));
                wtsrow.Count = row.Count
                statsrows = gp.SearchCursor(Statistics,'rastervalu = %i'%row.Value)
                if statsrows:
                    statsrow = statsrows.Next()
                    if statsrow:
                        rowFreq = statsrow.Frequency
                    else:
                        rowFreq = 0
                wtsrow.Frequency = rowFreq
                #gp.addmessage('Desc: Class: %d, Count: %d,  Freq: %d'%(row.Value,row.Count, rowFreq))
                wtsrows.InsertRow(wtsrow)            
                row = rows.next()
            del wtsrows, wtsrow
           
        else: # Ascending or Categorical or Unique
            wtsrows = gp.InsertCursor(wtstable)
            rows = gp.SearchCursor(EvidenceLayer)
            row = rows.Next()
            while row:
                wtsrow = wtsrows.NewRow()
                wtsrow.rastervalu = row.Value
                wtsrow.SetValue('class',row.Value)                
                if CodeName != None and len(CodeName) > 0: 
                    wtsrow.Code = row.GetValue(CodeName)
                #arcpy.AddMessage("DEBUG: Rowcount:%s"%(str(row.Count)));
                wtsrow.Count = row.Count
                statsrows = gp.SearchCursor(Statistics,'rastervalu = %i'%row.Value)
                if statsrows:
                    statsrow = statsrows.Next()
                    if statsrow:
                        wtsrow.Frequency = statsrow.Frequency
                    else:
                        wtsrow.Frequency = 0                    
                wtsrows.InsertRow(wtsrow)            
                row = rows.Next()
            del wtsrows, wtsrow
        del row,rows
     # Calculate fields
        #gp.AddMessage('Calculating weights...')
        #gp.AddMessage("[count] * %f * %f /1000000.0"%(cellsize,cellsize))
        arcpy.CalculateField_management(wtstable, "area",  "!count! * %f / 1000000.0"%(cellsize**2), "PYTHON_9.3")
        arcpy.CalculateField_management(wtstable, "areaunits",  "!area! / %f"% Unitarea, "PYTHON_9.3")

        #gp.CalculateField_management (wtstable, "area", "!count! * %f / 1000000.0"%(cellsize**2))
        #gp.CalculateField_management (wtstable, "areaunits", "!area! / %f"% Unitarea)
     # Calculate accumulative fields
        if Type in ("Ascending", "Descending"):
            wtsrows = gp.UpdateCursor(wtstable)
            wtsrows.reset()
            wtsrow = wtsrows.Next()
            if wtsrow:
                if wtsrow.GetValue('class') is not MissingDataValue:
                    lastTotalTP = wtsrow.Frequency
                    lastTotalArea = wtsrow.Area # sq km
                    lastTotalAreaUnits = wtsrow.AreaUnits # unit cells
                    wtsrow.NO_POINTS = lastTotalTP
                    wtsrow.AREA_SQ_KM = lastTotalArea # sq km
                    wtsrow.AREA_UNITS = lastTotalAreaUnits # unit cells
                else:
                    lastTotalTP = 0
                    lastTotalArea = 0
                    lastTotalAreaUnits = 0
                    wtsrow.NO_POINTS = wtsrow.Frequency
                    wtsrow.AREA_SQ_KM = wtsrow.Area # sq km
                    wtsrow.AREA_UNITS = wtsrow.AreaUnits # unit cells
                #gp.addmessage('%s: Freq: %d, Area: %f,  UnitAreas: %f'%(Type, wtsrow.Frequency,wtsrow.Area, wtsrow.AreaUnits))
                wtsrows.UpdateRow(wtsrow)
                wtsrow = wtsrows.Next()
            while wtsrow:
                if wtsrow.GetValue('class') is not MissingDataValue:
                    lastTotalTP += wtsrow.Frequency
                    lastTotalArea += wtsrow.Area
                    lastTotalAreaUnits += wtsrow.AreaUnits
                    wtsrow.NO_POINTS = lastTotalTP
                    wtsrow.AREA_SQ_KM = lastTotalArea # sq km
                    wtsrow.AREA_UNITS = lastTotalAreaUnits # unit cells
                else:
                    wtsrow.NO_POINTS = wtsrow.Frequency
                    wtsrow.AREA_SQ_KM = wtsrow.Area #sq km
                    wtsrow.AREA_UNITS = wtsrow.AreaUnits # unit cells
                #gp.addmessage('%s: Freq: %d, Area: %f,  UnitAreas: %f'%(Type,wtsrow.Frequency,wtsrow.Area, wtsrow.AreaUnits))
                wtsrows.UpdateRow(wtsrow)
                wtsrow = wtsrows.Next()
            totalArea = lastTotalArea # sq km
            totalTPs = lastTotalTP
            del wtsrow,wtsrows
        #Calculate non-accumulative fields
        elif Type in ("Categorical", "Unique"):
            totalArea = 0
            totalTPs = 0
            wtsrows = gp.UpdateCursor(wtstable)
            wtsrow = wtsrows.Next()
            while wtsrow:
                wtsrow.NO_POINTS = wtsrow.Frequency
                wtsrow.AREA_SQ_KM = wtsrow.Area # sq km
                wtsrow.AREA_UNITS = wtsrow.AreaUnits # unit cells
                #gp.addMessage("Debug class: " + str(wtsrow.GetValue('class')));
                
                if wtsrow.getValue("class") != MissingDataValue:  
                    totalTPs += wtsrow.Frequency
                    totalArea += wtsrow.Area
                wtsrows.UpdateRow(wtsrow)
                wtsrow = wtsrows.Next()
            del wtsrow,wtsrows
        else:
            gp.AddWarning('Type %s not implemented'%Type)
            
        #Calculate weights, etc from filled-in fields
        wtsrows = gp.UpdateCursor(wtstable)
        wtsrow = wtsrows.Next()
        while wtsrow:
            #gp.AddMessage('Got to here...%i'%wtsrow.Class)
            #No calculations for missingdata class
            if wtsrow.GetValue('class') == MissingDataValue:
                wtsrow.wplus = 0.0
                wtsrow.s_wplus = 0.0
                wtsrow.wminus = 0.0
                wtsrow.s_wminus = 0.0
                wtsrow.contrast = 0.0
                wtsrow.s_contrast = 0.0
                wtsrow.stud_cnt = 0.0
            else:
                #gp.addMessage("Debug:" + str((wtsrow.NO_POINTS, wtsrow.AREA_SQ_KM, Unitarea, totalTPs, totalArea, Type)));
                wts = MakeWts(wtsrow.NO_POINTS, wtsrow.AREA_SQ_KM, Unitarea, totalTPs, totalArea, Type)
                if not wts:
                    gp.AddError("Weights calculation aborted.")
                    raise ErrorExit
                (wp,sp,wm,sm,c,sc,c_sc) = wts
                #gp.AddMessage( "Debug out: " +  str((wp,sp,wm,sm,c,sc,c_sc)))
                wtsrow.wplus = wp
                wtsrow.s_wplus = sp
                wtsrow.wminus = wm
                wtsrow.s_wminus = sm
                wtsrow.contrast = c
                wtsrow.s_contrast = sc
                wtsrow.stud_cnt = c_sc
            wtsrows.UpdateRow(wtsrow)
            wtsrow = wtsrows.Next()
        del wtsrow,wtsrows
            
    #Generalize table
        #Get Study Area size in Evidence counts    
        evRows = gp.SearchCursor(EvidenceLayer)
        evRow = evRows.Next()
        studyArea = 0
        while evRow:
            studyArea = studyArea + evRow.Count
            evRow = evRows.Next()
        del evRow, evRows
        #gp.AddMessage("studyArea size(cells)=" + str(studyArea))
        
        #Get total number of training points    
        ds = gp.GetCount_management(tempTrainingPoints) #TP selected
        #gp.AddMessage("ds="+str(ds))

        Success = True #Assume Valid Table: Has confident classes
        if Type in ("Ascending", "Descending", "Categorical"):
            #gp.AddMessage("Generalizing " + Type + "...")
            if Type != "Categorical": #i.e., Ascending or Descending
                #Select confident rows
                WgtsTblRows = gp.SearchCursor(wtstable,"STUD_CNT >= " + str(Confident_Contrast))
                #Get confidence row OID with maximum contrast
                WgtsTblRow = WgtsTblRows.Next()
                maxContrast = -9999999.0
                patNoTPs = 0; patArea = 0.0
                maxOID = -1
                while WgtsTblRow:
                    if WgtsTblRow.Class is not MissingDataValue:
                        if (WgtsTblRow.Contrast > maxContrast) and (WgtsTblRow.STUD_CNT >= Confident_Contrast):
                            maxContrast = WgtsTblRow.Contrast
                            maxOID = WgtsTblRow.GetValue(OIDName)
                            maxWplus = WgtsTblRow.Wplus
                            maxWplus_Std = WgtsTblRow.S_Wplus
                            maxWminus = WgtsTblRow.Wminus
                            maxWminus_Std = WgtsTblRow.S_Wminus
                            maxStdContr = WgtsTblRow.STUD_CNT
                            patNoTPs += WgtsTblRow.No_points
                            patArea += WgtsTblRow.Area_units
                    WgtsTblRow = WgtsTblRows.Next()
                #Set state of calculation
                #gp.AddMessage("Max OID: " + str(maxOID))
                if maxOID >= 0:
                    #Select rows with OID <= maxOID and Set new field values
                    Where = OIDName + " <= " + str(maxOID)
                    WgtsTblRows = gp.UpdateCursor(wtstable, Where)
                    WgtsTblRow = WgtsTblRows.Next()
                    while WgtsTblRow:
                        """ Missing data row should be processed after Gen_Class=2 is complete.
                            Then MD row should be found. If found, get area and num points of
                            pattern=2 and compute MD std.
                        """
                        if WgtsTblRow.Class == MissingDataValue:
                            WgtsTblRow.Gen_Class = MissingDataValue
                            WgtsTblRow.Weight = 0.0
                            WgtsTblRow.W_Std = 0.0
                        else:
                            WgtsTblRow.Gen_Class = 2
                            WgtsTblRow.Weight = maxWplus
                            WgtsTblRow.W_Std = maxWplus_Std
                        WgtsTblRows.UpdateRow(WgtsTblRow)
                        WgtsTblRow = WgtsTblRows.Next()
                    #gp.AddMessage("Set IN rows.")

                    #Select rows with OID > maxOID and Set new field values
                    Where = OIDName + " > " + str(maxOID)
                    WgtsTblRows = gp.UpdateCursor(wtstable, Where)
                    WgtsTblRow = WgtsTblRows.Next()
                    while WgtsTblRow:
                        if WgtsTblRow.Class == MissingDataValue:
                            #gp.AddMessage("Setting missing data gen_class...")
                            WgtsTblRow.Gen_Class = MissingDataValue
                            WgtsTblRow.Weight = 0.0
                            WgtsTblRow.W_Std = 0.0
                        else:
                            WgtsTblRow.Gen_Class = 1
                            WgtsTblRow.Weight = maxWminus
                            WgtsTblRow.W_Std = maxWminus_Std
                        WgtsTblRows.UpdateRow(WgtsTblRow)
                        WgtsTblRow = WgtsTblRows.Next()        
                    #gp.AddMessage("Set OUT rows.")
                else:
                    gp.AddWarning("No Contrast for type %s satisfied the user defined confidence level %s"%(Type,Confident_Contrast))
                    gp.AddWarning("Table %s is incomplete."%wtstable)
                    #gp.Delete(wtstable)
                    Success = False # Invalid Table: No confidence
        
            else: #Categorical
                #Get Wts and Wts_Std for class values outside confidence
                Out_Area = 0
                Out_NumTPs = 0.0
                #Out_SumWStds = 0.0
                Out_Num = 0

                #>>>>>>>>>>>>>>>>Out Rows>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                #Select rows having less than specified absolute confidence; they are assigned to Out_Gen_Class
                WhereClause = "(STUD_CNT > -%f) and (STUD_CNT < %f)" %(Confident_Contrast, Confident_Contrast)
                #gp.AddMessage(WhereClause)
                WgtsTblRows = gp.SearchCursor(wtstable, WhereClause)
                WgtsTblRow = WgtsTblRows.Next()
                #Categorical might have a Class.Value = 0
                Out_Gen_Class = int(99)
                if WgtsTblRow:
                    #gp.AddMessage("Processing no-confidence rows...")
                    while WgtsTblRow:
                        #gp.AddMessage("Class="+str(WgtsTblRow.Class))
        ##            
        ##                Missing data row should be processed after Outside classes are complete.
        ##                Then MD row should be found. If found, get area and num points of
        ##                Outside classes and compute MD std.
        ##            
                        if WgtsTblRow.Class != MissingDataValue:
                        #Process Out Rows for total TPs=Out_NumTPs, total Area=Out_Area, number=Out_Num
                        #Categorical might have a Class.Value = 0, therefore
                        #Give Outside generalized class a value=10^n + 99, some n >= 0...
                            if WgtsTblRow.Class >= Out_Gen_Class: Out_Gen_Class += 100
                            Out_NumTPs += WgtsTblRow.no_points
                            Out_Area += WgtsTblRow.Area
                            Out_Num = Out_Num + 1
                        WgtsTblRow = WgtsTblRows.Next()
          
                    #Calculate Wts from Out Area and Out TPs for combined Out Rows
                    if Out_Num>0:
                        if Out_NumTPs == 0: Out_NumTPs = 0.001
                        Wts = MakeWts(float(Out_NumTPs), Out_Area, Unitarea, totalTPs, totalArea, Type)
                        if not Wts:
                            gp.AddError("Weights calculation aborted.")
                        #raise ErrorExit
                    #gp.AddMessage("Num Out TPs=%d, Area Out Rows=%f: %f, %f"%(Out_NumTPs,Out_Area,Wts[0],Wts[1]))
                    #gp.AddMessage("Got wts stats." + str(Wts))
                    #At,Aj,Adt,Adj = studyArea/Unit,float(ds),Out_Area/fac/Unit,float(Out_NumTPs)
                #<<<<<<<<<<<<<<<<<Out Rows<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                #Select all rows and Set new field values
                WgtsTblRows = gp.UpdateCursor(wtstable)
                WgtsTblRow = WgtsTblRows.Next()
                In_Num = 0
                while WgtsTblRow:
                    if WgtsTblRow.Class == MissingDataValue:
                        WgtsTblRow.Gen_Class = MissingDataValue
                        WgtsTblRow.Weight = 0.0
                        WgtsTblRow.W_Std = 0.0
                        #gp.AddMessage('got md std....')
                    elif abs(WgtsTblRow.STUD_CNT) >= Confident_Contrast: #In Rows
                        WgtsTblRow.Gen_Class = WgtsTblRow.Class
                        WgtsTblRow.Weight = WgtsTblRow.Wplus
                        WgtsTblRow.W_Std = WgtsTblRow.S_Wplus
                        In_Num += 1
                    elif Out_Num > 0: #Out Rows
                        if WgtsTblRow.Class == Out_Gen_Class:
                            gp.AddError("Categorical: Class value of the outside generalized class is same as an inside class.")
                            raise ErrorExit
                        WgtsTblRow.Gen_Class = Out_Gen_Class
                        WgtsTblRow.Weight = Wts[2]
                        WgtsTblRow.W_Std = Wts[3]
                    WgtsTblRows.UpdateRow(WgtsTblRow)
                    #gp.AddMessage("Class=" + str(WgtsTblRow.Class))
                    WgtsTblRow = WgtsTblRows.Next()
                if In_Num == 0:
                    gp.AddWarning("No row Contrast for type %s satisfied the user confidence contrast = %s"%(Type,Confident_Contrast))
                    gp.AddWarning("Table %s is incomplete."%wtstable)
                    Success = False  # Invalid Table: fails confidence test
        #end of Categorical generalization
        else: #Type is Unique
            #gp.AddMessage("Setting Unique Generalization")
            WgtsTblRows = gp.UpdateCursor(wtstable)
            WgtsTblRow = WgtsTblRows.Next()
            while WgtsTblRow:
                WgtsTblRow.Gen_Class = WgtsTblRow.Class
                WgtsTblRow.Weight = 0.0
                WgtsTblRow.W_Std = 0.0
                WgtsTblRows.UpdateRow(WgtsTblRow)
                #gp.AddMessage("Class=" + str(WgtsTblRow.Class))
                WgtsTblRow = WgtsTblRows.Next()
        del WgtsTblRow, WgtsTblRows
        gp.AddMessage("Done creating table.")
        gp.AddMessage("Success: %s"%str(Success))
     #Delete extraneous fields
        gp.DeleteField_management(wtstable, "area;areaunits;count;rastervalu;frequency;sum_raster")
     #Set Output Parameter
        gp.SetParameterAsText(4, gp.Describe(wtstable).CatalogPath)
        arcpy.AddMessage("Setting success parameter..")
        arcpy.SetParameterAsText(8, Success)

    except ErrorExit:
        Success = False  # Invalid Table: Error
        gp.SetParameterAsText(8, Success)
        print ('Aborting wts calculation')
    except arcpy.ExecuteError as e:
        #TODO: Clean up all these execute errors in final version
        arcpy.AddError("\n");
        arcpy.AddMessage("Calculate weights caught arcpy.ExecuteError: ");
        if (len(e.args) > 0):
            args = e.args[0];
            args.split('\n')
            arcpy.AddError(args);
                    
        arcpy.AddMessage("-------------- END EXECUTION ---------------");        
        raise arcpy.ExecuteError;   
        
    except Exception as msg:
        # get the traceback object
        import sys;
        import traceback;
        gp.AddMessage(msg);
        errors = gp.GetMessages(2);
        
        # generate a message string for any geoprocessing tool errors
        msgs = "\n\nCW - GP ERRORS:\n" + gp.GetMessages(2) + "\n"
        gp.AddMessage("GPMEs: " + str(len(errors)) + " " + gp.GetMessages(2));
        if (len(errors) > 0):
            gp.AddError(msgs)
        
        tb = sys.exc_info()[2]
        
        # tbinfo contains the line number that the code failed on and the code from that line
        tbinfo = traceback.format_tb(tb)[0]
        # concatenate information together concerning the error into a message string
        pymsg = "CW - PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
            str(traceback.format_exc)+ "\n" #+  : " + str(sys.exc_value) + "\n"
        
        # return gp messages for use with a script tool
        if (len(errors) < 1):
            gp.AddError(pymsg)

        # print messages for use in Python/PythonWin
        print (pymsg)
        print (msgs)
        raise