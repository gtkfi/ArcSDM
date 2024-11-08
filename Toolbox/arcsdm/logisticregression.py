# -*- coding: utf-8 -*-
"""

      
    Logistic Regression for two or more evidence layers.
    Converted for ArcSDM 5 - ArcGis PRO
    2016-2018 Tero Rönkkö / GTK
    Updated by Arianne Ford, Kenex Ltd. 2018 - bug fixes for 10.x, allowing ascending and descending types for evidence.

    History: 
    5-6.10.2020 modifications by Arto Laiho, GTK/GFS
    - os.path.basename(Input_Raster[:9]) changed to os.path.basename(Input_Raster)[:9]
    - #gp.Lookup_sa(Temp_Raster, "GEN_CLASS", Output_Raster) changed to Output_Raster = gp.Lookup_sa(Temp_Raster, "GEN_CLASS")
    - Fortran application sdmlr.exe don't work if length of File System Scratch Workspace name is more than 51 characters
    - Evidential Theme to LRcoeff truncated if more than 256 characters
    - If using GDB database, remove numbers and underscore from the beginning of Output_Raster name
    21.7.2020 combined with Unicamp fixes (made 25.10.2018) / Arto Laiho, GTK/GFS
    12.6.2020 gp.JoinField_management and gp.Combine don't work on ArcGIS Pro 2.5 with File System workspace but works on V2.6 #AL 120620,140820
    25.9.2018 Merged changes from https://github.com/gtkfi/ArcSDM/issues/103 by https://github.com/Eliasmgprado
    
    Spatial Data Modeller for ESRI* ArcGIS 9.3
    Copyright 2009
    Gary L Raines, Reno, NV, USA: production and certification
    Don L Sawatzky, Spokane, WA, USA: Python software development

"""
import sys, string, os, math, tempfile, traceback, operator, importlib
import arcpy
import arcpy.management
import arcpy.sa
from arcsdm import sdmvalues
from arcsdm import workarounds_93
from arcsdm.floatingrasterarray import FloatRasterSearchcursor
import arcsdm.common
from arcpy.sa import *

PY2 = sys.version_info[0] == 2
PY34 = sys.version_info[0:2] >= (3, 4)

if PY2:
    from imp import reload;
if PY34:
    import importlib


    
debuglevel = 0;

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

# TODO: This should be in external file - like all other common things TR 
def CheckEnvironment():
    arcpy.AddMessage("Checking environment...");
    #arcpy.AddMessage('Cell size:{}'.format(arcpy.env.cellSize));
    if (arcpy.env.cellSize == 'MAXOF'):
        arcpy.AddError("  Cellsize must not be MAXOF! Set integer value to Cell Size on Environment settings."); #AL 100620
        raise arcpy.ExecuteError;


def Execute(self, parameters, messages):
    if PY2:
        reload(arcsdm.common)
    if PY34:
        importlib.reload(arcsdm.common)
    
    # Check out any necessary licenses
    arcpy.CheckOutExtension("spatial")

    # Logistic Regression don't work on ArcGIS Pro 2.5 when workspace is File System but works on V2.6! #AL 140820
    desc = arcpy.Describe(arcpy.env.workspace)
    install_version=str(arcpy.GetInstallInfo()['Version'])
    if str(arcpy.GetInstallInfo()['ProductName']) == "ArcGISPro" and install_version <= "2.5" and desc.workspaceType == "FileSystem":
        arcpy.AddError ("ERROR: Logistic Regression don't work on ArcGIS Pro " + install_version + " when workspace is File System!")
        raise

    # Fortran application sdmlr.exe don't work if length of File System Scratch Workspace name is more than 51 characters #AL 051020
    if desc.workspaceType == "FileSystem" and len(os.path.abspath(arcpy.env.scratchWorkspace)) > 51:
        arcpy.AddError ("ERROR: File System Scratch Workspace name length cannot be more than 51 characters")
        raise

    arcpy.env.overwriteOutput = True
    arcpy.env.addOutputsToMap = True

    # Load required toolboxes...

    # Script arguments...
    try:
        unitCell = parameters[5].value
        CheckEnvironment();
        if unitCell < (float(arcpy.env.cellSize)/1000.0)**2:
            unitCell = (float(arcpy.env.cellSize)/1000.0)**2
            arcpy.AddWarning('Unit Cell area is less than area of Study Area cells.\n'+
                        'Setting Unit Cell to area of study area cells: %.0f sq km.'%unitCell)

        #Get evidence layer names
        Input_Rasters = parameters[0].valueAsText.split(';')
        #Remove group layer names 
        for i, s in enumerate(Input_Rasters):
            Input_Rasters[i] = s.strip("'"); #arcpy.Describe( s.strip("'")).file;
            dwrite (arcpy.Describe( s.strip("'")).file);
            dwrite (Input_Rasters[i]);
            #if arcsdm.common.testandwarn_filegeodatabase_source(s):    #AL removed 260520
            #    return;
        arcpy.AddMessage("Input rasters: " + str(Input_Rasters))    # Unicamp added 251018 (AL 210720)

        #Get evidence layer types
        Evidence_types = parameters[1].valueAsText.lower().split(';')
        arcpy.AddMessage('Evidence_types: %s'%(str(Evidence_types)))
        if len(Evidence_types) != len(Input_Rasters):
            arcpy.AddError("Not enough Evidence types!")
            raise Exception
        for evtype in Evidence_types:
            if not evtype[0] in 'ofcad':
                arcpy.AddError("Incorrect Evidence type: %s"%evtype)
                raise Exception
        #Get weights tables names
        Wts_Tables = parameters[2].valueAsText.split(';')
        arcpy.AddMessage('Wts_Tables: %s'%(str(Wts_Tables)))
        for i, s in enumerate(Wts_Tables):                  
            arcpy.AddMessage(s);
            #if arcsdm.common.testandwarn_filegeodatabase_source(s):  #AL removed 190520
            #    return;
        if len(Wts_Tables) != len(Wts_Tables):
            arcpy.AddError("Not enough weights tables!")
            raise Exception
        #Get Training sites feature layer
        TrainPts = parameters[3].valueAsText
        arcpy.AddMessage('TrainPts: %s'%(str(TrainPts)))
        #Get missing data values
        MissingDataValue = parameters[4].valueAsText
        lstMD = [MissingDataValue for ras in Input_Rasters]
        arcpy.AddMessage('MissingDataValue: %s'%(str(MissingDataValue)))
        #Get output raster name
        thmUC = arcpy.CreateScratchName("tmp_UCras", '', 'raster', arcpy.env.scratchWorkspace)

        #Print out SDM environmental values
        #sdmvalues.appendSDMValues(unitCell, TrainPts)
        #Create Generalized Class tables
        Wts_Rasters = []
        mdidx = 0
        arcpy.AddMessage("Creating Generalized Class rasters.")
        for Input_Raster, Wts_Table in zip(Input_Rasters, Wts_Tables):
            arcpy.AddMessage("%-20s %s" % ("Input_Raster: ", str(Input_Raster)))
            Output_Raster = arcpy.CreateScratchName(os.path.basename(Input_Raster)[:9] + "_G", '', 'raster', arcpy.env.scratchWorkspace)  #AL 051020
            arcpy.AddMessage("%-20s %s" % ("Output_Raster: ", str(Output_Raster)))

            # If using GDB database, remove numbers and underscore from the beginning of the Output_Raster #AL 061020
            if desc.workspaceType != "FileSystem":
                outbase = os.path.basename(Output_Raster)
            while len(outbase) > 0 and (outbase[:1] <= "9" or outbase[:1] == "_"):
                outbase = outbase[1:]
            Output_Raster = os.path.dirname(Output_Raster) + "\\" + outbase

            arcpy.env.snapRaster = Input_Raster
            out_bands_raster = arcpy.ia.ExtractBand(Input_Raster, [1])
            arcpy.AddWarning(f'out_bands_raster: {out_bands_raster}')
            
            #Temp_Raster = arcpy.CreateScratchName('tmp_rst', '', 'raster', arcpy.env.scratchWorkspace)
            #arcpy.AddWarning(f'arcpy.env.workspace: {arcpy.env.workspace}')
            #arcpy.management.MakeRasterLayer(Input_Raster, 'temp_raster')
            #temp_raster_path = arcpy.Describe('temp_raster').catalogPath
            #arcpy.AddWarning(f'Temp_Raster: {temp_raster_path}')
            
            #arcpy.management.AddIndex(Input_Raster, [field.name for field in arcpy.ListFields(Input_Raster)], 'Value_Index', 'UNIQUE')
            #arcpy.management.CopyRaster(Input_Raster, temp_raster_path) # Input_Raster & Temp_Raster: Rowid, VALUE, COUNT
            #arcpy.AddJoin_management(Input_Raster, 'Value_Index', Wts_Table, 'CLASS')
            gdb_path = arcpy.env.workspace
            feature_class = Input_Raster
            table_to_join = f"{gdb_path}\\{Wts_Table}"
            # Specify the fields to join on
            join_field_fc = "Value"  # Field in the feature class
            join_field_table = "CLASS"  # Field in the table
            # Create a new field in the Input_Raster to store the joined values
            joined_field = "CLASS"
            if not arcpy.ListFields(Input_Raster, joined_field):
                arcpy.AddField_management(Input_Raster, joined_field)

            # Create a dictionary to store the join values from Wts_Table
            join_values = {}
            with arcpy.da.SearchCursor(Wts_Table, [join_field_table, "GEN_CLASS"]) as cursor:
                for row in cursor:
                    join_values[row[0]] = row[1]

            # Update the Input_Raster with the joined values
            with arcpy.da.UpdateCursor(Input_Raster, [join_field_fc, joined_field]) as cursor:
                for row in cursor:
                    if row[0] in join_values:
                        row[1] = join_values[row[0]]
                    else:
                        row[1] = None  # or some default value for missing join
                    cursor.updateRow(row)
            arcpy.AddMessage("%-20s %s (%d)" % ("Temp_Raster:", str(Input_Raster), int(arcpy.GetCount_management(Input_Raster).getOutput(0))))
            Output_Raster = arcpy.sa.Lookup(Input_Raster, "GEN_CLASS") #AL 051020
            arcpy.AddMessage("%-20s %s (%d)" % ("Output_Raster:", str(Output_Raster), int(arcpy.GetCount_management(Output_Raster).getOutput(0))))
            if not arcpy.Exists(Output_Raster):
                arcpy.AddError(Output_Raster + " does not exist.")
                raise Exception
            Wts_Rasters.append(arcpy.Describe(Output_Raster).catalogPath)

        #Create the Unique Conditions raster from Generalized Class rasters
        Input_Combine_rasters = ";".join(Wts_Rasters)
        arcpy.AddMessage('Combining...%s'%Input_Combine_rasters) #AL fixed Input_rasters to Input_Combine_rasters
        if arcpy.Exists(thmUC): arcpy.Delete_management(thmUC)
        thmUC = arcpy.sa.Combine(Input_Combine_rasters)    #AL changed 280520
        arcpy.AddMessage('Combine done successfully.')

        #Get UC lists from combined raster
        UCOIDname = arcpy.Describe(thmUC).OIDFieldName
        #First get names of evidence fields in UC raster
        evflds = []
        ucflds = arcpy.ListFields(thmUC)
        for ucfld in ucflds:
            if (ucfld.name == UCOIDname) or (ucfld.name.upper() in ('VALUE', 'COUNT')):
                pass
            else:
                evflds.append(ucfld.name)
        lstsVals = [[] for fld in evflds]
        cellSize = float(arcpy.env.cellSize)
        lstAreas = [[] for fld in evflds]
        if arcpy.Describe(thmUC).dataType == 'RasterLayer':
            thmUCRL = arcpy.Describe(thmUC).catalogPath
        else:
            thmUCRL = thmUC
        with arcpy.da.SearchCursor(thmUCRL, evflds + ['COUNT']) as ucrows:
            for ucrow in ucrows:
                for i, fld in enumerate(evflds):
                    lstsVals[i].append(ucrow[i])
                    lstAreas[i].append(ucrow[-1] * cellSize * cellSize / (1000000.0 * unitCell))

        #Check Maximum area of conditions so not to exceed 100,000 unit areas
        #This is a limitation of Logistic Regression: sdmlr.exe
        maxArea = max(lstAreas[0])
        if (maxArea/unitCell)/100000.0 > 1:
            unitCell = math.ceil(maxArea/100000.0)
            arcpy.AddWarning('UnitCell is set to minimum %.0f sq. km. to avoid area limit in Logistic Regression!'%unitCell)

        #Get Number of Training Sites per UC Value
        #ExtrTrainPts = workarounds_93.ExtractValuesToPoints(thmUC, TrainPts, "TPFID")
        siteFIDName = 'TPFID'
        if siteFIDName not in [field.name for field in arcpy.ListFields(TrainPts)]:
            arcpy.management.AddField(TrainPts, siteFIDName, 'LONG')            
            
        idfield = workarounds_93.GetIDField(TrainPts); 
        arcpy.CalculateField_management(TrainPts, siteFIDName, "!{}!".format(idfield))

        tempExtrShp = arcpy.CreateScratchName('Extr', 'Tmp', 'shapefile', arcpy.env.scratchWorkspace)

        arcpy.sa.ExtractValuesToPoints(TrainPts, thmUC, tempExtrShp)
        #Make dictionary of Counts of Points per RasterValue
        CntsPerRasValu = {}
        with arcpy.da.SearchCursor(ExtrTrainPts, ['RasterValu']) as tpFeats:
            for tpFeat in tpFeats:
                if tpFeat[0] in CntsPerRasValu.keys():
                    CntsPerRasValu[tpFeat[0]] += 1
                else:
                    CntsPerRasValu[tpFeat[0]] = 1
        # Make Number of Points list in RasterValue order
        # Some rastervalues can have no points in them
        lstPnts = []
        numUC = len(lstsVals[0])
        for i in range(1, numUC + 1):  # Combined raster values start at 1
            if i in CntsPerRasValu.keys():
                lstPnts.append(CntsPerRasValu.get(i))
            else:
                lstPnts.append(0)
        lstsMC = []
        mcIndeces = []
        for et in Evidence_types:
            if et.startswith('o') or et.startswith('a') or et.startswith('d'):
                mcIndeces.append(-1)
            elif et.startswith('f') or et.startswith('c'):
                mcIndeces.append(1)
            else:
                arcpy.AddError('Incorrect evidence type')
            raise Exception
        if len(mcIndeces) != len(Input_Rasters):
            arcpy.AddError("Incorrect number of evidence types.")
            raise Exception
        # arcpy.AddMessage('mcIndeces: %s'%(str(mcIndeces)))
        catMCLists = [[], mcIndeces]
        evidx = 0
        for mcIdx in mcIndeces:
            catMCLists[0].append([])
            if mcIdx < 0:
                pass
            else:
                # Make a list of free raster values
                wts_g = arcpy.CreateScratchName("Wts_G")
                arcpy.MakeRasterLayer_management(Wts_Rasters[evidx], wts_g)
                with arcpy.da.SearchCursor(wts_g, ["Value"]) as evrows:
                    for evrow in evrows:
                        if evrow[0] not in catMCLists[0][evidx]:
                            catMCLists[0][evidx].append(evrow[0])
            evidx += 1
        # arcpy.AddMessage('catMCLists: %s'%(catMCLists))
        lstWA = CalcVals4Msng(lstsVals, lstAreas[0], lstMD, catMCLists)
        # arcpy.AddMessage('lstWA: %s'%(str(lstWA)))
        ot = [['%s, %s' % (Input_Rasters[i], Wts_Tables[i])] for i in range(len(Input_Rasters))]
        # arcpy.AddMessage("ot=%s"%ot)
        strF2 = "case.dat"
        fnCase = os.path.join(arcpy.env.scratchFolder, strF2)
        fCase = open(fnCase, 'w')
        if not fCase:
            arcpy.AddError("Can't create 'case.dat'.")
            raise Exception
        nmbUC = len(lstsVals[0])
        getNmbET = True  # True when first line of case.dat
        nmbET = 0  # Number of ET values in a line of case.dat
        arcpy.AddMessage("Writing Logistic Regression input files...")
        ''' Reformat the labels for free evidence '''
        for j in range(len(lstsVals)):
            mcIdx = mcIndeces[j]
            if mcIdx > -1:
                listVals = catMCLists[0][j]
                lstLV = listVals[:]
                lstLV = RemoveDuplicates(lstLV)
                elOT = ot[j]
                tknTF = elOT[0].split(',')
                strT = tknTF[0].strip()
                strF = tknTF[1].strip()
                first = True
                for lv in lstLV:
                    if lv == lstMD[j]:
                        continue
                    if first:
                        elOT = ["%s (%s)" % (elOT[0], lv)]
                        first = False
                    else:
                        elOT.append("%s, %s (%s)" % (strT, strF, lv))
            ot[j] = elOT
        # arcpy.AddMessage('ot=%s'%(str(ot)))
        ''' Loop through the unique conditions, substituting
            the weighted average of known classes for missing data
            and 'expanding' multi-class free data themes to
            a series of binary themes '''
        #gp.AddMessage('lstWA: %s'%lstWA)
        for i in range(nmbUC):
            numPoints = lstPnts[i]
        ##        #>>> This is a kluge for problem in case.dat for sdmlr.exe
        ##        if numPoints == 0: continue
        ##        #Fractional numpoints is not accepted
        ##        #This means that UC area had no extracted points,
        ##        #and should not be a case here.
        ##        #<<< End kluge for case.dat
            wLine = ""
            wLine = wLine + ('%-10d'%(i+1))
            j = 0
            for lst in lstsVals:
                missing = lstMD[j]
                theVal = lst[i]
                mcIdx = mcIndeces[j]
            if mcIdx < 0: #ordered evidence
                if getNmbET: nmbET = nmbET + 1
                if theVal == missing:
                    theVal = lstWA[j][0] #avgweighted
                    wLine = wLine + '%-20s'%theVal
            else: #free evidence
                listVals = catMCLists[0][j]
                #arcpy.AddMessage('catMCLists[%d]: %s'%(j, catMCLists[0][j]))
                OFF = 0
                ON = 1
                if theVal == missing:
                    m=0
                for v in listVals:
                    if v == missing:
                        continue
                    else:
                        #arcpy.AddMessage('lstWA[%d][%d]=%s'%(j, m, lstWA[j]))
                        valWA = lstWA[j][m]
                        wLine = wLine + '%-20s'%valWA
                        m += 1
                        if getNmbET: nmbET += 1
                else:
                    for v in listVals:
                        if v == missing:
                            continue
                        elif getNmbET: nmbET += 1
                        if theVal == v:
                            wLine = wLine + '%-20s'%ON
                        else:
                            wLine = wLine + '%-20s'%OFF
            j += 1
            wLine = wLine + '%-10d'%numPoints
            theArea = lstAreas[0][i] / unitCell
            wLine = wLine + '%-20s' %theArea
            fCase.write(wLine + '\n')
            getNmbET = False
        fCase.close()
        ##' Write a parameter file to the ArcView extension directory
        ##'----------------------------------------------
        arcpy.AddMessage("Write a parameter file to the ArcView extension directory")
        strF1 = "param.dat"
        fnParam = os.path.join(arcpy.env.scratchFolder, strF1) #param.dat file
        fParam = open(fnParam, 'w')
        if not fParam:
            arcpy.AddError("Error writing logistic regression parameter file.")
            raise Exception
        fParam.write('%s\\\n' %(arcpy.env.scratchFolder))
        fParam.write('%s\n' %strF2)
        fParam.write("%d %g\n" %(nmbET, unitCell))
        fParam.close()

        ### RunLR ------------------------------------------------------------------------------------------------------------
        # Check input files
        # Check input files exist
        Paramfile = os.path.join(arcpy.env.scratchFolder, 'param.dat')
        if arcpy.Exists(Paramfile):
            pass
        else:
            arcpy.AddError("Logistic regression parameter file does not exist: %s" % Paramfile)
            raise Exception

        # Place input files folder in batch file
        # sdmlr.exe starts in input files folder.
        sdmlr = os.path.join(sys.path[0], 'bin', 'sdmlr.exe')
        if not os.path.exists(sdmlr):
            arcpy.AddError("Logistic regression executable file does not exist: %s" % sdmlr)
            raise Exception

        os.chdir(arcpy.env.scratchFolder)
        if os.path.exists('logpol.tba'): os.remove('logpol.tba')
        if os.path.exists('logpol.out'): os.remove('logpol.out')
        if os.path.exists('cumfre.tba'): os.remove('cumfre.tba')
        if os.path.exists('logco.dat'): os.remove('logco.dat')

        fnBat = os.path.join(arcpy.env.scratchFolder, 'sdmlr.bat')
        fBat = open(fnBat, 'w')
        fBat.write("%s\n" % os.path.splitdrive(arcpy.env.scratchFolder)[0])
        fBat.write("CD %s\n" % os.path.splitdrive(arcpy.env.scratchFolder)[1])
        fBat.write('"%s"\n' % sdmlr)
        fBat.close()

        params = []
        try:
            import subprocess
            p = subprocess.Popen([fnBat, params]).wait()
            arcpy.AddMessage('Running %s: ' % fnBat)
        except OSError:
            arcpy.AddMessage('Execution failed %s: ' % fnBat)

        if not os.path.exists('logpol.tba'):
            arcpy.AddError("Logistic regression output file %s\\logpol.tba does not exist.\n Error in case.dat or param.dat. " % arcpy.env.scratchFolder)
            raise Exception

        arcpy.AddMessage("Finished running Logistic Regression")

    ###ReadLRResults -------------------------------------------------------------------------------------------------------

        thmuc = thmUC
        vTabUC = 'thmuc_lr'
        arcpy.MakeRasterLayer_management(thmuc, vTabUC)
        strFN = "logpol.tba"
        strFnLR = os.path.join(arcpy.env.scratchFolder, strFN)

        if not arcpy.Exists(strFnLR):
            arcpy.AddError("Reading Logistic Regression Results\nCould not find file: %s" % strFnLR)
            raise Exception('Existence error')
        arcpy.AddMessage("Opening Logistic Regression Results: %s" % strFnLR)
        fLR = open(strFnLR, "r")
        if not fLR:
            arcpy.AddError("Input Error - Unable to open the file: %s for reading." % strFnLR)
            raise Exception('Open error')
        read = 0
        fnNew = parameters[6].valueAsText
        tblbn = os.path.basename(fnNew)
        [tbldir, tblfn] = os.path.split(fnNew)
        if tbldir.endswith(".gdb"):
            tblfn = tblfn[:-4] if tblfn.endswith(".dbf") else tblfn
            fnNew = fnNew[:-4] if fnNew.endswith(".dbf") else fnNew
            tblbn = tblbn[:-4] if tblbn.endswith(".dbf") else tblbn
        arcpy.AddMessage("fnNew: %s" % fnNew)
        arcpy.AddMessage('Making table to hold logistic regression results (param 6): %s' % fnNew)
        fnNew = tblbn
        arcpy.CreateTable_management(tbldir, tblfn)
        arcpy.AddMessage('Making table to hold logistic regression results: %s' % fnNew)
        fnNew = os.path.join(tbldir, fnNew)

        arcpy.AddField_management(fnNew, 'ID', 'LONG', 6)
        arcpy.AddField_management(fnNew, 'LRPostProb', 'Double', "#", "#", "#", "LR_Posterior_Probability")
        arcpy.AddField_management(fnNew, 'LR_Std_Dev', 'Double', "#", "#", "#", "LR_Standard_Deviation")
        arcpy.AddField_management(fnNew, 'LRTValue', 'Double', "#", "#", "#", "LR_TValue")
        arcpy.DeleteField_management(fnNew, "Field1")
        vTabLR = fnNew
        strLine = fLR.readline()
        arcpy.AddMessage("Reading Logistic Regression Results: %s" % strFnLR)
        vTabLRrows = arcpy.da.InsertCursor(vTabLR, ["ID", "LRPostProb", "LR_Std_Dev", "LRTValue"])
        while strLine:
            if strLine.strip() == 'DATA':
                read = 1
            elif read:
                lstLine = strLine.split()
            if len(lstLine) > 5:
                vTabLRrows.insertRow([int(lstLine[1].strip()), float(lstLine[3].strip()), float(lstLine[5].strip()), float(lstLine[4].strip())])
            strLine = fLR.readline()
        fLR.close()
        del vTabLRrows
        arcpy.AddMessage('Created table to hold logistic regression results: %s' % fnNew)
        ##' Get the coefficients file
        ##'----------------------------------------------
        strFN2 = "logco.dat"
        fnLR2 = os.path.join(arcpy.env.scratchFolder, strFN2)
        ##  ' Open file for reading
        ##  '----------------------------------------------
        arcpy.AddMessage("Opening Logistic Regression coefficients Results: %s" % fnLR2)
        fLR2 = open(fnLR2, "r")
        read = 0
        ##  ' Expand object tag list of theme, field, value combos
        ##  '----------------------------------------------
        arcpy.AddMessage('Expanding object tag list of theme, field, value combos')
        lstLabels = []
        for el in ot:
            for e in el:
                lstLabels.append(e.replace(' ', ''))
        ##  ' Make vtab to hold theme coefficients
        ##  '----------------------------------------------
        fnNew2 = parameters[7].valueAsText
        tblbn = os.path.basename(fnNew2)
        [tbldir, tblfn] = os.path.split(fnNew2)
        if tbldir.endswith(".gdb"):
            tblfn = tblfn[:-4] if tblfn.endswith(".dbf") else tblfn
            fnNew2 = fnNew2[:-4] if fnNew2.endswith(".dbf") else fnNew2
            tblbn = tblbn[:-4] if tblbn.endswith(".dbf") else tblbn
        fnNew2 = tblbn
        arcpy.AddMessage("Making table to hold theme coefficients: " + fnNew2)
        arcpy.CreateTable_management(tbldir, tblfn)
        arcpy.AddField_management(fnNew2, "Theme_ID", 'LONG', 6, "#", "#", "Theme_ID")
        arcpy.AddField_management(fnNew2, "Theme", 'TEXT', "#", "#", 256, "Evidential_Theme")
        arcpy.AddField_management(fnNew2, "Coeff", 'DOUBLE', "#", "#", "#", 'Coefficient')
        arcpy.AddField_management(fnNew2, "LR_Std_Dev", 'DOUBLE', "#", "#", "#", "LR_Standard_Deviation")
        arcpy.DeleteField_management(fnNew2, "Field1")
        vTabLR2 = fnNew2
        strLine = fLR2.readline()
        i = 0
        first = 1
        arcpy.AddMessage("Reading Logistic Regression Coefficients Results: %s" % fnLR2)
        vTabLR2rows = arcpy.da.InsertCursor(vTabLR2, ["Theme_ID", "Theme", "Coeff", "LR_Std_Dev"])
        while strLine:
            if len(strLine.split()) > 1:
                if strLine.split()[0].strip() == 'pattern':
                    read = 1
                    strLine = fLR2.readline()
                    continue
            if read:
                lstLine = strLine.split()
                if len(lstLine) > 2:
                    vTabLR2row = vTabLR2rows.insertRow()
                    vTabLR2row[0] = int(lstLine[0].strip()) + 1
                    if not first:
                        try:
                            lbl = lstLabels.pop(0)
                            if len(lbl) > 256:
                                lbl = lbl[:256]
                            vTabLR2row[1] = lbl
                        except IndexError:
                            arcpy.AddError('Evidence info %s not consistent with %s file' % (otfile, fnLR2))
                        i += 1
                    else:
                        vTabLR2row[1] = "Constant Value"
                        first = 0
                    vTabLR2row[2] = float(lstLine[1].strip())
                    vTabLR2row[3] = float(lstLine[2].strip())
                    vTabLR2rows.insertRow(vTabLR2row)
                else:
                    break
            strLine = fLR2.readline()
        fLR2.close()
        if len(lstLabels) != 0:
            arcpy.AddError('Evidence info %s not consistent with %s file' % (otfile, fnLR2))
        del vTabLR2row, vTabLR2rows
        arcpy.AddMessage('Created table to hold theme coefficients: %s' % fnNew2)
        vTabLR2rows = arcpy.da.InsertCursor(vTabLR2, ["Theme_ID", "Theme", "Coeff", "LR_Std_Dev"])
        print("Starting to read LR_Coeff")
        while strLine:
            print("Rdr:", strLine)
            if len(strLine.split()) > 1:
                if strLine.split()[0].strip() == 'pattern':
                    read = 1
                    strLine = fLR2.readline()
                    continue
            if read:
                lstLine = strLine.split()
                if len(lstLine) > 2:
                    vTabLR2row = [None] * 4
                    vTabLR2row[0] = int(lstLine[0].strip()) + 1
                    if not first:
                        try:
                            lbl = lstLabels.pop(0)
                            if len(lbl) > 256:
                                lbl = lbl[:256]
                            print("Lbl:", lbl)
                            vTabLR2row[1] = lbl
                        except IndexError:
                            arcpy.AddError('Evidence info %s not consistent with %s file' % (otfile, fnLR2))
                        i += 1
                    else:
                        vTabLR2row[1] = "Constant Value"
                        first = 0
                    print("Coeff:", lstLine[1].strip())
                    vTabLR2row[2] = float(lstLine[1].strip())
                    print("LR_std_dev:", lstLine[2].strip())
                    vTabLR2row[3] = float(lstLine[2].strip())
                    vTabLR2rows.insertRow(vTabLR2row)
            else:
                break
        strLine = fLR2.readline()
        fLR2.close()
        if len(lstLabels) != 0:
            arcpy.AddError('Evidence info %s not consistent with %s file' % (otfile, fnLR2))
        del vTabLR2row, vTabLR2rows
        arcpy.AddMessage('Created table to hold theme coefficients: %s' % fnNew2)
        # Creating LR Response Rasters
        arcpy.AddMessage("Creating LR Response Rasters")
        
        # Join LR polynomial table to unique conditions raster and copy to get a raster with attributes
        cmb = thmUC
        tbl = parameters[6].valueAsText
        tbltv = 'tbltv'
        arcpy.MakeTableView_management(tbl, tbltv)
        
        cmb_cpy = arcpy.CreateScratchName("cmb_cpy", '', 'raster', arcpy.env.scratchWorkspace)
        arcpy.CopyRaster_management(cmb, cmb_cpy)
        arcpy.JoinField_management(cmb_cpy, 'Value', tbltv, 'ID')
                                      
        # Make output float rasters from attributes of joined unique conditions raster
        arcpy.AddMessage("Make output float rasters from attributes of joined unique conditions raster (param 6)")
        outRaster1 = parameters[8].valueAsText
        outRaster2 = parameters[9].valueAsText
        outRaster3 = parameters[10].valueAsText
        
        outcon1 = Con(cmb_cpy, Lookup(cmb_cpy, "LRPOSTPROB"), 0, "LRPOSTPROB > 0")
        outcon1.save(outRaster1)
        outcon2 = Con(cmb_cpy, Lookup(cmb_cpy, "LR_STD_DEV"), 0, "LR_STD_DEV > 0")
        outcon2.save(outRaster2)
        outcon3 = Con(cmb_cpy, Lookup(cmb_cpy, "LRTVALUE"), 0, "LRTVALUE > 0")
        outcon3.save(outRaster3)
        
        # Add to display
        arcpy.SetParameterAsText(6, tbl)
        arcpy.SetParameterAsText(7, arcpy.Describe(vTabLR2).catalogPath)
        arcpy.SetParameterAsText(8, outRaster1)
        arcpy.SetParameterAsText(9, outRaster2)
        arcpy.SetParameterAsText(10, outRaster3)
    except arcpy.ExecuteError as e:
        arcpy.AddError("\n");
        arcpy.AddMessage("Caught ExecuteError in logistic regression. Details:");
        args = e.args[0];
        args.split('\n')
        arcpy.AddError(args);
        # get the traceback object
        tb = sys.exc_info()[2]
         # tbinfo contains the line number that the code failed on and the code from that line
        tbinfo = traceback.format_tb(tb)[0]
        # concatenate information together concerning the error into a message string
        msgs = "Traceback\n" + tbinfo + "\nError Info:\n" + str(sys.exc_info()[1])
        arcpy.AddError(msgs)
        raise 
    except:
        # get the traceback object
        tb = sys.exc_info()[2]
         # tbinfo contains the line number that the code failed on and the code from that line
        tbinfo = traceback.format_tb(tb)[0]
        # concatenate information together concerning the error into a message string
        msgs = "Traceback\n" + tbinfo + "\nError Info:\n" + "\n".join([str(errori) for errori in sys.exc_info()])
        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + msgs + "\n"
        
        erroreita = traceback.format_exc()
        
        # generate a message string for any geoprocessing tool errors
        msgs = "GP ERRORS:\n" + arcpy.GetMessages(2) + "\n"
        arcpy.AddError(msgs)
        # return gp messages for use with a script tool
        arcpy.AddError(pymsg)
        
        arcpy.AddError(erroreita)
        # print messages for use in Python/PythonWin
        print (pymsg)
        raise
    
def RemoveDuplicates(lst):
    """ Remove duplicates without sorting """
    unique = []
    for l in lst:
        if l not in unique: unique.append(l)
    #arcpy.AddMessage('RemoveDuplicates: %s'%unique)
    return unique

def CalcWeightedAvg(lstVals, lstValsNew, lstAreas, nmbMD, sumArea, mcf):
    """Converted from Avenue script gSDI.CalcWeightedAvg
    ' gSDI.CalcWeightedAvg
    '
    ' Topics:  Spatial Data Modeller
    '
    ' Description:  Takes a list of values and
    '          areas and calculates a weighted
    '          average.  Option to identify
    '          areas for current values but to
    '          calculate weighted average based
    '          on new values.
    '
    ' Requires:
    '
    ' Self:   0 -- lstVals -- list of current values,
    '                         doesn't have to be unique
    '         1 -- lstValsNew -- list of new values, Nil if none
    '         2 -- lstAreas -- list of areas corresponding to lstVals
    '         3 -- nmbMD -- number (integer) defining missing data value
    '         4 -- sumArea -- total area of areas in lstAreas
    '         5 -- mcf -- Nil if not multi-class free theme, else list of unique class values
    ' Returns:
    '==============================================
    """
    try:
        if not mcf: #Is ordered evidence
            #Make unique list of UC values
            lUnique = lstVals[:]
            lUnique = RemoveDuplicates(lUnique)
            #For each value in list of UC values
            #increment the area associated with it
            lstAreasSums = [0]*len(lUnique) #[0 for v in lUnique]
            i = 0
            for v in lstVals:
                area =lstAreas[i]
                idx = lUnique.index(v)
                lstAreasSums[idx] += area
                i+=1
            # If the current values contain the missing data
            # integer, decrease the denominator by its corresponding
            # area and remove the areas value from the area list
            #------------------------------------------------
            numerator = 0.0
            denominator = sumArea
            if nmbMD in lUnique:
                idx = lUnique.index(nmbMD)
                denominator = 0
                for i in range(len(lstAreasSums)):
                    if i != idx:
                        denominator += lstAreasSums[i]
                del lUnique[idx]
                del lstAreasSums[idx]
            ##  ' If there are new values to use for average, loop
            ##  ' through these and calculate the average, otherwise
            ##  ' use the current values
            ##  '--------------------------------------------------
            lstWeight = lUnique #Never any new values
            #Calculate weighted average value for
            #missing data in an ordered evidence layer
            i = 0
            for v in lstWeight:
                idx = lstWeight.index(v)
                if v != nmbMD:
                    area = lstAreasSums[idx]
                    numerator += (v * area)
                else:
                    denominator -= lstAreasSums[idx]
                i += 1
            nmbWA = numerator / denominator
            return [nmbWA]

        else: #Is Multi-Class, Free evidence
            #Fill ValueSum and AreaSum lists
            lstValSum = []
            lstAreaSum = []
            #arcpy.AddMessage('Free: %s %s %s %s'%(lstVals, lstValSum, lstAreaSum, sumArea))
            i = 0
            for v in lstVals:
                if v not in lstValSum:
                    lstValSum.append(v)
                    lstAreaSum.append(lstAreas[i])
                else:
                    idxV = lstValSum.index(v)
                    lstAreaSum[idxV] += lstAreas[i]
                i += 1
            #Remove areas, vals for missing data
            #arcpy.AddMessage('Free: nmbMD: %s\nlstVals: %s\nlstValSum: %s\nlstAreaSum: %s\nsumArea: %s'%(nmbMD, lstVals, lstValSum, lstAreaSum, sumArea))
            if nmbMD in lstVals:
                idxV = lstValSum.index(nmbMD)
                del lstValSum[idxV]
                areaMD = lstAreaSum[idxV]
                sumArea -= areaMD
            #arcpy.AddMessage('Free: nmbMD: %s\nlstVals: %s\nlstValSum: %s\nlstAreaSum: %s\nsumArea: %s'%(nmbMD, lstVals, lstValSum, lstAreaSum, sumArea))
            #Generate list of wtd. averages
            lstNmbWA = []
            i = 0
            for v in lstValSum:
                vb0 = 1
                area0 = lstAreaSum[i]
                vb1 = 0
                area1 = sumArea - area0
                numerator = (vb0*area0) + (vb1*area1)
                wa = numerator / sumArea
                lstNmbWA.append(wa)
                i += 1
            #arcpy.AddMessage('lstNmbWA: %s'%(lstNmbWA))
            return lstNmbWA
    except:
        # get the traceback object
        tb = sys.exc_info()[2]
        # tbinfo contains the line number that the code failed on and the code from that line
        tbinfo = traceback.format_tb(tb)[0]
        # concatenate information together concerning the error into a message string
        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
            str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
        # generate a message string for any geoprocessing tool errors
        #msgs = "GP ERRORS:\n" + arcpy.GetMessages(2) + "\n"
        msgs = "GP ERRORS:\n" + gp.GetMessages(2) + "\n"  # Unicamp changed 251018 (AL 210720)
        gp.AddError(msgs)

        # return gp messages for use with a script tool
        gp.AddError(pymsg)

        # print messages for use in Python/PythonWin
        print (pymsg)
        print (msgs)
        raise Exception
##------------- end of definition ----------------------------------------------------------------------

def CalcVals4Msng(lstUCVals, lstAreas, lstMD, lstMCF):
    """Converted from Avenue script gSDI.CalcVals4Msng
    'Name:  gSDI.CalcVals4Msng
    '
    ' Topics:  Spatial Data Modeller
    '
    ' Description:  Calculates a weighted average for
    '               filling in areas of missing data, based
    '               on values found in the rest of the theme.
    '
    '
    '
    ' Requires:
    '
    ' Self:  List containing:
    '          0 -- list of lists of values in unique conditions table
    '          1 -- list of areas of each unique condition
    '          2 -- integer used to define missing data
    '
    ' Returns:  List of weighted averages, one for each theme
    '           in unique conditions table, to be used to
    '           fill in areas of missing data.
    '==============================================
    'Get arguments
    '----------------------------------------------
    lstUCVals = self.Get(0)    'list of lists of uc values
    lstAreas = self.Get(1)     'list of areas for each unique condition
    lstMD = self.Get(2)      'missing data values for each theme
    lstMCF = self.Get(3)    ' list of expanded values for multi-class, free data type themes
    lstThmMCF = lstMCF.Get(0)
    'list of values corresponding to list of themes, empty of theme is not multi-class free
    """
    try:
        #arcpy.AddMessage("CalcVals4Msng: lstUCVals=%s"%(lstUCVals, ))
        lstThmMCF = lstMCF[0]
        lstMCFIdx = lstMCF[1]
        #arcpy.AddMessage("CalcVals4Msng.lstMCF: %s=%s, %s"%(lstMCF, lstThmMCF, lstMCFIdx))
        #Calculate the total area of the unique conditions
        totalArea = 0
        for area in lstAreas:
            totalArea += area
        #Initialize list to hold weighted averages to be calculated
        lstWAVals = []
        #For each list of values (from a single evidence layer)
        k = 0
        for lstVals in lstUCVals:
            idxMCF = lstMCFIdx[k]
            mcf = idxMCF > -1
            lstMD_k = lstMD[k] #Get MissingDataValue for evidence layer
            nmbWA = CalcWeightedAvg(lstVals, None, lstAreas, lstMD_k, totalArea, mcf)
            lstWAVals.append(nmbWA)
            k += 1
        #arcpy.AddMessage('lstWAVals: %s'%(lstWAVals))
        return lstWAVals
    except:
        # get the traceback object
        tb = sys.exc_info()[2]
        # tbinfo contains the line number that the code failed on and the code from that line
        tbinfo = traceback.format_tb(tb)[0]
        # concatenate information together concerning the error into a message string
        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
            str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
        # generate a message string for any geoprocessing tool errors
        #msgs = "GP ERRORS:\n" + arcpy.GetMessages(2) + "\n"
        msgs = "GP ERRORS:\n" + gp.GetMessages(2) + "\n"  # Unicamp changed 251018 (AL 210720)
        arcpy.AddError(msgs)

        # return gp messages for use with a script tool
        arcpy.AddError(pymsg)

        # print messages for use in Python/PythonWin
        print (pymsg)
        print (msgs)
        raise Exception
