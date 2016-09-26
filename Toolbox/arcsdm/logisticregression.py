# -*- coding: utf-8 -*-
"""

    Logistic Regression for two or more evidence layers.
    Converted for ArcSDM 5 - ArcGis PRO
    2016 Tero Rönkkö / GTK

    Spatial Data Modeller for ESRI* ArcGIS 9.3
    Copyright 2009
    Gary L Raines, Reno, NV, USA: production and certification
    Don L Sawatzky, Spokane, WA, USA: Python software development

"""
import sys, string, os, math, tempfile, arcgisscripting, traceback, operator, importlib
import arcpy
import arcgisscripting
from arcsdm import sdmvalues
from arcsdm import workarounds_93
from  arcsdm.floatingrasterarray import FloatRasterSearchcursor

# TODO: This should be in external file - like all other common things TR 
def CheckEnvironment():
    arcpy.AddMessage("Checking environment...");
    #arcpy.AddMessage('Cell size:{}'.format(arcpy.env.cellSize));
    if (arcpy.env.cellSize == 'MAXOF'):
        arcpy.AddError("  Cellsize must not be MAXOF!");
        raise arcpy.ExecuteError;


def Execute(self, parameters, messages):

    gp = arcgisscripting.create()

    # Check out any necessary licenses
    gp.CheckOutExtension("spatial")

    gp.OverwriteOutput = 1
    gp.LogHistory = 1

    # Load required toolboxes...

    # Script arguments...
    try:
        unitCell = parameters[5].value
        CheckEnvironment();
        if unitCell < (float(gp.CellSize)/1000.0)**2:
            unitCell = (float(gp.CellSize)/1000.0)**2
            gp.AddWarning('Unit Cell area is less than area of Study Area cells.\n'+
                        'Setting Unit Cell to area of study area cells: %.0f sq km.'%unitCell)

        #Get evidence layer names
        Input_Rasters = parameters[0].valueAsText.split(';')
        #gp.AddMessage('Input_Rasters: %s'%(str(Input_Rasters)))
        #Get evidence layer types
        Evidence_types = parameters[1].valueAsText.lower().split(';')
        gp.AddMessage('Input_Rasters: %s'%(str(Input_Rasters)))
        if len(Evidence_types) != len(Input_Rasters):
            gp.AddError("Not enough Evidence types!")
            raise Exception
        for evtype in Evidence_types:
            if not evtype[0] in 'ofc':
                gp.AddError("Incorrect Evidence type: %s"%evtype)
                raise Exception
        #Get weights tables names
        Wts_Tables = parameters[2].valueAsText.split(';')
        gp.AddMessage('Wts_Tables: %s'%(str(Wts_Tables)))
        if len(Wts_Tables) != len(Wts_Tables):
            gp.AddError("Not enough weights tables!")
            raise Exception
        #Get Training sites feature layer
        TrainPts = parameters[3].valueAsText
        gp.AddMessage('TrainPts: %s'%(str(TrainPts)))
        #Get missing data values
        MissingDataValue = parameters[4].valueAsText
        lstMD = [MissingDataValue for ras in Input_Rasters]
        gp.AddMessage('MissingDataValue: %s'%(str(MissingDataValue)))
        #Get output raster name
        thmUC = gp.createscratchname("tmp_UCras", '', 'raster', arcpy.env.scratchFolder)

        #Print out SDM environmental values
        sdmvalues.appendSDMValues(gp, unitCell, TrainPts)

        #Create Generalized Class tables
        Wts_Rasters = []
        mdidx = 0
        gp.AddMessage("Creating Generalized Class rasters.")
        for Input_Raster, Wts_Table in zip(Input_Rasters, Wts_Tables):
            Output_Raster = gp.CreateScratchName(os.path.basename(Input_Raster[:9]) + "_G", '', 'raster', arcpy.env.scratchFolder)
            #gp.AddMessage('Output_Raster: %s'%(str(Output_Raster)))
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            #++ Need to create in-memory Raster Layer for AddJoin
            RasterLayer = "OutRas_lyr"
            gp.makerasterlayer(Input_Raster, RasterLayer)
            gp.AddJoin_management(RasterLayer, "Value", Wts_Table, "CLASS")
            Temp_Raster = gp.CreateScratchName('temp_ras', '', 'raster', arcpy.env.scratchFolder)
            gp.AddMessage('Temp_Raster: %s'%(str(Temp_Raster)))
            gp.CopyRaster_management(RasterLayer, Temp_Raster)
            gp.Lookup_sa(Temp_Raster, "GEN_CLASS", Output_Raster)
            gp.delete(RasterLayer)
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            #gp.AddMessage(Output_Raster + " exists: " + str(gp.Exists(Output_Raster)))
            if not gp.Exists(Output_Raster):
                gp.AddError(Output_Raster + " does not exist.")
                raise Exception
            Wts_Rasters.append(gp.Describe(Output_Raster).CatalogPath)
        #Create the Unique Conditions raster from Generalized Class rasters
    ##    #>>>> Comment out for testing >>>>>>>>>>>>>>>>>>>>>>>>>>
        Input_Combine_rasters = ";".join(Wts_Rasters)
        #Combine created Wts_Rasters and add to TOC
        #gp.AddMessage('Combining...%s'%Input_rasters)
        if gp.exists(thmUC): gp.delete_management(thmUC)
        gp.Combine_sa(Input_Combine_rasters, thmUC)
    ##    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        #gp.AddMessage('Combine done...')

        #Get UC lists from combined raster
        UCOIDname = gp.describe(thmUC).OIDfieldname
        #First get names of evidence fields in UC raster
        evflds = []
        ucflds = gp.ListFields(thmUC)
        ucfld = ucflds.Next()
        while ucfld:
            if (ucfld.Name == UCOIDname) or (ucfld.Name.upper() in ('VALUE', 'COUNT')):
            #if ucfld.Name == UCOIDname or ucfld.Name == 'Value' or ucfld.Name == 'Count':
                pass
            else:
                evflds.append(ucfld.Name)
            ucfld = ucflds.Next()
        #gp.AddMessage('evflds: %s'%str(evflds))
        #Set up empty list of lists
        lstsVals = [[] for fld in evflds]
        #gp.AddMessage('lstsVals: %s'%str(lstsVals))
        #Put UC vals and areas for each evidence layer in lstsVals
        cellSize = float(gp.CellSize)
        lstAreas = [[] for fld in evflds]
        if gp.describe(thmUC).datatype == 'RasterLayer':
            thmUCRL = gp.describe(thmUC).catalogpath
        else:
            thmUCRL = thmUC
        ucrows = workarounds_93.rowgen(gp.SearchCursor(thmUCRL))
        for ucrow in ucrows:
            for i, fld in enumerate(evflds):
                lstsVals[i].append(ucrow.GetValue(fld))
                lstAreas[i].append(ucrow.Count * cellSize * cellSize / (1000000.0 * unitCell))
        #gp.AddMessage('lstsVals: %s'%(str(lstsVals)))
        #gp.AddMessage('lstAreas: %s'%(str(lstAreas)))

        #Check Maximum area of conditions so not to exceed 100,000 unit areas
        #This is a limitation of Logistic Regression: sdmlr.exe
        maxArea = max(lstAreas[0])
        if (maxArea/unitCell)/100000.0 > 1:
            unitCell = math.ceil(maxArea/100000.0)
            gp.AddWarning('UnitCell is set to minimum %.0f sq. km. to avoid area limit in Logistic Regression!'%unitCell)

        #Get Number of Training Sites per UC Value
        #First Extract RasterValues to Training Sites feature layer
        #ExtrTrainPts = os.path.join(gp.ScratchWorkspace, "LRExtrPts.shp")
        #ExtrTrainPts = gp.CreateScratchName('LRExtrPts', 'shp', 'shapefile', gp.scratchworkspace)
        #gp.ExtractValuesToPoints_sa(TrainPts, thmUC, ExtrTrainPts, "NONE", "VALUE_ONLY")
        ExtrTrainPts = workarounds_93.ExtractValuesToPoints(gp, thmUC, TrainPts, "TPFID")
        #Make dictionary of Counts of Points per RasterValue
        CntsPerRasValu = {}
        tpFeats = workarounds_93.rowgen(gp.SearchCursor(ExtrTrainPts))
        for tpFeat in tpFeats:
            if tpFeat.RasterValu in CntsPerRasValu.keys():
                CntsPerRasValu[tpFeat.RasterValu] += 1
            else:
                CntsPerRasValu[tpFeat.RasterValu] = 1
        #gp.AddMessage('CntsPerRasValu: %s'%(str(CntsPerRasValu)))
        #Make Number of Points list in RasterValue order
        #Some rastervalues can have no points in them
        lstPnts = []
        numUC = len(lstsVals[0])
        for i in range(1, numUC+1): #Combined raster values start at 1
            if i in CntsPerRasValu.keys():
                lstPnts.append(CntsPerRasValu.get(i))
            else:
                lstPnts.append(0)
        #gp.AddMessage('lstPnts: %s'%(lstPnts))
        lstsMC = []
        mcIndeces = []
        for et in Evidence_types:
            if et.startswith('o'):
                mcIndeces.append(-1)
            elif et.startswith('f') or et.startswith('c'):
                mcIndeces.append(1)
            else:
                gp.AddError('Incorrect evidence type')
                raise Exception
        if len(mcIndeces) != len(Input_Rasters):
            gp.AddError("Incorrect number of evidence types.")
            raise Exception
        #gp.AddMessage('mcIndeces: %s'%(str(mcIndeces)))
        catMCLists = [[], mcIndeces]
        evidx = 0
        for mcIdx in mcIndeces:
            catMCLists[0].append([])
            if mcIdx<0:
                pass
            else:
                #Make a list of free raster values
                #evidx = len(catMCLists[0]) - 1
                #gp.AddMessage(Wts_Rasters[evidx])
                wts_g = gp.createscratchname("Wts_G")
                gp.MakeRasterLayer_management(Wts_Rasters[evidx], wts_g)
                #evrows = gp.SearchCursor("Wts_G")
                evrows = FloatRasterSearchcursor(gp, wts_g)
                #evrow = evrows.next()
                for evrow in evrows:
                    #gp.AddMessage("Value: %s"%evrow.value)
                    if evrow.Value not in catMCLists[0][evidx]:
                        catMCLists[0][evidx].append(evrow.Value)
                    #evrow = evrows.next()
            evidx += 1
        #gp.AddMessage('catMCLists: %s'%(catMCLists))
        lstWA = CalcVals4Msng(lstsVals, lstAreas[0], lstMD, catMCLists)
        #gp.AddMessage('lstWA: %s'%(str(lstWA)))
        ot = [['%s, %s'%(Input_Rasters[i], Wts_Tables[i])] for i in range(len(Input_Rasters))]
        #gp.AddMessage("ot=%s"%ot)
        strF2 = "case.dat"
        fnCase = os.path.join(arcpy.env.scratchFolder, strF2)
        fCase = open(fnCase, 'w')
        if not fCase :
            gp.AddError("Can't create 'case.dat'.")
            raise Exception
        nmbUC = len(lstsVals[0])
        getNmbET = True # True when first line of case.dat
        nmbET = 0 # Number of ET values in a line of case.dat
        #gp.AddMessage("Writing Logistic Regression input files...")
        ''' Reformat the labels for free evidence '''
        for j in range(len(lstsVals)):
            mcIdx = mcIndeces[j]
            if mcIdx > -1:
                listVals = catMCLists[0][j]
                #gp.AddMessage('listVals: %s'%(listVals))
                lstLV = listVals[:]
                lstLV = RemoveDuplicates(lstLV)
                elOT = ot[j]
                tknTF = elOT[0].split(',')
                strT = tknTF[0].strip()
                strF = tknTF[1].strip()
                first = True
               #gp.AddMessage("lstLV=%s"%lstLV)
                #gp.AddMessage("elOT=%s"%elOT)
                for lv in lstLV:
                    if lv == lstMD[j]: continue
                    if first:
                        elOT = ["%s (%s)"%(elOT[0], lv)]
                        first = False
                    else:
                        elOT.append("%s, %s (%s)"%(strT, strF, lv))
                    #gp.AddError("elOT=%s"%elOT)
                ot[j] = elOT
        #gp.AddMessage('ot=%s'%(str(ot)))
    ##' Loop through the unique conditions, substituting
    ##' the weighted average of known classes for missing data
    ##' and 'expanding' multi-class free data themes to
    ##' a series of binary themes
    ##'----------------------------------------------
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
                    #gp.AddMessage('catMCLists[%d]: %s'%(j, catMCLists[0][j]))
                    OFF = 0
                    ON = 1
                    if theVal == missing:
                        m=0
                        for v in listVals:
                            if v == missing:
                                continue
                            else:
                                #gp.AddMessage('lstWA[%d][%d]=%s'%(j, m, lstWA[j]))
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
        strF1 = "param.dat"
        fnParam = os.path.join(arcpy.env.scratchFolder, strF1) #param.dat file
        fParam = open(fnParam, 'w')
        if not fParam:
            gp.AddError("Error writing logistic regression parameter file.")
            raise Exception
        fParam.write('%s\\\n' %(arcpy.env.scratchFolder))
        fParam.write('%s\n' %strF2)
        fParam.write("%d %g\n" %(nmbET, unitCell))
        fParam.close()

    ### RunLR ------------------------------------------------------------------------------------------------------------
    #Check input files
        #Check input files exist
        #Paramfile = os.path.join(gp.scratchworkspace, 'param.dat')
        Paramfile = os.path.join(arcpy.env.scratchFolder, 'param.dat')
        if gp.exists(Paramfile):
            pass
            #gp.AddMessage("\nUsing the following input file in Logistic Regression: %s"%(Paramfile))
        else:
            gp.AddError("Logistic regression parameter file does not exist: %s"%Paramfile)
            raise Exception
        #Place input files folder in batch file
        #sdmlr.exe starts in input files folder.
        sdmlr = os.path.join(sys.path[0], 'sdmlr.exe')
        if not os.path.exists(sdmlr):
            gp.AddError("Logistic regression executable file does not exist: %s"%sdmlr)
            raise Exception
        os.chdir(arcpy.env.scratchFolder)
        if os.path.exists('logpol.tba'): os.remove('logpol.tba')
        if os.path.exists('logpol.out'): os.remove('logpol.out')
        if os.path.exists('cumfre.tba'): os.remove('cumfre.tba')
        if os.path.exists('logco.dat'): os.remove('logco.dat')
        #fnBat = os.path.join(gp.ScratchWorkspace, 'sdmlr.bat')
        fnBat = os.path.join( sys.path[0], 'sdmlr.bat')
        fBat = open(fnBat, 'w')
        #fBat.write("%s\n"%os.path.splitdrive(gp.ScratchWorkspace)[0])
        fBat.write("%s\n"%os.path.splitdrive(arcpy.env.scratchFolder)[0])
        fBat.write("CD %s\n"%os.path.splitdrive(arcpy.env.scratchFolder)[1])
        fBat.write('"%s"\n'%sdmlr)
        fBat.close()
        params = []
        try:
            #os.spawnv(os.P_WAIT, fnBat, params) # <==RDB  07/01/2010  replace with subprocess
            import subprocess
            p = subprocess.Popen([fnBat,params]).wait()
            gp.AddMessage('Running %s: '%fnBat)
        except OSError:
            gp.AddMessage('Exectuion failed %s: '%fnBat)

        if not os.path.exists('logpol.tba'):
            gp.AddError("Logistic regression output file %s\\logpol.tba does not exist.\n Error in case.dat or param.dat. "%arcpy.env.scratchFolder)
            raise Exception
        #gp.AddMessage("Finished running Logistic Regression")

    ###ReadLRResults -------------------------------------------------------------------------------------------------------

        thmuc = thmUC
        vTabUC = 'thmuc_lr'
        gp.MakeRasterLayer_management(thmuc, vTabUC)
        strFN = "logpol.tba"
        #strFnLR = os.path.join(gp.ScratchWorkspace, strFN)
        strFnLR = os.path.join(arcpy.env.scratchFolder, strFN)

        if not gp.Exists(strFnLR):
            gp.AddError("Reading Logistic Regression Results\nCould not find file: %s"%strFnLR)
            raise 'Existence error'
        #gp.AddMessage("Opening Logistic Regression Results: %s"%strFnLR)
        fLR = open(strFnLR, "r")
        if not fLR:
            gp.AddError("Input Error - Unable to open the file: %s for reading." %strFnLR)
            raise 'Open error'
        read = 0
        #fnNew = gp.GetParameterAsText(6)
        fnNew = parameters[6].valueAsText
        tblbn = os.path.basename(fnNew)
        [tbldir, tblfn] = os.path.split(fnNew)
        if tbldir.endswith(".gdb"):
            tblfn = tblfn[:-4] if tblfn.endswith(".dbf") else tblfn
            fnNew = fnNew[:-4] if fnNew.endswith(".dbf") else fnNew
            tblbn = tblbn[:-4] if tblbn.endswith(".dbf") else tblbn
        gp.AddMessage("fnNew: %s"%fnNew)
        gp.AddMessage('Making table to hold logistic regression results: %s'%fnNew)
        fnNew = tblbn
        print ("Table dir: ", tbldir);
        gp.CreateTable_management(tbldir, tblfn)
        print('Making table to hold logistic regression results: %s'%fnNew)
        fnNew = tbldir + "/" + fnNew;

        #To point to REAL table

        gp.AddField_management(fnNew, 'ID', 'LONG', 6)
        gp.AddField_management(fnNew, 'LRPostProb', 'Double', "#", "#", "#", "LR_Posterior_Probability")
        gp.AddField_management(fnNew, 'LR_Std_Dev', 'Double', "#", "#", "#", "LR_Standard_Deviation")
        gp.AddField_management(fnNew, 'LRTValue', 'Double', "#", "#", "#", "LR_TValue")
        gp.DeleteField_management(fnNew, "Field1")
        vTabLR = fnNew
        strLine = fLR.readline()
        vTabUCrows = workarounds_93.rowgen(gp.SearchCursor(vTabUC))
        #vTabUCrow = vTabUCrows.Next()
        ttl = 0
        #while vTabUCrow:
        for vTabUCrow in vTabUCrows: ttl += 1
            #vTabUCrow = vTabUCrows.Next()
        #gp.AddMessage("Reading Logistic Regression Results: %s"%strFnLR)
        vTabLRrows = gp.InsertCursor(vTabLR)
        while strLine:
            print (strLine);
            if strLine.strip() == 'DATA':
                read = 1
            elif read:
                vTabLRrow = vTabLRrows.NewRow()
                lstLine = strLine.split()
                if len(lstLine) > 5:
                    #gp.AddMessage('lstLine: %s'%lstLine)
                    vTabLRrow.SetValue("ID", int(lstLine[1].strip()))
                    vTabLRrow.SetValue("LRPostProb", float(lstLine[3].strip()))
                    vTabLRrow.SetValue("LR_Std_Dev", float(lstLine[5].strip()))
                    vTabLRrow.SetValue("LRTValue", float(lstLine[4].strip()))
                    vTabLRrows.InsertRow(vTabLRrow)
            strLine = fLR.readline()
        fLR.close()
        del vTabLRrow, vTabLRrows
        #gp.AddMessage('Created table to hold logistic regression results: %s'%fnNew)

    ##' Get the coefficients file
    ##'----------------------------------------------
        strFN2 = "logco.dat"
        fnLR2 = os.path.join(arcpy.env.scratchFolder, strFN2)
    ##  ' Open file for reading
    ##  '----------------------------------------------
        #gp.AddMessage("Opening Logistic Regression coefficients Results: %s"%fnLR2)
        fLR2 = open(fnLR2, "r")
        read = 0
    ##  ' Expand object tag list of theme, field, value combos
    ##  '----------------------------------------------
        #gp.AddMessage('Expanding object tag list of theme, field, value combos')
        lstLabels = []
        for el in ot:
            for e in el:
                lstLabels.append(e.replace(' ', ''))
        #gp.AddMessage('lstLabels: %s'%lstLabels)
    ##  ' Make vtab to hold theme coefficients
    ##  '----------------------------------------------
        #fnNew2 = gp.GetParameterAsText(7)
        fnNew2 = parameters[7].valueAsText
        tblbn = os.path.basename(fnNew2)
        [tbldir, tblfn] = os.path.split(fnNew2)
        if tbldir.endswith(".gdb"):
            tblfn = tblfn[:-4] if tblfn.endswith(".dbf") else tblfn
            fnNew2 = fnNew2[:-4] if fnNew2.endswith(".dbf") else fnNew2
            tblbn = tblbn[:-4] if tblbn.endswith(".dbf") else tblbn
        fnNew2 = tblbn
        print ("Tabledir: ", tbldir);
        #gp.AddMessage('Making table to hold theme coefficients: %s'%fnNew2)
        print('Making table to hold theme coefficients: %s'%fnNew2)
        #fnNew2 = tbldir + "/" + fnNew2;
        fnNew2 = os.path.join(tbldir, fnNew2)
        gp.AddMessage('Making table to hold theme coefficients: %s'%fnNew2)
        gp.CreateTable_management(tbldir, tblfn)
        gp.AddField_management(fnNew2, "Theme_ID", 'Long', 6, "#", "#", "Theme_ID")
        gp.AddField_management(fnNew2, "Theme", 'text', "#", "#", 80, "Evidential_Theme")
        gp.AddField_management(fnNew2, "Coeff", 'double', "#", "#", "#", 'Coefficient')
        gp.AddField_management(fnNew2, "LR_Std_Dev", 'double', "#", "#", "#", "LR_Standard_Deviation")
        gp.DeleteField(fnNew2, "Field1")
        vTabLR2 = fnNew2
        strLine = fLR2.readline()
        i = 0
        first = 1
        #gp.AddMessage("Reading Logistic Regression Coefficients Results: %s"%fnLR2)
        vTabLR2rows = gp.InsertCursor(vTabLR2)
        print ("Starting to read LR_Coeff")
        while strLine:
            print ("Rdr:" , strLine);
            if len(strLine.split()) > 1:
                if strLine.split()[0].strip() == 'pattern':
                    read = 1
                    strLine = fLR2.readline()
                    continue
            if read:

                lstLine = strLine.split()
                if len(lstLine) > 2:
                    vTabLR2row = vTabLR2rows.NewRow()
                    #vTabLR2row.SetValue('Theme_ID', long(lstLine[0].strip())+1)
                    print ("Theme: ", lstLine[0].strip());
                    vTabLR2row.SetValue('Theme_ID', int(lstLine[0].strip())+1)
                    if not first:
                        try:
                            #For all but first...
                            lbl = lstLabels.pop(0);
                            print ("Lbl:", lbl);
                            vTabLR2row.SetValue('Theme', lbl)
                        except IndexError:
                            gp.AddError('Evidence info %s not consistent with %s file'%(otfile, fnLR2))
                        i = i+1
                    else:
                        vTabLR2row.SetValue('Theme', "Constant Value")
                        first = 0
                    print ("Coeff:", lstLine[1].strip());
                    vTabLR2row.SetValue("Coeff", float(lstLine[1].strip()))
                    print ("LR_std_dev:", lstLine[2].strip());
                    vTabLR2row.SetValue("LR_Std_Dev", float(lstLine[2].strip()))
                    vTabLR2rows.InsertRow(vTabLR2row)
                else:
                    break
            strLine = fLR2.readline()
        fLR2.close()
        if len(lstLabels) != 0:
            gp.AddError('Evidence info %s not consistent with %s file'%(otfile, fnLR2))
        del vTabLR2row, vTabLR2rows
        #gp.AddMessage('Created table to hold theme coefficients: %s'%fnNew2)

        #Creating LR Response Rasters
        #Join LR polynomial table to unique conditions raster and copy
        #to get a raster with attributes
        cmb = thmUC
        cmbrl = 'cmbrl'
        gp.makerasterlayer_management(cmb, cmbrl)
        #tbl = gp.GetParameterAsText(6)
        tbl = parameters[6].valueAsText
        tbltv = 'tbltv'
        gp.maketableview_management(tbl, tbltv)
        gp.addjoin_management(cmbrl, 'Value', tbltv, 'ID')
        cmb_cpy = gp.createscratchname("cmb_cpy", '', 'raster', arcpy.env.scratchFolder)
        gp.copyraster_management(cmbrl, cmb_cpy)
        #Make output float rasters from attributes of joined unique conditions raster
        #outRaster1 = gp.GetParameterAsText(8)
        #outRaster2 = gp.GetParameterAsText(9)
        #outRaster3 =  gp.GetParameterAsText(10)
        outRaster1 =  parameters[8].valueAsText
        outRaster2 =  parameters[9].valueAsText
        outRaster3 =  parameters[10].valueAsText
        gp.addmessage("="*41+'\n'+"="*41)
        ##template = {'cmbrl':cmb_cpy}
        ##InExp = "CON(%(cmbrl)s.LRPOSTPROB >= 0, %(cmbrl)s.LRPOSTPROB, 0)"%template
        ##gp.SingleOutputMapAlgebra_sa(InExp, outRaster1)
        ##InExp = "CON(%(cmbrl)s.LR_STD_DEV >= 0, %(cmbrl)s.LR_STD_DEV, 0)"%template
        ##gp.SingleOutputMapAlgebra_sa(InExp, outRaster2)
        ##InExp = "CON(%(cmbrl)s.LRTVALUE >= 0, %(cmbrl)s.LRTVALUE, 0)"%template
        ##gp.SingleOutputMapAlgebra_sa(InExp, outRaster3) # <==RDB  07/01/2010
        # <==RDB  07/01/2010 -  SOMA expression is crashing in version 10. Changed to use Con tool.
        
        gp.Con_sa(cmb_cpy,cmb_cpy+".LRPOSTPROB",outRaster1,"0","LRPOSTPROB > 0")
        gp.Con_sa(cmb_cpy,cmb_cpy+".LR_STD_DEV",outRaster2,"0","LR_STD_DEV > 0")
        gp.Con_sa(cmb_cpy,cmb_cpy+".LRTVALUE",outRaster3,"0","LRTVALUE > 0")

        #Add t0 display
        #gp.SetParameterAsText(6, tbl)
        arcpy.SetParameterAsText(6,tbl)
        #gp.SetParameterAsText(7, gp.describe(vTabLR2).catalogpath)
        arcpy.SetParameterAsText(7, gp.describe(vTabLR2).catalogpath)
        #gp.SetParameterAsText(8, outRaster1)
        arcpy.SetParameterAsText(8, outRaster1)
        #gp.SetParameterAsText(9, outRaster2)
        arcpy.SetParameterAsText(9, outRaster2)
        #gp.SetParameterAsText(10, outRaster3)
        arcpy.SetParameterAsText(10, outRaster3)
    except arcpy.ExecuteError as e:
        #arcpy.AddError("\n");
        #arcpy.AddMessage("Caught ExecuteError in logistic regression. Details:");
        #args = e.args[0];
        #args.split('\n')
        #arcpy.AddError(args);
        raise 
    except:
        # get the traceback object
        #tb = sys.exc_info()[2]
        # tbinfo contains the line number that the code failed on and the code from that line
        #tbinfo = traceback.format_tb(tb)[0]
        #tbinfo = traceback.format_tb()
        # concatenate information together concerning the error into a message string
        #pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
        #str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
        # generate a message string for any geoprocessing tool errors
        #msgs = "GP ERRORS:\n" + gp.GetMessages(2) + "\n"
        #gp.AddError(msgs)

        # return gp messages for use with a script tool
        #gp.AddError(pymsg)

        # print messages for use in Python/PythonWin
        #print (pymsg)
        raise
    
def RemoveDuplicates(lst):
    """ Remove duplicates without sorting """
    unique = []
    for l in lst:
        if l not in unique: unique.append(l)
    #gp.Addmessage('RemoveDuplicates: %s'%unique)
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
           #gp.AddMessage('Free: %s %s %s %s'%(lstVals, lstValSum, lstAreaSum, sumArea))
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
            #gp.AddMessage('Free: nmbMD: %s\nlstVals: %s\nlstValSum: %s\nlstAreaSum: %s\nsumArea: %s'%(nmbMD, lstVals, lstValSum, lstAreaSum, sumArea))
            if nmbMD in lstVals:
                idxV = lstValSum.index(nmbMD)
                del lstValSum[idxV]
                areaMD = lstAreaSum[idxV]
                sumArea -= areaMD
            #gp.AddMessage('Free: nmbMD: %s\nlstVals: %s\nlstValSum: %s\nlstAreaSum: %s\nsumArea: %s'%(nmbMD, lstVals, lstValSum, lstAreaSum, sumArea))
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
            #gp.AddMessage('lstNmbWA: %s'%(lstNmbWA))
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
        msgs = "GP ERRORS:\n" + gp.GetMessages(2) + "\n"
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
        #gp.AddMessage("CalcVals4Msng: lstUCVals=%s"%(lstUCVals, ))
        lstThmMCF = lstMCF[0]
        lstMCFIdx = lstMCF[1]
        #gp.AddMessage("CalcVals4Msng.lstMCF: %s=%s, %s"%(lstMCF, lstThmMCF, lstMCFIdx))
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
        #gp.AddMessage('lstWAVals: %s'%(lstWAVals))
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
        msgs = "GP ERRORS:\n" + gp.GetMessages(2) + "\n"
        gp.AddError(msgs)

        # return gp messages for use with a script tool
        gp.AddError(pymsg)

        # print messages for use in Python/PythonWin
        print (pymsg)
        print (msgs)
        raise Exception



