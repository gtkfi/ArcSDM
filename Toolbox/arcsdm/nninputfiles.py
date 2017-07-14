"""
    ArcSDM 5 
    Converted by Tero Ronkko, GTK 2017
    
    Spatial Data Modeller for ESRI* ArcGIS 9.3
    Copyright 2009
    Gary L Raines, Reno, NV, USA: production and certification
    Don L Sawatzky, Spokane, WA, USA: Python software development
    
"""
import sys, os, traceback
import arcgisscripting
import workarounds_93

def rowgen( rows ):
    rows.reset()
    row = rows.next()
    while row:
        yield row
        row = rows.next()
    del rows

def getSelectedRows( sites, extr_sites ):
    """ extr_sites created by ExtractValuesToPoints tool in ArcGIS 9.3 are ALL selected """
    if gp.GetCount_management(sites) < gp.GetCount_management(extr_sites):
        #This can happen in ArcGIS 9.3 and ScriptVersion 9.2 
        sitesrows = rowgen(gp.searchcursor(sites))
        fids = '"FID" IN ('
        fidlist = []
        for siterow in sitesrows:
            fidlist.append(siterow.fid)
        fids += ",".join(map(str,fidlist)) + ")"
        #print len(fidlist), fids
        return rowgen(gp.searchcursor(extr_sites, fids))
    return rowgen(gp.searchcursor(extr_sites))

def MaxFZMforUC( TPs, TP_RasVals, RasValFld, FZMbrFld, TPFID ):
    ''' Get maximum fuzzy membership from among given UC/RASTERVALU in TP_RasVals
        Make dictionary with UC keys and number of time UC appears in RASTERVALU field.
        Make dictionary with UC keys and max fuzzy membership
        Make dictionary with UC keys and training sites FID for max fuzzy membership
        UC keys not present in TP_Dict should return 0.
    '''
    try:
        TP_Dict = {}
        TPFID_Dict = {}
        TPFZM_Dict = {}
        for sel_row in getSelectedRows(TPs, TP_RasVals):
            rasval = sel_row.GetValue(RasValFld)
            if rasval in TP_Dict:        
                TP_Dict[rasval] += 1
                if FZMbrFld:
                    if sel_row.getValue(FZMbrFld) > TPFZM_Dict[rasval]:
                        TPFID_Dict[rasval] = sel_row.getValue(TPFID)
                        TPFZM_Dict[rasval] = sel_row.getValue(FZMbrFld)
            else:
                TP_Dict[rasval] = 1
                if FZMbrFld:
                    TPFID_Dict[rasval] = sel_row.getValue(TPFID)
                    TPFZM_Dict[rasval] = sel_row.getValue(FZMbrFld)
        ###Check above results       
        ##for key in sorted(TP_Dict.keys()):
        ##    gp.addwarning ("%d, %d, %d, %f"%(key, TP_Dict[key], TPFID_Dict[key], \
        ##                                     TPFZM_Dict[key]))
        return TP_Dict, TPFID_Dict, TPFZM_Dict

    except Exception, Msg:
        # get the traceback object
        tb = sys.exc_info()[2]
        # tbinfo contains the line number that the code failed on and the code from that line
        tbinfo = traceback.format_tb(tb)[0]
        # concatenate information together concerning the error into a message string
        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
            str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
        # generate a message string for any geoprocessing tool errors
        msgs = "gp ERRORS:\n" + gp.GetMessages(2) + "\n"
        gp.AddError(msgs)

        # return gp messages for use with a script tool
        gp.AddError(pymsg)

        # print messages for use in Python/PythonWin
        print pymsg
        print msgs
        raise

def getMinMaxValues( uc, evidence_names ):
    minmaxdict = {}
    for evidence_name in evidence_names:
        minmaxdict[evidence_name] = {'minval':sys.maxint, 'maxval':-sys.maxint-1}
    for ucrow in rowgen(gp.searchcursor(uc)):
        for evidence_name in evidence_names:
            evidence_value = ucrow.getValue(evidence_name)
            if evidence_value < minmaxdict[evidence_name]['minval']:
                minmaxdict[evidence_name]['minval'] = evidence_value
            elif evidence_value > minmaxdict[evidence_name]['maxval']:
                minmaxdict[evidence_name]['maxval'] = evidence_value
    return minmaxdict

def getBandStatsFileMinMax( Output_statistics_file, evidence_names ):
    """
    # >>>> Format of such a file <<<<
    ###               STATISTICS of INDIVIDUAL LAYERS
    ###   Layer           MIN          MAX          MEAN         STD
    ### ---------------------------------------------------------------
    ##1            1.0000      21.0000       7.8410       4.1690
    ##2            1.0000     128.0000      25.5144      35.8494
    ##3          296.9573    4073.6306    1565.5359     763.9803
    ##4            0.3333     127.5000      51.5314      29.7958
    ##. . .
    # ===============================================================
    # <<<< end of file format <<<<
    Layer order same as left-to-right order in unique-conditions table
    """
        
# Script arguments...
    # Local variables...
    try:
    # Process: Band Collection Statistics...
    #
        BandFileName = Output_statistics_file #Band statistics ASCII file
        # Process: BandCollectionStats
        minmaxdict = {}
        for evidence_name in evidence_names:
            minmaxdict[evidence_name] = {'minval':sys.maxint, 'maxval':-sys.maxint-1}
        iter_ev_names = iter(evidence_names)
        fd = open(BandFileName,"r")
        for i in range(6): fd.next()# Blow off header
        for fileline in fd:
            if not fileline.startswith("#"):
                tokens = fileline.split()
                if len(tokens) > 2:
                    evidence_name = iter_ev_names.next()                    
                    minmaxdict[evidence_name]['minval'] = long(float(tokens[1]))
                    minmaxdict[evidence_name]['maxval'] = long(float(tokens[2]))
        fd.close()
        return minmaxdict
    
    except Exception, Msg:
        gp.AddError("Exception occurred in GetBandStatsFileMinMax.\n" + gp.GetMessages())
        # get the traceback object
        tb = sys.exc_info()[2]
        # tbinfo contains the line number that the code failed on and the code from that line
        tbinfo = traceback.format_tb(tb)[0]
        # concatenate information together concerning the error into a message string
        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
            str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
        # generate a message string for any geoprocessing tool errors
        msgs = "gp ERRORS:\n" + gp.GetMessages(2) + "\n"
        gp.AddError(msgs)

        # return gp messages for use with a script tool
        gp.AddError(pymsg)

        # print messages for use in Python/PythonWin
        print pymsg
        print msgs
        raise

def getValuesList( evidence_values, minmaxValues ):
    frac_values = []
    for evidence_name, evidence_value in evidence_values:
        minmaxes = minmaxValues[evidence_name]
        minvalue = minmaxes['minval']
        maxvalue = minmaxes['maxval']
        frac_values.append( (float(evidence_value) - minvalue) / (maxvalue - minvalue) )
    return frac_values
        
def composeDTAline( lineno, TPFid, unique_condition, value_list, fuzzy_mbrshp):
    value_list = ",".join(["%-7.4f"%value for value in value_list])
    s = ("%-6d,%-6d,%-6d,"+value_list+",%-7.4f\n")%(lineno, TPFid, unique_condition, fuzzy_mbrshp)
    return s

    
def execute(self, parameters, messages):
    try:       
        gp = arcgisscripting.create()

        #Arguments from tool dialog
        ucs = gp.getParameterAsText(0) #Unique Conditions raster
        #ucs_path = 'ucs_path'
        #gp.makerasterlayer_management(ucs, ucs_path)
        ucs_path = gp.describe(ucs).catalogpath
        TPs = gp.getParameterAsText(1) #Training sites
        FZMbrFld = gp.getParameterAsText(2) #Fuzzy membership field
        NDTPs = gp.getParameterAsText(3) #Nondeposit training sites
        NDFZMbrFld = gp.getParameterAsText(4) #Fuzzy membership field
        #Make Train file path
        traindta_filename = gp.getparameterastext(5) #Make train file or not
        traintable = True
        classtable = gp.getparameter(6) # Make class file or not
        classdta_filename = None
        #Make Train file path
        if not traindta_filename:
            UCName = os.path.splitext(os.path.basename(ucs))[0]
    ##        traindta_filename = UCName + "_train"
    ##        OutWrkSpc = gp.Workspace
    ##        traindta_filename = gp.createuniquename(traindta_filename + ".dta", OutWrkSpc)
    ##        if classtable:
    ##            classdta_filename = traindta_filename.replace('_train', '_class')
        else:
            UCName = traindta_filename
    ##        traindta_filename = UCName + "_train"
    ##        OutWrkSpc = gp.Workspace
    ##        traindta_filename = gp.createuniquename(traindta_filename + ".dta", OutWrkSpc)
    ##        if classtable:
    ##            classdta_filename = traindta_filename.replace('_train', '_class')
        traindta_filename = UCName + "_train"
        OutWrkSpc = gp.Workspace
        traindta_filename = gp.createuniquename(traindta_filename + ".dta", OutWrkSpc)
        if classtable:
            classdta_filename = traindta_filename.replace('_train', '_class')
        #Make Class file path
        if classtable and not classdta_filename:        
            classdta_filename = gp.createuniquename(UCName + "_class" + ".dta", gp.workspace)
        #Get min/max values of evidence fields in unique conditions raster
        BandStatsFile = gp.getparameterastext(7) #Prepared band statistics file or not
        evidence_names = [row.name for row in rowgen(gp.listfields(ucs))][3:]
        if BandStatsFile:
            minmaxValues = getBandStatsFileMinMax(BandStatsFile, evidence_names)
        else:
            minmaxValues = getMinMaxValues(ucs_path, evidence_names)
        UnitArea = 1.0 #gp.getparameter(8) #1.0
        #gp.AddMessage("Got arguments..."+time.ctime())
        
        gp.AddMessage("Training file = " + str(traindta_filename))
        if classtable:
            gp.AddMessage("Class file = " + str(classdta_filename))

        #Derive other values
        RasValFld = NDRasValFld = 'RASTERVALU'
        #Feature classes to be gotten with Extract tool
        TP_RasVals = WorkArounds_93.ExtractValuesToPoints(gp, ucs, TPs, 'TPFID')
        NDTP_RasVals = WorkArounds_93.ExtractValuesToPoints(gp, ucs, NDTPs, 'NDTPFID')    
        TP_Dict, TPFID_Dict, TPFZM_Dict = MaxFZMforUC( TPs, TP_RasVals, RasValFld, FZMbrFld, 'TPFID' )
        NDTP_Dict, NDTPFID_Dict, NDTPFZM_Dict = MaxFZMforUC( NDTPs, NDTP_RasVals, NDRasValFld, NDFZMbrFld, 'NDTPFID' )
        CellSize = float(gp.cellsize)
        train_lineno = 0
        class_lineno = 0
        train_lines = []
        class_lines = []
        #Compose the lines of the files
        for ucrow in rowgen(gp.searchcursor(ucs_path)):
            #Read the UC raster rows, get evidence values
            UCValue = ucrow.value
            #gp.addwarning('%d'%UCValue)
            evidence_values = getValuesList( \
                [(evidence_name, ucrow.getValue(evidence_name)) for evidence_name in evidence_names], \
                minmaxValues
                )
            if classtable:
                #Compose the class table line
                class_lineno += 1
                wLine = str(class_lineno) + ","
                wLine = wLine + str(TP_Dict.get(UCValue, 0)) + ","
                Area = "%.1f" % (ucrow.Count * CellSize * CellSize / 1000000.0 / UnitArea)
                wLine = wLine + str(Area) + ","
                for theVal in evidence_values:
                    wLine = wLine + "%.5f" % theVal + ","
                wLine += "0\n"
                class_lines.append(wLine)
            if traintable:
                #Compose the train table line
                if TP_Dict.get(UCValue, 0):
                    if FZMbrFld:
                        if NDFZMbrFld:
                            if NDTP_Dict.get(UCValue, 0):
                                #Use max fuzzy mbrshp between dep and non-dep site
                                TP_Mbrship = TPFZM_Dict[UCValue]
                                NDTP_Mbrship = NDTPFZM_Dict[UCValue]
                                if TP_Mbrship > NDTP_Mbrship:
                                    TPFid = TPFID_Dict[UCValue]
                                    fuzzy_mbrshp = TP_Mbrship
                                else:
                                    TPFid = NDTPFID_Dict[UCValue] + 1000
                                    fuzzy_mbrshp = NDTP_Mbrship
                            else:
                                #Use fuzzy mbrshp of dep site
                                TPFid = TPFID_Dict[UCValue]
                                fuzzy_mbrshp = TPFZM_Dict[UCValue]
                        else:
                            #Use fuzzy mbrshp of dep site
                            TPFid = TPFID_Dict[UCValue]
                            fuzzy_mbrshp = TPFZM_Dict[UCValue]
                    elif NDFZMbrFld:
                        if NDTP_Dict.get(UCValue, 0):
                            #Use fuzzy mbrshp of non=dep site
                            TPFid = NDTPFID_Dict[UCValue] + 1000
                            fuzzy_mbrshp = NDTPFZM_Dict[UCValue]
                        else:
                            #Use default fuzzy mbrshp of dep site
                            TPFid = 0
                            fuzzy_mbrshp = 1.0
                    else:
                        #Use default fuzzy mbrshp of dep site
                        TPFid = 0
                        fuzzy_mbrshp = 1.0
                elif NDTP_Dict.get(UCValue, 0):
                    if NDFZMbrFld:
                        #Use fuzzy mbrshp of non-dep site
                        TPFid = NDTPFID_Dict[UCValue] + 1000
                        fuzzy_mbrshp = NDTPFZM_Dict[UCValue]
                    else:
                        #Use default fuzzy mbrshp of non=dep site
                        TPFid = 1000
                        fuzzy_mbrshp = 0.0                    
                else:
                    #No sites within UC area
                    continue #Do not write line
                train_lineno += 1
                try:
                    train_lines.append(composeDTAline(train_lineno, TPFid, UCValue, evidence_values, fuzzy_mbrshp))
                except:
                    gp.addwarning(str(train_lineno))
                
        if traintable:
            #Write out the train file
            trainfd_dta = open(traindta_filename, 'w')
            trainfd_dta.write('%-d\n'%len(evidence_names))
            trainfd_dta.write(str(gp.getcount_management(TPs))+'\n')
            trainfd_dta.write('1\n')
            trainfd_dta.write(str(train_lineno)+'\n')
            trainfd_dta.writelines(train_lines)
            trainfd_dta.close()
        if classtable:
            #Write out the class file
            classfd_dta = open(classdta_filename, 'w')
            classfd_dta.write('%-d\n'%len(evidence_names))
            if traintable:
                classfd_dta.write(str(gp.getcount_management(TPs))+'\n')
            else: classfd_dta.write('5\n')
            classfd_dta.write('1\n')
            classfd_dta.write('%-d\n'%class_lineno)
            classfd_dta.writelines(class_lines)
            classfd_dta.close()
        
    except:
        # Get the geoprocessing error messages
        #
        msgs = gp.GetMessage(0)
        msgs += gp.GetMessages(2)

        # Return gp error messages for use with a script tool
        #
        gp.AddError(msgs)

        # Print gp error messages for use in Python/PythonWin
        #
        print msgs
        # Get the traceback object
        #
        tb = sys.exc_info()[2]
        tbinfo = traceback.format_tb(tb)[0]

        # Concatenate information together concerning the error into a 
        #   message string
        #
        pymsg = tbinfo + "\n" + str(sys.exc_type)+ ": " + str(sys.exc_value)

        # Return python error messages for use with a script tool
        #
        gp.AddError(pymsg)

        # Print Python error messages for use in Python/PythonWin
        #
        print pymsg
