"""

    ArcSDM 5 for Arcgis pro and desktop
    Converted by Tero Ronkko, GTK,  2017

    Spatial Data Modeller for ESRI* ArcGIS 9.2
    Copyright 2007
    Gary L Raines, Reno, NV, USA: production and certification
    Don L Sawatzky, Spokane, WA, USA: Python software development
    
' Name:  gSDI.ReadNNResults
'
' Topics:  Spatial Data Modeller
'
' Description:  Reads the text file produced
'       by the logistic regression code and
'       writes a new dBase file in the
''       user specified location.  Adds this
'       dBase file to the project and joins
'       it to the unique conditions table.
'
' Returns:
'==============================================
"""
# Import system modules
import arcpy
import sys, string, os
import math, types, traceback

# Create the Geoprocessor object
#gp = win32com.client.Dispatch("esriGeoprocessing.GpDispatch.1")
import arcgisscripting
gp = arcgisscripting.create()

gp.OverwriteOutput = 1

WrkSpace = gp.Workspace

#gp.SetProduct("arcview")
# Check out any necessary licenses
gp.CheckOutExtension("spatial")

#set a toolbox
#gp.toolbox = "management"
#
def execute(self, parameters, messages):




    #gp.AddMessage('A')
    try:
        fldP = 1
        X = 1
        fldFC = 1
        fldFM = 1
        dTitle = "Reading Neural Network Results"
        thmUC = parameters[0].valueAsText # gp.getParameterAsText(0) #Unique conditions raster
        fnRBNDX = parameters[1].valueAsText #gp.getParameterAsText(1) #DataExplor .rbn file name
        rbndx = fnRBNDX != "" and fnRBNDX != None 
        #gp.AddMessage("rbndx="+str(rbndx))
        fnRBNGX = parameters[2].valueAsText #gp.getParameterAsText(2) #GeoExplor .pnn output
        rbngx = fnRBNGX != "" and fnRBNGX != None 
        #gp.AddMessage("rbngx="+str(rbngx))
        rbn = rbndx or rbngx
        #gp.AddMessage("rbn="+str(rbn))
        fnFC = parameters[3].valueAsText #gp.getParameterAsText(3) #Fuzzy Classification .fuz file
        fc = fnFC != "" and fnFC != None 
        #gp.AddMessage("fc="+str(fc))
        if (not rbndx) and (not rbngx) and (not fc):
            gp.AddError("No Neural Network results files to process.");
            raise arcpy.ExecuteError("InputError");
            #os.sys.exit(0)
        gp.AddMessage("Got arguments.")
        #'' Open specified file(s)
        if rbndx:
            try:
                fRBNDX = open(fnRBNDX,"r")
                gp.AddWarning("Opened " + fnRBNDX)
            except IOError:
                gp.AddError("Cannot find/open input file: " + fnRBNDX)
                raise arcpy.ExecuteError("InputError")
        else:
            fRBNDX = None
        if rbngx:
            try:
                fRBNGX = open(fnRBNGX,"r")
                gp.AddWarning("Opened " + fnRBNGX)
            except IOError:
                gp.AddError("Cannot find/open input file: " + fnRBNGX)
                raise arcpy.ExecuteError("InputError")
        else:
            fRBNGX = None
        if fc:
            try:
                fFC = open(fnFC,"r")
                gp.AddWarning("Opened " + fnFC)
            except IOError:
                gp.AddError("Cannot find/open input file: " + fnFC)
                raise arcpy.ExecuteError ("InputError")
        else:
            fFC = None
        #gp.AddMessage("Opened input Neural Network files.")
        output_name = parameters[4].valueAsText
        #gp.AddMessage("Table filename: " + output_name)
        #gp.AddMessage("Workspace: " + WrkSpace)
        try:
            if not gp.Exists(output_name):
                (wkspath,vTabNN) = os.path.split(output_name)
                gp.CreateTable_management(wkspath,vTabNN)
                vTabNN = os.path.join(wkspath,vTabNN)
                #gp.AddMessage("Created table: " + vTabNN)
            else:
                #gp.SelectLayerByAttribute_management(vTabNN,"CLEAR_SELECTION")
                #gp.SelectLayerByAttribute_management(vTabNN,"SWITCH_SELECTION")
                vTabNN = output_name
                gp.DeleteRows_management(vTabNN)
                #gp.AddMessage("Got existing table: " + vTabNN)
        except:
            gp.AddError(gp.GetMessages())
            raise arcpy.ExecuteError("InputError")
        fldID = ""
        #fldPtrn = ""
        fldMshp1 = ""
        fldMshp2 = ""
        fldFClst = ""
        #gp.AddMessage('got to here 1: %s' %vTabNN)
        lstFlds = gp.ListFields(vTabNN,"ID","LONG")
        if not lstFlds.Next():
            gp.AddField_management(vTabNN,"ID","LONG")
            #gp.AddMessage("Added 'ID' field")
        fldID = "ID"
        #gp.AddMessage("fldID: " + fldID)

    ##    if rbn:
    ##        gp.AddMessage("rbn: " + str(rbn) + "\nfldP: " + str(fldP) + "\nrbndx: " + str(rbndx) +
    ##                      "\nrbngx: " + str(rbngx) + "\nfc: " + str(fc) + "\nfldFC: " + str(fldFC))
        if rbn:
            if fldP:
                lstFlds = gp.ListFields(vTabNN,"RBFLNPatrn")
                fld = lstFlds.Next()
                if not fld:
                    ##gp.AddField_management(vTabNN,"RBFLNPatrn","LONG","#","#","4","RBFLN Pattern")
                    ##gp.AddMessage("Added 'RBFLNPatrn' field")
                    pass
                else:
                    #gp.AddMessage(fld.type)
    ##                if fld.type <> 'Integer':
    ##                    gp.AddError("Found 'RBFLNPatrn' field not type 'Long'. Restart")
                        gp.DeleteField_management(vTabNN,fld.name)
    ##                    raise "Field error"
    ##                else:
    ##                    #gp.AddMessage("Found 'RBFLNPatrn' field")
    ##                    pass
                ##fldPtrn = "RBFLNPatrn"
            if X:
                if rbndx:
                    lstFlds = gp.ListFields(vTabNN,"RBFLN")
                    fld = lstFlds.Next()
                    if not fld:
                        gp.AddField_management(vTabNN,"RBFLN","DOUBLE","10","6","#","RBFLN DataXplore Pattern Membership")
                        #gp.AddMessage("Added 'RBFLN' field")
                    else:
                        #gp.AddMessage(fld.type)
                        if type(fld) != 'Double':
                            #gp.AddMessage("Found 'RBFLN' field not of type 'Double'. Restart")
                            gp.DeleteField_management(vTabNN,fld.name)
                            raise "Field error"
                        else:
                            #gp.AddMessage("Found 'RBFLN' field")
                            pass
                    fldMshp1 = "RBFLN"
            if rbngx:
                lstFlds = gp.ListFields(vTabNN,"PNN")
                fld = lstFlds.Next()
                if not fld:
                #if not lstFlds.Next():
                    
                    gp.AddField_management(vTabNN,"PNN","DOUBLE","10","6","#","RBFLN GeoXplore Pattern Membership")
                    #gp.AddMessage("Added 'PNN' field")
                else:
                    #gp.AddMessage(fld.type)
                    if fld.type != 'Double':
                        gp.AddError("Found 'PNN' field not of type 'Double'. Restart")
                        gp.DeleteField_management(vTabNN,fld.name)
                        raise "Field error"
                    else:
                        #gp.AddMessage("Found 'PNN' field")
                        pass
                fldMshp2 = "PNN"
        if fc:
            if fldFC:
                lstFlds = gp.ListFields(vTabNN,"FzzyClstr","LONG")
                fld = lstFlds.Next()
                if not fld:
                    gp.AddField_management(vTabNN,"FzzyClstr","LONG","#","#","#","Fuzzy Cluster")
                    #gp.AddMessage("Added 'FzzyClstr' field")
                else:
                    #gp.AddMessage(fld.type)
                    if fld.type != 'Integer':
                        #gp.AddMessage("Found 'FzzyClstr' field not type 'Long'. Restart")
                        gp.DeleteField_management(vTabNN,fld.name)
                        raise "Field error"
                    else:
                        #gp.AddMessage("Found 'FzzyClstr' field")
                        pass
                fldFClst = "FzzyClstr"
        #Get rid of artifact of CreateTable tool
        lstFlds = gp.ListFields(vTabNN,"FIELD1")
        if lstFlds.Next():
            gp.DeleteField_management(vTabNN,"FIELD1")
        #gp.AddMessage("Checked output fields...")
    ##'' Check input files are same length
    ##''------------------------------------------------
        #gp.AddMessage("Reading files..")
        if fc and rbndx:
            if len(fRBNDX.readlines()) != len(fFC.readlines()):
                gp.AddError("RBN file and FUZ file not same length.")
                raise "InputError"
            else:
                pass
                #gp.AddMessage("RBN file and FUZ file same length.")
            fRBNDX.seek(0)
            fFC.seek(0)
        if fc and rbngx:
            if len(fRBNGX.readlines()) != len(fFC.readlines()):
                gp.AddError("PNN file and FUZ file not same length.")
                raise "InputError"
            else:
                pass
                #gp.AddMessage("PNN file and FUZ file same length.")
            fRBNGX.seek(0)
            fFC.seek(0)

        if rbndx:
            strLn01 = fRBNDX.readline()
            if not strLn01:
                gp.AddError("No data in file: " + fRBNDX.name)
                raise "InputError"
            #gp.AddMessage("Read first line of fRBNDX file.")
        else:
            strLn01 = ""

        if rbngx:
            strLn02 = fRBNGX.readline()
            if not strLn02:
                gp.AddError("No data in file: " + fRBNGX.name)
                raise "InputError"
            #gp.AddMessage("Read first line of fRBNGX file.")
        else:
            strLn02 = ""

        if fc:
            strLn1 = fFC.readline()
            if not strLn1:
                gp.AddError("No data in file: " + fFC.name)
                raise "InputError"
            #gp.AddMessage("Read first line of fFC file.")
        else:
            strLn1 = ""
        ndflds = 1

        if fldFM and ndflds:
            lstLn1 = strLn1.split(",")
            nmbFld = len(lstLn1) - 2
            #gp.AddMessage(strLn1 + "\nnmbFld=" + str(nmbFld))
            lstFldNames = []
            lstFlds = gp.ListFields(vTabNN)
            lstFlds.Reset()
            fld = lstFlds.Next()
            while fld:
                #gp.AddMessage(fld.Name)
                lstFldNames.append(fld.Name)
                fld = lstFlds.Next()
            for i in range(1, nmbFld+1):
                fldPat = "Pat" + str(i)
                #gp.AddMessage(fldPtrn)
                if fldPat not in lstFldNames:
                    #gp.AddMessage("Adding field " + fldPat)
                    try:
                        gp.AddField_management(vTabNN,fldPat,"DOUBLE","10","6","#","Pattern " + str(i))
                    except:
                        raise "Add field error. " + gp.GetMessages()
                else:
                    #gp.AddMessage("Field exists."+fldPat)
                    pass
        #gp.AddMessage("Reading neural network files...")
        ID = 1
        try:
            #gp.AddMessage("Getting insert cursor...")
            VTabNNCurs = gp.InsertCursor(vTabNN)
        except:
            raise "InsertCursor error"
        #gp.AddMessage("Got insert cursor.")
        
        while fRBNDX or fRBNGX or fFC:
            #gp.AddMessage("strLn1: " + strLn1)
            if len(strLn1) != 0:
                if len(strLn1.split(",")) < 3:
                    strLn1 = ""
            try:
                #gp.AddMessage("Creating new row...")
                r = VTabNNCurs.NewRow()
            except:
                raise "Error creating new row: " + gp.AddMessages()
            #gp.AddMessage("Created new row.")
            #gp.AddMessage("fldID=%s,ID=%s"%(fldID,ID))
            r.SetValue(fldID,ID)
            #gp.AddMessage("Set ID = " + str(ID))

            if rbndx and len(strLn01) > 0:
                #gp.AddMessage("Doing rnbdx...")
                strLn01 = string.strip(strLn01)
                lstLn01 = strLn01.split(",")
                #gp.AddMessage(str(lstLn01))
                if fldP:
                    if len(lstLn01) > 1:
                        ptrn = lstLn01[1]
                        #gp.AddMessage("ptrn=" + ptrn + "; " + str(long(ptrn)))
                        #gp.AddMessage("Setting fldPtrn:%s = %s" %(fldPtrn, ptrn))
                        #r.SetValue(fldPtrn,long(ptrn))
                        #gp.AddMessage("Set fldPtrn:%s = %s" %(fldPtrn, ptrn))
                if len(fldMshp1) > 0:
                    #gp.AddMessage("Setting " + fldMshp1)
                    if len(lstLn01) > 2:
                        #gp.AddMessage("len(lstLn01)=" + str(len(lstLn01)))
                        mshp = lstLn01[2]
                        #gp.AddMessage("lstLn01[2]=" + lstLn01[2])
                        #gp.AddMessage("Setting fldMshp1:%s = %s" %(fldMshp1,mshp))
                        r.SetValue(fldMshp1,float(mshp))
                        
    ##520:     If rbngx And (Len(strLn02) <> 0) Then
            if rbngx and len(strLn02) > 0:
                #gp.AddMessage("Doing rbngx...")
    ##            Dim lstLn02 As Variant
    ##522:        lstLn02 = Split(strLn02, ",")
                strLn02 = string.strip(strLn02)
                lstLn02 = strLn02.split(",")
    ##523:         If fldP Then
                if fldP:
    ##'      If (lstLn0.Count < 2) Then
    ##525:             If UBound(lstLn02) < 1 Then
                    if len(lstLn02) >= 1:
                        ptrn = lstLn02[1]
                        #gp.AddMessage("Setting fldPtrn:%s = %s" %(fldPtrn, ptrn))
                        #r.SetValue(fldPtrn, long(ptrn))
                        #gp.AddMessage("Set fldPtrn:%s = %s" %(fldPtrn,ptrn))
                if fldMshp2 != "":
    ##551:         If X Then
    ##'      If (lstLn0.Count < 3) Then
    ##553:             If UBound(lstLn02) < 2 Then
                    if len(lstLn02) > 2:
                        mshp2 = float(lstLn02[2])
                        #gp.AddWarning("Setting fldMshp2:%s = %f" %(fldMshp2, mshp2))
                        r.SetValue(fldMshp2,mshp2)
                        #gp.AddWarning("Set fldMshp2:%s = %f" %(fldMshp2, mshp2))
                        
    ##'  If ((fc) And (strLn1 <> Nil)) Then
    ##582:     If fc And (strLn1 <> "") Then
            if fc and len(strLn1) != 0:
                #gp.AddMessage("Doing fc...")
    ##'    lstLn1 = strLn1.AsTokens(",")
    ##             'Dim lstLn1 As Variant
    ##585:         lstLn1 = Split(strLn1, ",")
                strLn1 = string.strip(strLn1)
                lstLn1 = strLn1.split(",")
    ##'    If (lstLn1.Count > 2) Then
    ##587:         If UBound(lstLn1) > 1 Then
                if len(lstLn1) > 1:
    ##'      If (fldFC) Then
    ##589:             If fldFC Then
                    if fldFC:
                       fcVal = lstLn1[1]
                        #gp.AddWarning("Setting fldFClst:%s = %s" %(fldFClst, fcVal))
                       r.SetValue(fldFClst,long(fcVal))
                        #gp.AddWarning("Set fldFClst:%s = %s" %(fldFClst, fcVal))
    ##'          End
    ##614:                     End If
    ##'        End
    ##616:                 End If
    ##'      End
    ##618:             End If
    ##'      If (fldFM) Then
    ##620:             If fldFM Then
                    if fldFM:
    ##'        If (lstLn1.Count < 3) Then
    ##622:                 If UBound(lstLn1) < 2 Then
                        if len(lstLn1) < 3:
    ##'  '          av.Run("gSDI.ErrorReadingFile",{fFC,{fRBN},dTitle})
    ##624:                      gSDI_ErrorReadingFile Array(fFC, fnFC, Array(fRBNDX), dTitle)
                            gp.AddWarning("My bad.")
                        else:
                            #gp.AddMessage("lstLn1:\n" + str(lstLn1)+ "\n" +str(len(lstLn1)))
                            for i in range(2,len(lstLn1)):
                                #gp.AddMessage(str(i))
    ##663:                         fcMshp = Trim(CStr(lstLn1(i)))
                                fcMshp = lstLn1[i]
                                #gp.AddMessage(fcMshp)
                                fldPtn = "Pat"+str(i-1)
                                #gp.AddMessage("Setting fldPtn:" + fldPtn + "=" + fcMshp)
                                r.SetValue(fldPtn,float(fcMshp))
                                #gp.AddMessage("Set fldPtn:" + fldPtn + "=" + fcMshp)
            VTabNNCurs.InsertRow(r)
            #gp.AddMessage("Stored row.")
                                    
            #gp.AddMessage("Reading next line of input files.")
            if fRBNDX:
    ##'    strLn0 = fRBN.ReadElt
    ##700:         strLn01 = fRBNDX.ReadLine
                strLn01 = fRBNDX.readline()
    ##'    If (strLn0 <> Nil) Then
    ##702:         If strLn01 <> "" Then
                if strLn01:
    ##'      If (strLn0.AsTokens(",").Count < 2) Then
    ##704:             If UBound(Split(strLn01, ",")) < 1 Then
                    if len(strLn01.split(",")) < 2:
    ##'        strLn0 = Nil
    ##706:                 strLn01 = ""
                        strLn01 = ""
    ##'      End
    ##708:             End If
    ##'    End
                else:
                    break
    ##716:     If boolfRBNGX Then
            if fRBNGX:
    ##'    strLn0 = fRBN.ReadElt
    ##718:         strLn02 = fRBNGX.ReadLine
                strLn02 = fRBNGX.readline()
    ##'    If (strLn0 <> Nil) Then
    ##720:         If strLn02 <> "" Then
                if strLn02:
    ##'      If (strLn0.AsTokens(",").Count < 2) Then
    ##722:             If UBound(Split(strLn02, ",")) < 1 Then
                    if len(strLn02.split(",")) < 2: 
    ##'        strLn0 = Nil
    ##724:                 strLn02 = ""
                        strLn02 = ""
    ##'      End
    ##726:             End If
                else:
                    break
            if fFC:
    ##'  '    strLn1 = fFC.ReadElt
    ##736:         strLn1 = fFC.ReadLine
                strLn1 = fFC.readline()
    ##'    If (strLn1 <> Nil) Then
    ##738:         If strLn1 <> "" Then
                if strLn1:
    ##'      If (strLn1.AsTokens(",").Count < 2) Then
    ##            'Dim tokens As Variant
    ##741:             tokens = Split(strLn1, ",")
                    tokens = strLn1.split(",")
    ##742:             If UBound(tokens) < 1 Then
                    if len(tokens) <= 1:
    ##'        strLn1 = Nil
    ##744:                 strLn1 = ""
                        strLn1 = ""
    ##'      End
    ##746:             End If
                else:
                    break
            ID = ID + 1
        #gp.AddMessage("Closing input files.")
    ##760:     If Not fRBNDX Is Nothing Then fRBNDX.Close
        if fRBNDX:
            fRBNDX.close()
            #gp.AddMessage("Closed file: %s" % fRBNDX.name)
    ##761:     If Not fRBNGX Is Nothing Then fRBNGX.Close
        if fRBNGX:
            fRBNGX.close()
            #gp.AddMessage("Closed file: %s" % fRBNGX.name)
    ##'End
    ##'
    ##'If (fFC <> Nil) Then
    ##'
    ##'  fFC.Close
    ##767:     If Not fFC Is Nothing Then fFC.Close
        if fFC:
            fFC.close()
            #gp.AddMessage("Closed file: %s" % fFC.name)
    ##'End
    ##'

        ##<== RDB added extra parameter to specify output raster for NNoutput Files tool
        OutputRaster5 = parameters[5].valueAsText  # RDB

        gp.CopyRaster(thmUC, OutputRaster5)
        gp.JoinField_management(OutputRaster5, 'Value', vTabNN, 'ID')

        ##gp.AddMessage("Output Raster: " + OutputRaster5)
        #gp.MakeRasterLayer_management(thmUC,'thmUCLayer')  #<== RDB
        
        #gp.AddJoin_management('thmUCLayer','Value',vTabNN,'ID')
        #gp.CopyRaster('thmUCLayer', OutputRaster5)
        ##gp.AddJoin_management(thmUC,'Value',vTabNN,'ID')

        #gp.AddMessage("Deconstructing table cursor.")
        gp.SetParameterAsText(4,gp.Describe(vTabNN).CatalogPath)
        gp.SetParameterAsText(5,gp.Describe(OutputRaster5).CatalogPath) #<==RDB
        if VTabNNCurs:
            del VTabNNCurs
        #gp.AddMessage("Output table: %s" %gp.Describe(vTabNN).CatalogPath)
        #gp.AddMessage("Done.")

    except "InputError":
        pass
    except Exception:
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
