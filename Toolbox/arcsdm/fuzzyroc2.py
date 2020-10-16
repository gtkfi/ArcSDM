# -*- coding: utf-8 -*-
"""
    FuzzyROC - ArcSDM 5 for ArcGis pro 
    Arto Laiho & Johanna Torppa, Geological Survey of Finland, 11.5-23.6.2020, Version 2
    Added Midpoint and Spread value missing tests 11.8.2020/AL
    Fuzzy Overlay file selection has completely rewritten 15.10.2020/AL
"""
# Import system modules
import sys,os,traceback
import arcpy
import arcsdm
import math
from arcpy.sa import *
from arcpy import env

def Execute(self, parameters, messages):
    try:
        # Save Workspace
        enviWorkspace = env.workspace
        csvfile="?"

        #Get and print version information
        with open (os.path.join(os.path.dirname(__file__), "arcsdm_version.txt"), "r") as myfile:
            data=myfile.readlines()
        arcpy.AddMessage("%-20s %s" % ("", data[0]) ); 
        installinfo = arcpy.GetInstallInfo ();
        arcpy.AddMessage("%-20s %s (%s)" % ("Arcgis environment: ", installinfo['ProductName'] , installinfo['Version'] ));

        # Load parameters...
        # Param 0: Input rasters, Fuzzy Membership functions and parameters, DETable, multiValue=1, Required, Input
        # columns = Input raster, Membership type, Midpoint Min, Midpoint Max, Midpoint Count, Spread Min, Spread Max, Spread Count
        # Membership types: Gaussian, Small, Large, Near, MSLarge, MSSmall, Linear
        # parameters[0] = rc_till_co Gaussian 1 4 4 2 5 4;rc_till_ca Large 2 5 4 3 4 2;rc_till_au Small 3 4 2 4 5 2;...
        memberParams = parameters[0].valueAsText.split(';')
        if (len(memberParams) < 2):
            arcpy.AddError ("ERROR: Minimum number of Input Rasters is 2.")
            raise

        # Param 1: Fuzzy Overlay Parameters, DETable, Required, Input
        # columns = Overlay type, Parameter
        # Overlay types: And, Or, Product, Sum, Gamma
        # parameters[1] = And # (or: Gamma 5)
        overlayParams = parameters[1].valueAsText.split(' ')

        # Param 2: ROC True Positives Feature Class, DEFeatureClass, Required, Input
        # parameters[2] = feature class name
        truepositives = parameters[2].valueAsText
        trueDescr = arcpy.Describe(truepositives)
        true_coord_system=trueDescr.spatialReference.name
        arcpy.AddMessage("ROC True Positives: " + os.path.basename(truepositives) + " " + trueDescr.dataType + " " + true_coord_system)

        # Param 3: ROC Destination Folder, DEFolder, Required, Input, "File System"
        output_folder = parameters[3].valueAsText

        # Remove old files from ROC output folder if any
        arcpy.AddMessage("="*30)
        arcpy.AddMessage("Clean up workspace...")
        arcpy.env.workspace = output_folder
        count=len(os.listdir(output_folder))
        #arcpy.AddMessage("len(os.listdir(output_folder) = " + str(count))
        if (count > 0):
            count=0
            for oldfile in os.listdir(output_folder):
                #arcpy.AddMessage(oldfile)
                if ("results" in oldfile):
                    os.remove(output_folder + "\\" + oldfile)
                    count=count+1
            if (count > 0):
                arcpy.AddMessage(str(count) + " results* files removed from " + output_folder)

        # Remove old Raster Datasets from workspace
        arcpy.env.workspace = enviWorkspace
        wsdesc = arcpy.Describe(enviWorkspace)
        arcpy.AddMessage("Workspace is " + str(enviWorkspace) + " and its type is " + wsdesc.workspaceType)
        import shutil
        if (wsdesc.workspaceType == "FileSystem"):
            arcpy.AddMessage("Remove all FM_ files and all FO_ files from workspace File System")

            # Remove all FM_ files and all FO_ files from workspace File System
            count=len(os.listdir(output_folder))
            if (count > 0):
                fmcount=0
                focount=0
                for oldfile in os.listdir(output_folder):
                    if (oldfile[0:3] == "FM_"):
                        os.remove(enviWorkspace + "\\" + oldfile)
                        fmcount=fmcount+1
                    elif (oldfile[0:3] == "FO_"): 
                        os.remove(enviWorkspace + "\\" + oldfile)
                        focount=focount+1
                if (fmcount > 0):
                    arcpy.AddMessage(str(fmcount) + " FM_n_m files removed")
                if (focount > 0):
                    arcpy.AddMessage(str(focount) + " FO_n files removed")

            arcpy.AddMessage("remove all fm_ and fo_ subdirectories and subfiles")
            # remove all fm_ and fo_ subdirectories and subfiles
            count=0
            for olddir in os.listdir(enviWorkspace):
                if os.path.isdir(os.path.join(enviWorkspace,olddir)):
                    if (olddir[0:3] == "fm_" or olddir[0:3] == "fo_"):
                        shutil.rmtree(enviWorkspace + "\\" + olddir)
                        count=count+1
            if (count > 0):
                arcpy.AddMessage(str(count) + " fm_n_m or fo_n directories removed")
        else:
            arcpy.AddMessage("remove all raster datasets from File Geodatabase")
            # remove all raster datasets from File Geodatabase
            count=len(arcpy.ListRasters())
            if (count > 0):
                count=0
                for raster in arcpy.ListRasters():
                    if (raster[0:3] == "FM_" or raster[0:3] == "FO_"): 
                        arcpy.Delete_management(raster)
                        count=count+1
                if (count > 0):
                    arcpy.AddMessage(str(count) + " FM_ or FO_ raster datasets removed from " + enviWorkspace)

        # Open CSV file to test lines
        csvfile = open(output_folder + "\\FuzzyMembership.csv", "w")
        csvfile.write("Raster;Function;MidPoint;Midpoint Min;MidPoint Max;Midpoint Step;Spread;Spread Min;Spread Max;Spread Step;Result\n")

        # Run Fuzzy Memberships for each raster file and parameter combination
        arcpy.AddMessage("="*30)
        arcpy.AddMessage("Run Fuzzy Memberships...")

        fmcount = 0
        rasterNum = -1
        indexmax=[]
        files=[]
        required_files=1 # required count of FO files

        # Check input parameters of all rasters
        for memberParam in memberParams:
            rasterNum = rasterNum+1
            indexmax.append(-1)
            # memberparam = rc_till_co Gaussian 1 4 4 2 5 4 (raster function Midpoint Min Max Count Spread Min Max Count)
            fmparams = memberParam.split(' ')
            if (len(fmparams) != 8):
                arcpy.AddError ("ERROR: Wrong number of parameters in '" + memberParam + "'. Required: raster function Midpoint-Min Midpoint-Max Midpoint-Count Spread-Min Spread-Max Spread-Count")
                raise

            # Check Coordinate System of all Input Rasters must be same as True Positives Feature Class
            inputRaster=fmparams[0]
            memberType=fmparams[1]
            inputDescr = arcpy.Describe(inputRaster)
            coord_system=inputDescr.spatialReference.name
            arcpy.AddMessage("Input Raster: " + os.path.basename(inputRaster) + " " + inputDescr.dataType + " " + coord_system)
            if (true_coord_system != coord_system):
                arcpy.AddError("ERROR: Coordinate System must be " + true_coord_system)
                raise

            # Convert all member params to numeric

            # Midpoint Min value
            if fmparams[2] == "" or fmparams[2] == "#":
                arcpy.AddError("MidPoint Min value of " + str(inputRaster) + " is missing.")
                raise
            midmin=float(fmparams[2])

            # MidPoint Max value cannot be smaller than Midpoint Min
            if fmparams[3] == "" or fmparams[3] == "#":
                arcpy.AddError("MidPoint Max value of " + str(inputRaster) + " is missing.")
                raise
            midmax=float(fmparams[3])
            if (midmax < midmin):
                arcpy.AddError("ERROR: Midpoint Max cannot be smaller than Midpoint Min.") # Changed
                raise

            # MidPoint Count must be at least 1
            if fmparams[4] == "" or fmparams[4] == "#":
                arcpy.AddError("MidPoint Count value of " + str(inputRaster) + " is missing.")
                raise
            midcount=int(fmparams[4])
            if (midcount < 1):
                arcpy.AddError("ERROR: Midpoint Count must be at least 1.")
                raise

            # Calculate midpoint step
            midstep=0
            if (midmax > midmin): midstep=(midmax-midmin)/(midcount-1)
            if midstep == 0: midstep=1

            # Spread Min
            if fmparams[5] == "" or fmparams[5] == "#":
                arcpy.AddError("Spread Min value of " + str(inputRaster) + " is missing.")
                raise
            spreadmin=float(fmparams[5])

            # Spread Max cannot be smalled than Spread Min
            if fmparams[6] == "" or fmparams[6] == "#":
                arcpy.AddError("Spread Max value of " + str(inputRaster) + " is missing.")
                raise
            spreadmax=float(fmparams[6])
            if (spreadmax < spreadmin):
                arcpy.AddError("ERROR: Spread Max cannot be smalled than Spread Min.") # Changed
                raise

            # Spread Count must be at least 1
            if fmparams[7] == "" or fmparams[7] == "#":
                arcpy.AddError("Spread Count value of " + str(inputRaster) + " is missing.")
                raise
            spreadcount=int(fmparams[7])
            if (spreadcount < 1):
                arcpy.AddError("ERROR: Spread Count must be at least 1.")
                raise

            # Calculate the required number of FO files and spredstep
            required_files = required_files * midcount * spreadcount
            spreadstep=0
            if (spreadmax > spreadmin): spreadstep=(spreadmax-spreadmin)/(spreadcount-1)
            if spreadstep == 0: spreadstep=1

            # Loop asked count times from mimimum parameter values to maximum values
            midpoint = midmin
            while (midpoint <= midmax):
                spread=float(fmparams[5])
                while (spread <= spreadmax):
                    indexmax[rasterNum] = indexmax[rasterNum] + 1

                    # Run Fuzzy Memberships by function
                    fmout = "FM_" + str(rasterNum) + "_" + str(indexmax[rasterNum])
                    files.append(fmout)
                    arcpy.AddMessage (os.path.basename(inputRaster) + ", " + memberType + ", " + fmout)
                    csvfile.write (inputRaster + ";" + memberType + ";" + str(midpoint) + ";" + str(midmin) + ";" + str(midmax) + ";" + str(midstep) + ";" + str(spread) + ";" + str(spreadmin) + ";" + str(spreadmax) + ";" + str(spreadstep) + ";" + fmout + "\n")
                    fmcount=fmcount+1
                    
                    if (memberType == "Gaussian"):
                        outFzyMember = FuzzyMembership(inputRaster, FuzzyGaussian(midpoint, spread))
                    elif (memberType == "Large"):
                        outFzyMember = FuzzyMembership(inputRaster, FuzzyLarge(midpoint, spread))
                    elif (memberType == "Linear"):
                        outFzyMember = FuzzyMembership(inputRaster, FuzzyLinear(midpoint, spread))
                    elif (memberType == "MSLarge"):
                        outFzyMember = FuzzyMembership(inputRaster, FuzzyMSLarge(midpoint, spread))
                    elif (memberType == "MSSmall"):
                        outFzyMember = FuzzyMembership(inputRaster, FuzzyMSSmall(midpoint, spread))
                    elif (memberType == "Near"):
                        outFzyMember = FuzzyMembership(inputRaster, FuzzyNear(midpoint, spread))
                    elif (memberType == "Small"):
                        outFzyMember = FuzzyMembership(inputRaster, FuzzySmall(midpoint, spread))
                    outFzyMember.save(fmout)
                    spread=spread+spreadstep
                midpoint=midpoint+midstep

        # Close CSV file
        csvfile.close()
        arcpy.AddMessage(str(fmcount) + " FM outputs saved to " + env.workspace)
        arcpy.AddMessage("FM output names are written to " + env.workspace + "\\FuzzyMembership.csv")
        csvfile="?"

        # Define ROC Tool (Receiver Operator Characteristics)
        import arcgisscripting
        gp = arcgisscripting.create()
        parentfolder = os.path.dirname(sys.path[0])
        tbxpath = os.path.join(parentfolder,"toolbox\\arcsdm.pyt")
        arcpy.ImportToolbox(tbxpath)

        # Open CSV file to test lines
        csvfile = open(output_folder + "\\FuzzyOverlay.csv", "w")
        title = "Output" # Changed / added
        for i in range(rasterNum+1): # Changed / added
            title = title + ";Input"+str(i+1) # Changed / added
        title = title + "\n" # Changed / added
        csvfile.write(title) # Changed

        # Run Fuzzy Overlays and ROC 
        arcpy.AddMessage("="*30)
        arcpy.AddMessage("Running Fuzzy Overlays and ROC... " + str(required_files) + " FO output files should be created...")
        num = -1
        index = []
        indexmin = []
        indexmin.append(0)
        index.append(0)

        # Calculate limit values of FO file indexes
        j = 0
        sum = 0
        while j < rasterNum+1:
            sum = sum+indexmax[j]+1
            indexmin.append(sum)
            index.append(sum)
            j = j+1

        # Write merging expressions of FM files
        # The calculation is done with indices because the number of minimum and 
        # maximum values of rasters and parameters varies.
        i = 0
        while i < rasterNum:
            while index[i] < index[i+1]:
                j = 0
                while j < indexmin[i+1]:
                    fo_msg = files[j]
                    fo_csv = files[j]
                    fo_ovr = [files[j]]
                    k = 1
                    while k < rasterNum+1:
                        fo_msg = fo_msg + " + " + files[index[k]]
                        fo_csv = fo_csv + ";" + files[index[k]]
                        fo_ovr.append(files[index[k]])
                        k = k + 1
                    num=num+1
                    arcpy.AddMessage ("FO_" + str(num) + " = " + fo_msg)
                    csvfile.write ("FO_" + str(num) + ";" + fo_csv + "\n")
                    
                    overlays = [fo_ovr]
                    if (overlayParams[0] == "Gamma"):
                        outFzyOverlay = FuzzyOverlay(fo_ovr, "Gamma", overlayParams[1]) # Changed
                    else:
                        outFzyOverlay = FuzzyOverlay(fo_ovr, overlayParams[0]) # Changed
                    outFzyOverlay.save("FO_" + str(num))
                    
                    ##arcpy.AddMessage("FO_" + str(num) + " saved to " + env.workspace)
                    result = arcpy.ROCTool_ArcSDM(truepositives, "", "FO_" + str(num), output_folder)

                    j = j+1
                index[i+1] = index[i+1]+1
                i1 = i+1
                while i1 < rasterNum and index[i1] >= indexmin[i1+1]:
                    index[i1] = indexmin[i1]
                    index[i1+1] = index[i1+1]+1
                    i1 = i1+1
                if i1+1 >= len(index) or index[i1+1] > indexmin[len(indexmin)-1] or required_files == (num+1):
                    i = len(indexmin)+1
                    break
            i = i+1
        num = num+1
        arcpy.AddMessage(str(num) + " FO outputs saved to " + env.workspace)
        arcpy.AddMessage("FO output names are written to " + env.workspace + "\\FuzzyOverlay.csv")
        csvfile.close()
        csvfile="?"
        if num != required_files:
            arcpy.AddError("ERROR: The number of FO output files is " + str(num) + " but they should have become " + str(required_files) + ".")
            raise

        arcpy.AddMessage(" ")
        arcpy.AddMessage("Get AUC values from ROC output databases...")
        
        # Open CSV file to ROC results
        csvfile = open(output_folder + "\\FuzzyROC.csv", "w")
        csvfile.write("Model;Auc\n")

        # Get AUC values from ROC output databases
        fields = ['MODEL', 'AUC']
        arcpy.env.workspace = output_folder
        for inputfile in arcpy.ListFiles("results*.dbf"):
            with arcpy.da.SearchCursor(inputfile, fields, 'OID=0') as cursor:
                for row in cursor:
                    csvfile.write(row[0] + ";" + str(row[1]) + "\n")
                del row
            del cursor
        csvfile.close()
        csvfile="?"
        arcpy.AddMessage("ROC results are written to " + env.workspace + "\\FuzzyROC.csv")
        import gc
        gc.collect()

        # Restore Workspace
        arcpy.env.workspace = enviWorkspace

    except arcpy.ExecuteError:
        arcpy.AddMessage("*"*30)
        if csvfile!="?": csvfile.close()
        arcpy.AddError(arcpy.GetMessages(2))
        arcpy.AddError('Aborting FuzzyROC2 (1)')
    except:
        arcpy.AddMessage("*"*30)
        tb = sys.exc_info()[2]
        tbinfo = traceback.format_tb(tb)[0]
        pymsg = "PYTHON ERRORS:\nTraceback info:\n" + tbinfo + "\nError Info:\n" + str(sys.exc_info()[1])
        msgs = "ArcPy ERRORS:\n" + arcpy.GetMessages(2) + "\n"
        arcpy.AddError(pymsg)
        arcpy.AddError(msgs)
        if csvfile!="?": csvfile.close()
        arcpy.AddError('Aborting FuzzyROC2 (2)')
