# -*- coding: utf-8 -*-
"""
    FuzzyROC - ArcSDM 5 for ArcGis pro 
    Arto Laiho, Geological Survey of Finland, 11-14.5.2020
    Added Midpoint and Spread value missing tests 11.8.2020/AL
"""
# Import system modules
import sys,os,traceback
import arcpy
import arcsdm
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
        # Param 0: Input raster names, GPRasterLayer, multiValue=1, Required, Input
        # parameters[0] = raster name 1;raster name 2;...
        inputRasters = parameters[0].valueAsText.split(';')
        if (len(inputRasters) < 2):
            arcpy.AddError("ERROR: Input raster count must be at least 2.")
            raise

        # Param 1: Fuzzy Membership Parameters, DETable, Required, Input
        # columns = Membership type, Midpoint Min, Midpoint Max, Midpoint Count, Spread Min, Spread Max, Spread Count
        # Membership types: Gaussian, Small, Large, Near, MSLarge, MSSmall, Linear
        # parameters[1] = Gaussian 1 4 4 2 5 4;Large 2 5 4 3 4 2;Small 3 4 2 4 5 2;...
        memberParams = parameters[1].valueAsText.split(';')

        # Param 2: Fuzzy Overlay Parameters, DETable, Required, Input
        # columns = Overlay type, Parameter
        # Overlay types: And, Or, Product, Sum, Gamma
        # parameters[2] = And # (tai: Gamma 5)
        overlayParams = parameters[2].valueAsText.split(' ')

        # Param 3: ROC True Positives Feature Class, DEFeatureClass, Required, Input
        # parameters[3] = feature class name
        truepositives = parameters[3].valueAsText
        trueDescr = arcpy.Describe(truepositives)
        true_coord_system=trueDescr.spatialReference.name
        arcpy.AddMessage("ROC True Positives: " + os.path.basename(truepositives) + " " + trueDescr.dataType + " " + true_coord_system)

        # Param 4: ROC Destination Folder, DEFolder, Required, Input, "File System"
        output_folder = parameters[4].valueAsText
        arcpy.AddMessage("output_folder = " + str(output_folder))

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
            #arcpy.AddMessage("len(arcpy.ListRasters()) = " + str(count))
            if (count > 0):
                count=0
                for raster in arcpy.ListRasters():
                    #arcpy.AddMessage(raster)
                    if (raster[0:3] == "FM_" or raster[0:3] == "FO_"): 
                        arcpy.Delete_management(raster)
                        count=count+1
                if (count > 0):
                    arcpy.AddMessage(str(count) + " FM_ or FO_ raster datasets removed from " + enviWorkspace)

        # Open CSV file to test lines
        csvfile = open(output_folder + "\\FuzzyMembership.csv", "w")
        csvfile.write("Raster;Function;Midpoint Min;MidPoint Max;Midpoint Step;Spread Min;Spread Max;Spread Step;Result\n")

        # Run Fuzzy Memberships for each raster file and parameter combination
        arcpy.AddMessage("="*30)
        arcpy.AddMessage("Run Fuzzy Memberships...")
        ir=-1
        outputs=""
        fmcount=0
        
        for inputRaster in inputRasters:
            # Check Coordinate System of all Input Rasters must be same as True Positives Feature Class
            inputDescr = arcpy.Describe(inputRaster)
            coord_system=inputDescr.spatialReference.name
            arcpy.AddMessage("Input Raster: " + os.path.basename(inputRaster) + " " + inputDescr.dataType + " " + coord_system)
            #arcpy.AddMessage("Coordinate System of " + inputRaster + " is " + str(coord_system))
            if (true_coord_system != coord_system):
                arcpy.AddError("ERROR: Coordinate System must be " + true_coord_system)
                raise

            # Run Fuzzy Membership
            ir=ir+1  # first one is 0
            fmnum=-1 # first one is 0
            for memberParam in memberParams:
                # memberparam = # Gaussian 1 4 4 2 5 4 (function Midpoint Min Max Count Spread Min Max Count)
                fmparams = memberParam.split(' ')
                if (len(fmparams) != 7):
                    arcpy.AddError ("ERROR: Wrong number of parameters in '" + memberParam + "'. Required: function Midpoint-Min Midpoint-Max Midpoint-Count Spread-Min Spread-Max Spread-Count")
                    raise

                # Convert member params to numeric
                if fmparams[1] == "" or fmparams[1] == "#":
                    arcpy.AddError("MidPoint Min value of " + str(inputRaster) + " is missing.")
                    raise
                midmin=float(fmparams[1])
                if fmparams[2] == "" or fmparams[2] == "#":
                    arcpy.AddError("MidPoint Max value of " + str(inputRaster) + " is missing.")
                    raise
                midmax=float(fmparams[2])
                if (midmax < midmin):
                    arcpy.AddError("ERROR: Midpoint Max must be less than Midpoint Min.")
                    raise
                if fmparams[3] == "" or fmparams[3] == "#":
                    arcpy.AddError("MidPoint Count value of " + str(inputRaster) + " is missing.")
                    raise
                midcount=int(fmparams[3])
                if (midcount < 1):
                    arcpy.AddError("ERROR: Midpoint Count must be at least 1.")
                    raise
                midstep=0
                if (midmax > midmin):
                    midstep=(midmax-midmin)/(midcount-1)
                if fmparams[4] == "" or fmparams[4] == "#":
                    arcpy.AddError("Spread Min value of " + str(inputRaster) + " is missing.")
                    raise
                spreadmin=float(fmparams[4])
                if fmparams[5] == "" or fmparams[5] == "#":
                    arcpy.AddError("Spread Max value of " + str(inputRaster) + " is missing.")
                    raise
                spreadmax=float(fmparams[5])
                if (spreadmax < spreadmin):
                    arcpy.AddError("ERROR: Spread Max must be less than Spread Min.")
                    raise
                if fmparams[6] == "" or fmparams[6] == "#":
                    arcpy.AddError("Spread Count value of " + str(inputRaster) + " is missing.")
                    raise
                spreadcount=int(fmparams[6])
                if (spreadcount < 1):
                    arcpy.AddError("ERROR: Spread Count must be at least 1.")
                    raise
                spreadstep=0
                if (spreadmax > spreadmin):
                    spreadstep=(spreadmax-spreadmin)/(spreadcount-1)

                # Loop asked count times from mimimum parameter values to maximum values
                while (midmin <= midmax):
                    spreadmin=float(fmparams[4])
                    while (spreadmin <= spreadmax):
                        fmnum=fmnum+1
                        fmout = "FM_" + str(ir) + "_" + str(fmnum)
                        arcpy.AddMessage (fmout)
                        csvfile.write (inputRaster + ";" + fmparams[0] + ";" + str(midmin) + ";" + str(midmax) + ";" + str(midstep) + ";" + str(spreadmin) + ";" + str(spreadmax) + ";" + str(spreadstep) + ";" + fmout + "\n")
                        fmcount=fmcount+1

                        # Run Fuzzy Memberships by function
                        if (fmparams[0] == "Gaussian"):
                            outFzyMember = FuzzyMembership(inputRaster, FuzzyGaussian(midmin, spreadmin))
                        elif (fmparams[0] == "Large"):
                            outFzyMember = FuzzyMembership(inputRaster, FuzzyLarge(midmin, spreadmin))
                        elif (fmparams[0] == "Linear"):
                            outFzyMember = FuzzyMembership(inputRaster, FuzzyLinear(midmin, spreadmin))
                        elif (fmparams[0] == "MSLarge"):
                            outFzyMember = FuzzyMembership(inputRaster, FuzzyMSLarge(midmin, spreadmin))
                        elif (fmparams[0] == "MSSmall"):
                            outFzyMember = FuzzyMembership(inputRaster, FuzzyMSSmall(midmin, spreadmin))
                        elif (fmparams[0] == "Near"):
                            outFzyMember = FuzzyMembership(inputRaster, FuzzyNear(midmin, spreadmin))
                        elif (fmparams[0] == "Small"):
                            outFzyMember = FuzzyMembership(inputRaster, FuzzySmall(midmin, spreadmin))
                        outFzyMember.save(fmout)
                        if (spreadstep == 0):
                            break
                        spreadmin=spreadmin+spreadstep
                    if (midstep == 0):
                        break
                    midmin=midmin+midstep

        csvfile.close()
        arcpy.AddMessage(str(fmcount) + " FM outputs saved to " + env.workspace)
        csvfile="?"
        
        # Define ROC Tool (Receiver Operator Characteristics)
        import arcgisscripting
        gp = arcgisscripting.create()
        parentfolder = os.path.dirname(sys.path[0])
        tbxpath = os.path.join(parentfolder,"toolbox\\arcsdm.pyt")
        arcpy.ImportToolbox(tbxpath)

        # Open CSV file to test lines
        csvfile = open(output_folder + "\\FuzzyOverlay.csv", "w")
        csvfile.write("Output;Input 1;Input 2\n")

        # Run Fuzzy Overlays and ROC 
        arcpy.AddMessage("="*30)
        arcpy.AddMessage("Run Fuzzy Overlays and ROC...")
        num=-1
        #if (fmnum == 0):
        #    fmnum=1
        for i in range(0, ir):              # raster files
            for j in range(0, fmnum+1):     # output tables or output files (FM_x_y) per raster file, first half
                for k in range(0, fmnum+1): # output tables or output files (FM_x_y) per raster file, last half
                    num=num+1
                    arcpy.AddMessage ("FO_" + str(num) + " = FM_" + str(i) + "_" + str(j) + " + FM_" + str(i+1) + "_" + str(k))
                    csvfile.write ("FO_" + str(num) + ";FM_" + str(i) + "_" + str(j) + ";FM_" + str(i+1) + "_" + str(k) + "\n")
                    overlays = ["FM_" + str(i) + "_" + str(j), "FM_" + str(i+1) + "_" + str(k)]
                    if (overlayParams[0] == "Gamma"):
                        outFzyOverlay = FuzzyOverlay(overlays, "Gamma", overlayParams[1])
                    else:
                        outFzyOverlay = FuzzyOverlay(overlays, overlayParams[0])
                    outFzyOverlay.save("FO_" + str(num))
                    #arcpy.AddMessage("FO_" + str(num) + " saved to " + env.workspace)
                    result = arcpy.ROCTool_ArcSDM(truepositives, "", "FO_" + str(num), output_folder)

        arcpy.AddMessage(str(num+1) + " FO outputs saved to " + env.workspace)
        csvfile.close()
        csvfile="?"

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
        csvfile.close()
        csvfile="?"

        # Restore Workspace
        arcpy.env.workspace = enviWorkspace

    except arcpy.ExecuteError:
        arcpy.AddMessage("*"*30)
        if (csvfile != "?"):
            csvfile.close()
        arcpy.AddError(arcpy.GetMessages(2))
        arcpy.AddError ('Aborting FuzzyROC (1)')
    except:
        arcpy.AddMessage("*"*30)
        tb = sys.exc_info()[2]
        tbinfo = traceback.format_tb(tb)[0]
        pymsg = "PYTHON ERRORS:\nTraceback info:\n" + tbinfo + "\nError Info:\n" + str(sys.exc_info()[1])
        msgs = "ArcPy ERRORS:\n" + arcpy.GetMessages(2) + "\n"
        arcpy.AddError(pymsg)
        arcpy.AddError(msgs)
        if (csvfile != "?"):
            csvfile.close()
        arcpy.AddError ('Aborting FuzzyROC (2)')
