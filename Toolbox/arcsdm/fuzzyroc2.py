# -*- coding: utf-8 -*-
"""
    FuzzyROC - ArcSDM 5 for ArcGis pro 
    Arto Laiho & Johanna Torppa, Geological Survey of Finland, 11.5-23.6.2020, Version 2
    Added Midpoint and Spread value missing tests 11.8.2020/AL
    Fuzzy Overlay file selection has completely rewritten 15.10.2020/AL
    Fuzzy Membership plots added 10.11.2020/AL
"""
# Import system modules
import sys,os,traceback
import arcpy
import arcsdm
import math
import numpy
import matplotlib.pyplot as plt
from arcpy.sa import *
from arcpy import env

def Execute(self, parameters, messages):
    try:
        # Save Workspace and set constants and variables
        enviWorkspace = env.workspace
        plotname = ""
        csvfile = "?"

        #Get and print version information
        with open (os.path.join(os.path.dirname(__file__), "arcsdm_version.txt"), "r") as myfile:
            data=myfile.readlines()
        arcpy.AddMessage("%-20s %s" % ("", data[0]) ); 
        installinfo = arcpy.GetInstallInfo ();
        arcpy.AddMessage("%-20s %s (%s)" % ("Arcgis environment: ", installinfo['ProductName'] , installinfo['Version'] ));

        # Load parameters...
        # Param 0: Input rasters, Fuzzy Membership functions and parameters, DETable, multiValue=1, Required, Input
        # columns = Input raster, Membership type, Midpoint Min, Midpoint Max, Midpoint Count, Spread Min, Spread Max, Spread Count
        # Membership types: Small, Large
        # Example: rc_till_co Gaussian 1 4 4 2 5 4;rc_till_ca Large 2 5 4 3 4 2;rc_till_au Small 3 4 2 4 5 2;...
        memberParams = parameters[0].valueAsText.split(';')
        arcpy.AddMessage("memberParams = " + str(memberParams))

        # Param 1: Draw only Fuzzy Membership plots
        plots = parameters[1].value
        if (not plots and len(memberParams) < 2):
            arcpy.AddError ("ERROR: Minimum number of Input Rasters is 2.")
            raise arcpy.ExecuteError

        # Param 2: ROC True Positives Feature Class, DEFeatureClass, Required, Input
        truepositives = parameters[2].valueAsText
        trueDescr = arcpy.Describe(truepositives)
        true_coord_system=trueDescr.spatialReference.name

        # Param 3: Output Folder, DEFolder, Required, Input, "File System"
        output_folder = parameters[3].valueAsText

        # Param 4: Fuzzy Overlay Parameters, DETable, Required, Input
        # when plots = False
        # columns = Overlay type, Parameter
        # Overlay types: And, Or, Product, Sum, Gamma
        # parameters[1] = And # (or: Gamma 5)
        overlayParams = parameters[4].valueAsText.split(' ')

        # Param 5: Plot display method
        # when plots = True
        page_type = ""
        display_method = parameters[5].valueAsText
        if display_method == "To Window(s)":
            page_type = "Win"
        elif display_method == "To PDF file(s)":
            page_type = "pdf"
        elif display_method == "To PNG file(s)":
            page_type = "png"

        # Remove old files and datasets
        cleanup(output_folder, enviWorkspace)

        # Check and collect input parameters of all rasters
        inputRasters = []
        memberTypes = []
        midmins = []
        midmaxes = []
        midcounts = []
        midpointcount = 0
        midsteps = []
        spreadmins = []
        spreadmaxes = []
        spreadcounts = []
        spreadsteps = []
        spreadcountcount = 0

        for memberParam in memberParams:
            fmparams = memberParam.split(' ')
            if (len(fmparams) != 8):
                arcpy.AddError ("ERROR: Wrong number of parameters in '" + memberParam + "'. Required: raster function Midpoint-Min Midpoint-Max Midpoint-Count Spread-Min Spread-Max Spread-Count")
                raise arcpy.ExecuteError

            # Input Raster Name and Membership type
            inputRasters.append(fmparams[0])
            memberTypes.append(fmparams[1])
            if len(plotname) > 0: plotname = plotname + "_"
            plotname = plotname + os.path.basename(fmparams[0]) + "-" + fmparams[1]

            # Midpoint Min value
            if fmparams[2] == "" or fmparams[2] == "#":
                arcpy.AddError("MidPoint Min value of " + str(fmparams[0]) + " is missing.")
                raise arcpy.ExecuteError
            midmin=float(fmparams[2])
            midmins.append(midmin)

            # MidPoint Max value cannot be smaller than Midpoint Min
            if fmparams[3] == "" or fmparams[3] == "#":
                arcpy.AddError("MidPoint Max value of " + str(fmparams[0]) + " is missing.")
                raise arcpy.ExecuteError
            midmax=float(fmparams[3])
            if (midmax < midmin):
                arcpy.AddError("ERROR: Midpoint Max cannot be smaller than Midpoint Min.")
                raise arcpy.ExecuteError
            midmaxes.append(midmax)

            # MidPoint Count must be at least 1
            if fmparams[4] == "" or fmparams[4] == "#":
                arcpy.AddError("MidPoint Count value of " + str(fmparams[0]) + " is missing.")
                raise arcpy.ExecuteError
            midcount=int(fmparams[4])
            if (midcount < 1):
                arcpy.AddError("ERROR: Midpoint Count must be at least 1.")
                raise arcpy.ExecuteError
            midcounts.append(midcount)
            midpointcount = midpointcount + midcount

            # Calculate midpoint step
            midstep=0
            if (midmax > midmin): midstep=(midmax-midmin)/(midcount-1)
            if midstep == 0: midstep=1
            midsteps.append(midstep)

            # Spread Min
            if fmparams[5] == "" or fmparams[5] == "#":
                arcpy.AddError("Spread Min value of " + str(fmparams[0]) + " is missing.")
                raise arcpy.ExecuteError
            spreadmin=float(fmparams[5])
            spreadmins.append(spreadmin)

            # Spread Max cannot be smalled than Spread Min
            if fmparams[6] == "" or fmparams[6] == "#":
                arcpy.AddError("Spread Max value of " + str(fmparams[0]) + " is missing.")
                raise arcpy.ExecuteError
            spreadmax=float(fmparams[6])
            if (spreadmax < spreadmin):
                arcpy.AddError("ERROR: Spread Max cannot be smalled than Spread Min.")
                raise arcpy.ExecuteError
            spreadmaxes.append(spreadmax)

            # Spread Count must be at least 1
            if fmparams[7] == "" or fmparams[7] == "#":
                arcpy.AddError("Spread Count value of " + str(fmparams[0]) + " is missing.")
                raise arcpy.ExecuteError
            spreadcount=int(fmparams[7])
            if (spreadcount < 1):
                arcpy.AddError("ERROR: Spread Count must be at least 1.")
                raise arcpy.ExecuteError
            spreadcounts.append(spreadcount)
            spreadcountcount = spreadcountcount + spreadcount

            # Calculate spread step
            spreadstep=0
            if (spreadmax > spreadmin): spreadstep=(spreadmax-spreadmin)/(spreadcount-1)
            if spreadstep == 0: spreadstep=1
            spreadsteps.append(spreadstep)

        # Draw only Fuzzy Membership plots
        if plots:
            draw_plots(memberParams, memberTypes, truepositives, inputRasters, output_folder, page_type, 
            spreadcountcount, spreadmins, spreadmaxes, spreadsteps, midpointcount, midmins, midmaxes, midsteps, 
            plotname)
        else:
            # Run Fuzzy Memberships, Fuzzy Overlays and ROC Tool
            calculation(inputRasters, output_folder, true_coord_system, memberParams, memberTypes, 
            midcounts, midmins, midmaxes, midsteps, spreadcounts, spreadmins, spreadmaxes, spreadsteps, 
            overlayParams, truepositives, enviWorkspace)

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

# Cleanup
# -------
def cleanup(output_folder, enviWorkspace):
    try:
        # Remove old files from ROC output folder if any
        arcpy.AddMessage("="*30)
        arcpy.AddMessage("Cleaning up workspace...")
        arcpy.env.workspace = output_folder
        count=len(os.listdir(output_folder))
        if (count > 0):
            count=0
            for oldfile in os.listdir(output_folder):
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
            arcpy.AddMessage("Removing all FM_ files and all FO_ files from workspace File System...")

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

            # Remove all fm_ and fo_ subdirectories and subfiles
            arcpy.AddMessage("Removing all fm_ and fo_ subdirectories and subfiles...")
            count=0
            for olddir in os.listdir(enviWorkspace):
                if os.path.isdir(os.path.join(enviWorkspace,olddir)):
                    if (olddir[0:3] == "fm_" or olddir[0:3] == "fo_"):
                        shutil.rmtree(enviWorkspace + "\\" + olddir)
                        count=count+1
            if (count > 0):
                arcpy.AddMessage(str(count) + " fm_n_m or fo_n directories removed")
        else:
            # Remove all raster datasets from File Geodatabase
            arcpy.AddMessage("Removing all raster datasets from File Geodatabase...")
            count=len(arcpy.ListRasters())
            if (count > 0):
                count=0
                for raster in arcpy.ListRasters():
                    if (raster[0:3] == "FM_" or raster[0:3] == "FO_"): 
                        arcpy.Delete_management(raster)
                        count=count+1
                if (count > 0):
                    arcpy.AddMessage(str(count) + " FM_ or FO_ raster datasets removed from " + enviWorkspace)

        # Remove old OutputPoints.shp files
        for oldfile in os.listdir(output_folder):
            if "OutputPoints" in oldfile:
                try:
                    os.remove(output_folder + "\\" + oldfile)
                except:
                    arcpy.AddMessage("Cannot remove " + oldfile)
    except:
        arcpy.AddMessage("*"*30)
        tb = sys.exc_info()[2]
        tbinfo = traceback.format_tb(tb)[0]
        pymsg = "PYTHON ERRORS:\nTraceback info:\n" + tbinfo + "\nError Info:\n" + str(sys.exc_info()[1])
        msgs = "ArcPy ERRORS:\n" + arcpy.GetMessages(2) + "\n"
        arcpy.AddError(pymsg)
        arcpy.AddError(msgs)

# Draw only Fuzzy Membership plots
# --------------------------------
def draw_plots(memberParams, memberTypes, truepositives, inputRasters, output_folder, page_type, 
spreadcountcount, spreadmins, spreadmaxes, spreadsteps, midpointcount, midmins, midmaxes, midsteps, 
plotname):
    arcpy.AddMessage("="*30)
    arcpy.AddMessage("Preparing plots...")

    # Set plot scale values
    plotcols = midpointcount
    plotrows = spreadcountcount + 1
    if page_type == "Win":
        if plotrows > 9: plotrows = 9
        if plotcols > 15: plotcols = 15
    pagewidth = round(midpointcount * 1.4)
    pagelength = round(plotrows * 1.2)
    xmax = 10
    ymax = 1
    number = 0
    page_number = 1

    try:
        for index in range(0, len(memberParams)):
            inputRaster = inputRasters[index]
            arcpy.AddMessage("Preparing " + os.path.basename(inputRaster) + "...")
            output_points = arcpy.CreateUniqueName("OutputPoints.shp", output_folder)
            ExtractValuesToPoints(truepositives, inputRaster, output_points, "INTERPOLATE", "VALUE_ONLY")

            # Extract the RASTERVALU values from the output_points shape into a RASTERVALUES list
            rastervalues=[]
            with arcpy.da.SearchCursor(output_points, "RASTERVALU") as cursor:
                for row in cursor:
                    rastervalues.append(row[0])

            # Convert the input raster to raster object and to NumPy table
            raster = arcpy.Raster(inputRaster)
            value_minimum = raster.minimum
            value_maximum = raster.maximum
            desc = arcpy.Describe(inputRaster)

            # If the raster is not an integer type, then there is no need to futz with NaNs
            if desc.pixelType.startswith('F'):
                input_raster_array = arcpy.RasterToNumPyArray(inputRaster, nodata_to_value=numpy.NaN)

            # Since integer arrays can't contain NaNs, you must do wackiness
            else:
                # Convert to an array, setting NoData cells to the unique value
                null_value = value_minimum - 1 # value that does not occur in the raster
                input_raster_array = arcpy.RasterToNumPyArray(inputRaster, nodata_to_value=null_value)
                input_raster_array = input_raster_array.astype('float')           # Convert the array of integers to an array of floats
                input_raster_array[input_raster_array==null_value] = numpy.NaN    # Replace the placeholder null value with NaNs

            # Calculate percentile and bin edges of input raster array
            vhis_arange = numpy.arange(0.0, 105.0, 5.0) # 28.10.20
            vhis = numpy.nanpercentile(input_raster_array, vhis_arange, interpolation='midpoint') # 28.10.20
            vhis = numpy.unique(vhis)
            bin_edges = numpy.r_[-1, 0.5 * (vhis[:-1] + vhis[1:]), 10000]
            rastercounts, rasterbins = numpy.histogram(input_raster_array, bins=bin_edges, range=(0, value_maximum))

            # If rastercounts list is shorter than rasterbins, remove last element from rasterbins list
            if len(rasterbins) > len(rastercounts):                 
                rasterbins = numpy.delete(rasterbins,len(rasterbins)-1)

            # Define font size and spaces between charts
            fig = plt.figure("PLOT_" + plotname + "_Pro_" + str(page_number), figsize=(pagewidth,pagelength))
            fig.suptitle("PLOT_" + plotname + "_Pro_" + str(page_number))
            plt.rcParams.update({'axes.titlesize': 10})
            plt.rcParams.update({'axes.labelsize': 10})
            plt.rcParams.update({'xtick.labelsize': 7})
            plt.rcParams.update({'ytick.labelsize': 7})
            plt.subplots_adjust(hspace=0.7, wspace=0.7)

            # Draw training points histogram
            # ------------------------------
            number = number + 1
            fig.add_subplot(plotrows,plotcols,number)
            plt.title(os.path.basename(truepositives))
            arcpy.AddMessage("Plotting " + os.path.basename(truepositives))
            plt.axis([0, len(rastervalues), 0, max(rastervalues)])
            plt.bar(numpy.arange(len(rastervalues)), rastervalues, align='center', width=0.5)

            # Draw input raster histogram
            # ---------------------------
            number = number + 1
            fig.add_subplot(plotrows,plotcols,number)
            plt.title(os.path.basename(inputRaster))
            arcpy.AddMessage("Plotting " + os.path.basename(inputRaster))
            plt.axis([0, len(rastercounts), 0, max(rastercounts)])
            plt.bar(rasterbins, rastercounts, align='center', width=0.5)

            # Calculate rastervalues histogram
            number = number + plotcols - 2 # continues on next line
            shpcounts, shpbins = numpy.histogram(rastervalues, bins=bin_edges, range=(0, value_maximum))

            # Remove zero values from rastercounts list
            indhis = []
            for i in range(0,len(rastercounts)):
                if rastercounts[i] != 0: indhis.append(i)

            # Calculate known points locations
            shpindices = shpcounts[indhis]
            ctrat = []
            for i in range(0, len(shpindices)):
                ctrat.append(float(shpindices[i]) / float(rastercounts[i]))
            maxctrat = float(max(ctrat))
            for i in range(0, len(ctrat)):
                ctrat[i] = ctrat[i] / maxctrat

            # Draw Fuzzy Membership curves
            arcpy.AddMessage("Plotting Fuzzy Membership plots...")
            spread = spreadmins[index]
            while spread <= spreadmaxes[index]:
                midpoint = midmins[index]
                while midpoint <= midmaxes[index]:
                    number = number + 1
                    if number > plotrows*plotcols:
                        new_page(plotname, page_number, page_type, fig)
                        page_number = page_number+1
                        number = 1
                        arcpy.AddMessage("figure " + str(page_number))
                        fig = plt.figure("PLOT_" + plotname + "_Pro_" + str(page_number), figsize=(pagewidth,pagelength))
                        fig.suptitle("PLOT_" + plotname + "_Pro_" + str(page_number))
                        plt.subplots_adjust(hspace=0.7, wspace=0.7)

                    # Calculate draw points
                    xvalues=[]
                    yvalues=[]
                    x = 0
                    xstep = xmax / 50.0
                    while x <= xmax:
                        xvalues.append(x)
                        try:
                            if memberTypes[index] == "Small":
                                yvalues.append(1.0/(1.0+float(float(x)/float(midpoint))**spread))
                            elif memberTypes[index] == "Large":
                                yvalues.append(1.0/(1.0+float(float(x)/float(midpoint))**(-spread)))
                        except:
                            arcpy.AddMessage("x = " + str(x) + ", midpoint = " + str(midpoint) + ", spread = " + str(spread))
                            yvalues.append(0.0)
                        x = x + xstep

                    # Draw chart
                    arcpy.AddMessage("Plotting " + str(number) + ": midpoint:" + str(midpoint) + ", spread:" + str(spread) + "...")
                    ax = fig.add_subplot(plotrows,plotcols,number)
                    plt.title("m:" + str(round(midpoint,1)) + ", s:" + str(round(spread,1)))
                    plt.axis([0, int(value_maximum), 0, ymax])
                    plt.plot(xvalues, yvalues)
                    plt.plot(rasterbins, ctrat, color='red', marker='o', markersize=3, linewidth=0)
                    midpoint = midpoint + midsteps[index]
                spread = spread + spreadsteps[index]

                # plotting continues on next line
                if page_type == "Win" and midpointcount > 15:
                    while int(number/plotcols)*plotcols != number: number = number + 1
            arcpy.AddMessage("Plots ready.")

        new_page(plotname, page_number, page_type, fig)
        plt.close("all")

    except:
        arcpy.AddMessage("*"*30)
        tb = sys.exc_info()[2]
        tbinfo = traceback.format_tb(tb)[0]
        pymsg = "PYTHON ERRORS:\nTraceback info:\n" + tbinfo + "\nError Info:\n" + str(sys.exc_info()[1])
        msgs = "ArcPy ERRORS:\n" + arcpy.GetMessages(2) + "\n"
        arcpy.AddError(pymsg)
        arcpy.AddError(msgs)
        arcpy.AddError('Aborting FuzzyROC2')

# Save plot page
# --------------
def new_page(plotname, page_number, page_type, fig):
    arcpy.AddMessage("Save page " + plotname + ", page_number " + str(page_number))
    # Display charts to Window
    if page_type == "Win":
        plt.show()
    else:
        # Save charts to PNG or tp PDF file
        if str(arcpy.GetInstallInfo()['ProductName']) == "ArcGISPro":
            # ArcGIS Pro
            plt.savefig("C:/ArcSDM/work/PLOT_" + plotname + "_Pro_" + str(page_number) + "." + page_type, bbox_inches='tight')
            arcpy.AddMessage("C:/ArcSDM/work/PLOT_" + plotname + "_Pro_" + str(page_number) + "." + page_type + " done")
        else:
            # ArcMap
            plt.savefig("C:/ArcSDM/work/PLOT_" + plotname + "_Map_" + str(page_number) + "." + page_type, bbox_inches='tight')
            arcpy.AddMessage("C:/ArcSDM/work/PLOT_" + plotname + "_Map_" + str(page_number) + "." + page_type + " done")
    plt.close()

# Run Fuzzy Memberships, Fuzzy Overlays and ROC Tool
# --------------------------------------------------
def calculation (inputRasters, output_folder, true_coord_system, memberParams, memberTypes, 
midcounts, midmins, midmaxes, midsteps, spreadcounts, spreadmins, spreadmaxes, spreadsteps, 
overlayParams, truepositives, enviWorkspace):
    csvfile = "?"

    try:
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

        # Let's start
        for index in range(0, len(memberParams)):
            rasterNum = rasterNum+1
            indexmax.append(-1)

            # Calculate the required number of FO files and spredstep
            required_files = required_files * midcounts[index] * spreadcounts[index]

            # Check Coordinate System of all Input Rasters must be same as True Positives Feature Class
            inputRaster = inputRasters[index]
            memberType = memberTypes[index]
            inputDescr = arcpy.Describe(inputRaster)
            coord_system=inputDescr.spatialReference.name
            arcpy.AddMessage("Input Raster: " + os.path.basename(inputRaster) + " " + inputDescr.dataType + " " + coord_system)
            if (true_coord_system != coord_system):
                arcpy.AddError("ERROR: Coordinate System must be " + true_coord_system)
                raise arcpy.ExecuteError

            # Loop asked count times from mimimum parameter values to maximum values
            midpoint = midmins[index]
            while (midpoint <= midmaxes[index]):
                spread = spreadmins[index]
                while (spread <= spreadmaxes[index]):
                    indexmax[rasterNum] = indexmax[rasterNum] + 1

                    # Run Fuzzy Memberships by function
                    fmout = "FM_" + str(rasterNum) + "_" + str(indexmax[rasterNum])
                    files.append(fmout)
                    arcpy.AddMessage (os.path.basename(inputRaster) + ", " + memberType + ", " + fmout)
                    csvfile.write (inputRaster + ";" + memberType + ";" + str(midpoint) + ";" + str(midmins[index]) + ";" + str(midmaxes[index]) + ";" + str(midsteps[index]) + ";" + str(spread) + ";" + str(spreadmins[index]) + ";" + str(spreadmaxes[index]) + ";" + str(spreadsteps[index]) + ";" + fmout + "\n")
                    fmcount=fmcount+1
                    
                    if (memberType == "Large"):
                        outFzyMember = FuzzyMembership(inputRaster, FuzzyLarge(midpoint, spread))
                    elif (memberType == "Small"):
                        outFzyMember = FuzzyMembership(inputRaster, FuzzySmall(midpoint, spread))
                    outFzyMember.save(fmout)
                    spread = spread+spreadsteps[index]
                midpoint = midpoint+midsteps[index]

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
        title = "Output"
        for i in range(rasterNum+1):
            title = title + ";Input"+str(i+1)
        title = title + "\n"
        csvfile.write(title)

        # Run Fuzzy Overlays and ROC 
        arcpy.AddMessage("="*30)
        arcpy.AddMessage("Running Fuzzy Overlays and ROC... " + str(required_files) + " FO output files should be created...")
        num = -1
        indexes = []
        indexmin = []
        indexmin.append(0)
        indexes.append(0)

        # Calculate limit values of FO file indexes
        j = 0
        sum = 0
        while j < rasterNum+1:
            sum = sum+indexmax[j]+1
            indexmin.append(sum)
            indexes.append(sum)
            j = j+1

        # Write merging expressions of FM files
        # The calculation is done with indices because the number of minimum and 
        # maximum values of rasters and parameters varies.
        i = 0
        while i < rasterNum:
            while indexes[i] < indexes[i+1]:
                j = 0
                while j < indexmin[i+1]:
                    fo_msg = files[j]
                    fo_csv = files[j]
                    fo_ovr = [files[j]]
                    k = 1
                    while k < rasterNum+1:
                        fo_msg = fo_msg + " + " + files[indexes[k]]
                        fo_csv = fo_csv + ";" + files[indexes[k]]
                        fo_ovr.append(files[indexes[k]])
                        k = k + 1
                    num=num+1
                    arcpy.AddMessage ("FO_" + str(num) + " = " + fo_msg)
                    csvfile.write ("FO_" + str(num) + ";" + fo_csv + "\n")
                        
                    overlays = [fo_ovr]
                    if (overlayParams[0] == "Gamma"):
                        outFzyOverlay = FuzzyOverlay(fo_ovr, "Gamma", overlayParams[1])
                    else:
                        outFzyOverlay = FuzzyOverlay(fo_ovr, overlayParams[0])
                    outFzyOverlay.save("FO_" + str(num))
                    result = arcpy.ROCTool_ArcSDM(truepositives, "", "FO_" + str(num), output_folder)

                    j = j+1
                indexes[i+1] = indexes[i+1]+1
                i1 = i+1
                while i1 < rasterNum and indexes[i1] >= indexmin[i1+1]:
                    indexes[i1] = indexmin[i1]
                    indexes[i1+1] = indexes[i1+1]+1
                    i1 = i1+1
                if i1+1 >= len(indexes) or indexes[i1+1] > indexmin[len(indexmin)-1] or required_files == (num+1):
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
            raise arcpy.ExecuteError

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
        if csvfile != "?": csvfile.close()
        arcpy.AddError(arcpy.GetMessages(2))
        arcpy.AddError('Aborting FuzzyROC2 (1)')
    except:
        arcpy.AddMessage("*"*30)
        tb = sys.exc_info()[2]
        tbinfo = traceback.format_tb(tb)[0]
        pymsg = "PYTHON ERRORS:\nTraceback info:\n" + tbinfo + "\nError Info:\n" + str(sys.exc_info()[1])
        msgs = "ArcPy ERRORS:\n" + arcpy.GetMessages(2) + "\n"
        if csvfile != "?": csvfile.close()
        arcpy.AddError(pymsg)
        arcpy.AddError(msgs)
        arcpy.AddError('Aborting FuzzyROC2 (2)')
