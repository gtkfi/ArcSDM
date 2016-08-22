# -*- coding: utf-8 -*-
import sys, string, os, math, traceback, math
import arcpy
from arcpy.sa import Float, Divide, Minus, Reclassify

def Calculate(self, parameters, messages):
    try:
        messages.addMessage("Starting toc fuzzification calculation");
        input_raster     = parameters[0].valueAsText
        reclass_field = parameters[1].valueAsText
        remap = parameters[2].valueAsText
        classes = parameters[3].value        
        fmtoc = parameters[4].valueAsText
        if fmtoc == '#' or not fmtoc:
            fmtoc = "%Workspace%/FMTOC" # provide a default value if unspecified
        rasterLayerName = os.path.split(fmtoc)[1]
        min_class_number = 1
        reclassified = Reclassify(input_raster, reclass_field, remap, "DATA")
        float_reclass = Float(reclassified)
        minus_float1 = Minus(float_reclass, min_class_number) 
        denominator = Minus(classes, min_class_number)
        fmtoc = Divide(minus_float1, denominator)
        addToDisplay(fmtoc, rasterLayerName, "BOTTOM")
    except Exception as Msg:
        # get the traceback object
        tb = sys.exc_info()[2]
        # tbinfo contains the line number that the code failed on and the code from that line
        tbinfo = traceback.format_tb(tb)[0]
        # concatenate information together concerning the error into a message string
        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
            str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
        messages.addErrorMessage(pymsg); #msgs)
        # print messages for use in Python/PythonWin
        print (pymsg)
        raise

def addToDisplay(layer, name, position):
    result = arcpy.MakeRasterLayer_management(layer, name)
    lyr = result.getOutput(0)
    product = arcpy.GetInstallInfo()['ProductName']
    if "Desktop" in product:
        mxd = arcpy.mapping.MapDocument("CURRENT")
        dataframe = arcpy.mapping.ListDataFrames(mxd)[0]
        arcpy.mapping.AddLayer(dataframe, lyr, position)
    elif "Pro" in product:
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        m = aprx.listMaps("Map")[0]
        m.addLayer(lyr, position)
