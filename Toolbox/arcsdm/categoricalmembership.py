# -*- coding: utf-8 -*-
import sys, string, os, math, traceback, math
import arcpy
from arcpy.sa import ReclassByTable, Float, Divide

def Calculate(self, parameters, messages):
    messages.addMessage("Starting categorical membership calculation");
    try:
        cat_evidence = parameters[0].valueAsText
        reclassification = parameters[1].valueAsText
        rescale_constant = parameters[2].value
        fmcat = parameters[3].valueAsText
        if fmcat == '#' or not fmcat:
            fmcat = "%Workspace%/FMCat" # provide a default value if unspecified
        reclass_cat_evidence = ReclassByTable(cat_evidence, reclassification, "VALUE", "VALUE", "FMx100", "NODATA")
        float_reclass_cat_evidence = Float(reclass_cat_evidence)
        result_raster = Divide(float_reclass_cat_evidence, rescale_constant)
        rasterLayerName = os.path.split(fmcat)[1]
        addToDisplay(result_raster, rasterLayerName, "BOTTOM")
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


