# -*- coding: utf-8 -*-
"""

    ArcSDM 5 - ArcSDM for ArcGis pro

    History:
    29.4.2020 fixed reclassification table field names / Arto Laiho, GTK/GSF
"""
import sys, string, os, math, traceback, math
import arcpy
from arcpy.sa import ReclassByTable, Float, Divide
from arcsdm.common import addToDisplay

def Calculate(self, parameters, messages):
    try:
        messages.addMessage("Starting categorical membership calculation")
        cat_evidence = parameters[0].valueAsText
        reclassification = parameters[1].valueAsText
        rescale_constant = parameters[2].value
        fmcat = parameters[3].valueAsText
        if fmcat == '#' or not fmcat:
            fmcat = "%Workspace%/FMCat" # provide a default value if unspecified
        #Next row doesn't work with Reclassification table from Categorical & Reclass / Arto Laiho, Geological survey of Finland
        #reclass_cat_evidence = ReclassByTable(cat_evidence, reclassification, "VALUE", "VALUE", "FMx100", "NODATA")
        product = arcpy.GetInstallInfo()['ProductName']
        if "Desktop" in product:
            reclass_cat_evidence = ReclassByTable(cat_evidence, reclassification, "FROM", "TO", "OUT", "NODATA") #AL 290420
        else:
            reclass_cat_evidence = ReclassByTable(cat_evidence, reclassification, "FROM_", "TO", "OUT", "NODATA") #AL 290420
        float_reclass_cat_evidence = Float(reclass_cat_evidence)
        result_raster = Divide(float_reclass_cat_evidence, rescale_constant)
        rasterLayerName = os.path.split(fmcat)[1]
        addToDisplay(result_raster, rasterLayerName, "BOTTOM")
    except:
        tb = sys.exc_info()[2]
        errors = traceback.format_exc()
        arcpy.AddError(errors)       
