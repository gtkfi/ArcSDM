# -*- coding: utf-8 -*-
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
        reclass_cat_evidence = ReclassByTable(cat_evidence, reclassification, "VALUE", "VALUE", "FMx100", "NODATA")
        float_reclass_cat_evidence = Float(reclass_cat_evidence)
        result_raster = Divide(float_reclass_cat_evidence, rescale_constant)
        rasterLayerName = os.path.split(fmcat)[1]
        addToDisplay(result_raster, rasterLayerName, "BOTTOM")
    except:
        tb = sys.exc_info()[2]
        errors = traceback.format_exc()
        arcpy.AddError(errors)       
