# -*- coding: utf-8 -*-
import sys, string, os, math, traceback, math
import arcpy
from arcpy.sa import Reclassify, Float, Divide
from arcsdm.common import addToDisplay

def Calculate(self, parameters, messages):
    try:
        messages.addMessage("Starting categorical reclass calculation")
        cat_evidence = parameters[0].valueAsText
        reclass_field = parameters[1].valueAsText
        reclassification = parameters[2].value
        fm_categorical = parameters[3].valueAsText
        divisor  = parameters[4].value
        reclassified = Reclassify(cat_evidence, reclass_field, reclassification, "DATA")
        float_reclassified = Float(reclassified)
        result_raster = Divide(float_reclassified, divisor)
        arcpy.MakeRasterLayer_management(result_raster, fm_categorical)
        arcpy.SetParameterAsText(3, fm_categorical)
        rasterLayerName = os.path.split(fm_categorical)[1]
        addToDisplay(result_raster, rasterLayerName, "BOTTOM")
    except:
        tb = sys.exc_info()[2]
        errors = traceback.format_exc()
        arcpy.AddError(errors)


