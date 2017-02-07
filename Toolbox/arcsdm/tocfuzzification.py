# -*- coding: utf-8 -*-
import sys, string, os, math, traceback, math
import arcpy
from arcpy.sa import Float, Divide, Minus, Reclassify
from arcsdm.common import addToDisplay

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
    except:
        tb = sys.exc_info()[2]
        errors = traceback.format_exc()
        arcpy.AddError(errors)         
