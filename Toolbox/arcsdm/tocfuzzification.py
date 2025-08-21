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
        fmtoc_path = parameters[4].valueAsText
        
        min_class_number = 1
        reclassified = Reclassify(input_raster, reclass_field, remap, "DATA")
        float_reclass = Float(reclassified)
        minus_float1 = Minus(float_reclass, min_class_number) 
        denominator = Minus(classes, min_class_number)
        fmtoc = Divide(minus_float1, denominator)

        fmtoc.save(fmtoc_path)

    # Return geoprocessing specific errors
    except arcpy.ExecuteError:    
        arcpy.AddError(arcpy.GetMessages(2))    
    except:
        # By default any other errors will be caught here
        e = sys.exc_info()[1]
        print(e.args[0])
        arcpy.AddError(e.args[0])       
