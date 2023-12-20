# ArcSDM 5 (Arcgis pro)
#Training site reduction  -tool
# 
#
# History:
# Previous version by Unknown (ArcSDM)
# 7.9.2016 Fixes
# 13.4.2016 Recoded for ArcSDM 5 / ArcGis pro
# 30.6.2016 As python toolbox tool module
# 8.8.2016 AG Desktop compatibility TR
# 13.7.2017 Check for non raster masks TR

import sys, string, os, math, traceback
import arcgisscripting
import arcpy
import math
import random

def ReduceSites(self, parameters, messages):
    messages.addMessage("Starting sites reduction")
    messages.addMessage("------------------------------")

    try:
        input_features = parameters[0].valueAsText
        output_feature = parameters[5].valueAsText

        is_thinning_selection_selected = parameters[1].value if parameters[1].value is not None else False
        unit_area_sq_km = parameters[2].value if parameters[2].value is not None else 0.0

        is_random_selection_selected = parameters[3].value if parameters[3].value is not None else False
        percentage = parameters[4].value if parameters[4].value is not None else 100

        messages.AddMessage(f"Training points: {input_features}")

        mask = arcpy.env.mask
        if mask:
            if not arcpy.Exists(mask):
                raise arcpy.ExecuteError("Mask doesn't exist! Set Mask under Analysis/Environments.")

            mask_type = arcpy.Describe(mask).dataType
            if mask_type not in ["FeatureLayer", "FeatureClass"]:
                raise arcpy.ExecuteError("Reduce training points tool requires mask of type 'FeatureLayer' or 'FeatureClass'!")

            arcpy.management.SelectLayerByLocation(input_features, "COMPLETELY_WITHIN", mask)
        else:
            arcpy.management.SelectLayerByAttribute(input_features)

        training_sites = dict()

        # TODO: select by fields instead of selecting all
        with arcpy.da.SearchCursor(input_features, "*") as features:
            for feature in features:
                training_sites[random.random()] = feature

        training_sites = dict(sorted(training_sites.items()))

        # Finally, clear selection
        #arcpy.management.SelectLayerByAttribute(input_features, "CLEAR_SELECTION")

    except arcpy.ExecuteError:
        e = sys.exc_info()[1]
        arcpy.AddError(f"Training sites reduction caught arcpy.ExecuteError: {e.args[0]}")

    except Exception:
        e = sys.exc_info()
        tb = e[2]
        # tbinfo contains the line number that the code failed on and the code from that line
        tb_info = traceback.format_tb(tb)[0]
        error_message = f"PYTHON ERRORS:\nTraceback Info:\n{tb_info}\nError Info:\n{e[0]}: {e[1]}\n"
        messages.AddError(error_message)