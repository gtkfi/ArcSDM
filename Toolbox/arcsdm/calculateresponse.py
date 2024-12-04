
import arcpy
import gc
import importlib
import math
import os
import sys
import traceback

import arcsdm.sdmvalues
import arcsdm.workarounds_93


def Execute(self, parameters, messages):
    try:
        evidence_rasters = parameters[0].valueAsText
        weights_tables = parameters[1].valueAsText
        training_point_feature = parameters[2].valueAsText
        is_ignore_missing_data_selected = parameters[3].value
        nodata_value = parameters[4].value

        evidence_rasters = evidence_rasters.split(";")
        weights_tables = weights_tables.split(";")

        if len(evidence_rasters) != len(weights_tables):
            raise ValueError("The number of evidence rasters should equal the number of weights tables!")

        # TODO: Add check for weights table columns depending on weights type (unique weights shouldn't have generalized columns)

        return None
    except arcpy.ExecuteError:
        arcpy.AddError(arcpy.GetMessages(2))
    except Exception:
        e = sys.exc_info()[1]
        print(e.args[0])
        arcpy.AddError(e.args[0])