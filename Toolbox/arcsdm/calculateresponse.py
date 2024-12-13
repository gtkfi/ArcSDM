
import arcpy
import gc
import importlib
import math
import os
import sys
import traceback

import arcsdm.sdmvalues
import arcsdm.wofe_common
import arcsdm.workarounds_93

from arcsdm.wofe_common import check_input_data


def Execute(self, parameters, messages):
    # TODO: Remove this after testing is done!
    # Make sure imported modules are refreshed if the toolbox is refreshed.
    importlib.reload(arcsdm.wofe_common)
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

        check_input_data(evidence_rasters, training_point_feature)

        # TODO: check that all evidence rasters have the same cell size?
        # TODO: use the env Cell Size instead. not all of the evidence rasters necessarily have the same cell size
        # TODO: evidence rasters should be resampled to env Cell Size?
        evidence_cellsize = arcpy.Describe(evidence_rasters[0]).MeanCellWidth
        
        # TODO: apply mask
        study_area_sq_km = 0.0
        study_area_cell_count = 0.0


        
    except arcpy.ExecuteError:
        arcpy.AddError(arcpy.GetMessages(2))
    except Exception:
        e = sys.exc_info()[1]
        print(e.args[0])
        arcpy.AddError(e.args[0])