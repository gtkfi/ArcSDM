
import arcpy
import importlib
import os
import sys

import arcsdm.wofe_common

from arcsdm.wofe_common import check_input_data, log_wofe


def Execute(self, parameters, messages):
    # TODO: Remove this after testing is done!
    # Make sure imported modules are refreshed if the toolbox is refreshed.
    importlib.reload(arcsdm.wofe_common)
    try:
        arcpy.env.overwriteOutput = True
        arcpy.AddMessage("Setting overwriteOutput to True")

        arcpy.SetLogHistory(True)
        arcpy.AddMessage("Setting LogHistory to True")

        evidence_rasters = parameters[0].valueAsText
        weights_tables = parameters[1].valueAsText
        training_point_feature = parameters[2].valueAsText
        is_ignore_missing_data_selected = parameters[3].value
        nodata_value = parameters[4].value
        unit_cell_area_sq_km = parameters[5].value
        output_pprb_raster = parameters[6].valueAsText
        output_std_raster = parameters[7].valueAsText
        output_md_variance = parameters[8].valueAsText
        output_total_stddev = parameters[9].valueAsText
        output_confidence_raster = parameters[10].valueAsText

        # TODO: handle unique case somehow - is not generalizes, so cannot be used as input

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

        total_area_sq_km_from_mask, training_point_count = log_wofe(unit_cell_area_sq_km, training_point_feature)
        area_cell_count = total_area_sq_km_from_mask / unit_cell_area_sq_km
        prior_probability = float(training_point_count) / area_cell_count

        arcpy.AddMessage("%-20s %s"% ("Prior probability:" , str(prior_probability)))
        arcpy.AddMessage(f"Input evidence rasters: {evidence_rasters}")

        workspace_type = arcpy.Describe(arcpy.env.workspace).workspaceType

        # TODO: check that the order of the evidence rasters and the associated weights tables matches, e.g. by checking the number of classes

        i = 0

        while i < len(evidence_rasters):
            input_raster = evidence_rasters[i]
            weights_table = weights_tables[i]

            arcpy.AddMessage(f"Processing evidence layer {input_raster} and weights table {weights_table}")

            if workspace_type == "FileSystem":
                if not weights_table.endswith(".dbf"):
                    weights_table += ".dbf"
            else:
                wtsbase = os.path.basename(weights_table)
                while len(wtsbase) > 0 and (wtsbase[:1] <= "9" or wtsbase[:1] == "_"):
                    wtsbase = wtsbase[1:]
                weights_table = os.path.dirname(weights_table) + "\\" + wtsbase
            


            # TODO: make sure this results in unique names
            output_raster_name = input_raster.replace(".", "_")
            output_raster_name = output_raster_name[:10] + "W"
            if workspace_type != "FileSystem":
                while len(output_raster_name) > 0 and (output_raster_name[:1] <= "9" or output_raster_name[:1] == "_"):
                    output_raster_name = output_raster_name[1:]
            
            tmp_W_raster = arcpy.CreateScratchName("", "", "RasterDataset", arcpy.env.scratchWorkspace)
            # arcpy.management.CopyRaster()

            
            # Note: the step in the old code where the mask gets used is the Lookup function
            # TODO: go properly through the old code to see if mask should be applied earlier
            # (it might affect the logic itself, and possibly the performance)

            i += 1

        
    except arcpy.ExecuteError:
        arcpy.AddError(arcpy.GetMessages(2))
    except Exception:
        e = sys.exc_info()[1]
        print(e.args[0])
        arcpy.AddError(e.args[0])