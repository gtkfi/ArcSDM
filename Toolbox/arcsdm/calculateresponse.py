
import arcpy
import importlib
import math
import os
import sys
import traceback

from arcsdm.common import log_arcsdm_details
from arcsdm.wofe_common import check_wofe_inputs, get_study_area_parameters, WofeInputError


def Execute(self, parameters, messages):
    try:
        arcpy.CheckOutExtension("Spatial")

        arcpy.env.overwriteOutput = True
        arcpy.AddMessage("Setting overwriteOutput to True")

        arcpy.SetLogHistory(True)
        arcpy.AddMessage("Setting LogHistory to True")

        input_evidence_rasters = parameters[0].valueAsText
        input_weights_tables = parameters[1].valueAsText
        training_points_feature = parameters[2].valueAsText
        is_ignore_missing_data_selected = parameters[3].value
        missing_data_value = parameters[4].value
        unit_cell_area_sq_km = parameters[5].value
        output_pprb_raster = parameters[6].valueAsText
        output_std_raster = parameters[7].valueAsText
        output_mdvar_raster = parameters[8].valueAsText
        output_total_std_raster = parameters[9].valueAsText
        output_confidence_raster = parameters[10].valueAsText

        nodata_value = missing_data_value
        if is_ignore_missing_data_selected:
            nodata_value = "#"

        input_rasters = input_evidence_rasters.split(";")
        weights_tables = input_weights_tables.split(";")

        if len(input_rasters) != len(weights_tables):
            raise WofeInputError("The number of evidence rasters should equal the number of weights tables!")

        # TODO: Add check for weights table columns depending on weights type (unique weights shouldn't have generalized columns)

        check_wofe_inputs(input_rasters, training_points_feature)

        for weights_table in weights_tables:
            fields = arcpy.ListFields(weights_table)
            fields = [str(f.baseName).lower() for f in fields]
            if ("weight" not in fields) or ("w_std" not in fields):
                raise WofeInputError(f"The weights table {weights_table} has not been generalized! Make sure each table has the columns 'WEIGHT' and 'W_STD'.")

        # TODO: check that all evidence rasters have the same cell size?
        # TODO: use the env Cell Size instead. not all of the evidence rasters necessarily have the same cell size
        # TODO: evidence rasters should be resampled to env Cell Size?
        evidence_cellsize = arcpy.Describe(input_rasters[0]).MeanCellWidth

        log_arcsdm_details()
        total_area_sq_km_from_mask, training_point_count = get_study_area_parameters(unit_cell_area_sq_km, training_points_feature)
        area_cell_count = total_area_sq_km_from_mask / unit_cell_area_sq_km
        prior_probability = training_point_count / area_cell_count

        arcpy.AddMessage("\n" + "=" * 21 + " Starting calculate response " + "=" * 21)
        arcpy.AddMessage("%-20s %s"% ("Prior probability:" , str(prior_probability)))
        arcpy.AddMessage(f"Input evidence rasters: {input_rasters}")

        workspace_type = arcpy.Describe(arcpy.env.workspace).workspaceType

        # TODO: check that the order of the evidence rasters and the associated weights tables matches, e.g. by checking the number of classes

        tmp_weights_rasters = []
        tmp_std_rasters = []
        tmp_rasters_with_missing_data = []

        i = 0

        # Create Weight and Standard Deviation rasters
        while i < len(input_rasters):
            input_raster = input_rasters[i]
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
            
            # TODO: check if shortening the name is necessary
            # TODO: make sure these result in unique names if evidence rasters have similar names
            w_raster_name = input_raster.replace(".", "_")
            w_raster_name = w_raster_name + "_W"

            std_raster_name = input_raster.replace(".", "_")
            std_raster_name = std_raster_name + "_S"

            if workspace_type != "FileSystem":
                while len(w_raster_name) > 0 and (w_raster_name[:1] <= "9" or w_raster_name[:1] == "_"):
                    w_raster_name = w_raster_name[1:]
                while len(std_raster_name) > 0 and (std_raster_name[:1] <= "9" or std_raster_name[:1] == "_"):
                    std_raster_name = std_raster_name[1:]

            # Check for unsigned integer raster - cannot have negative missing data
            if (nodata_value != "#") and (arcpy.Describe(input_raster).pixelType.upper().startswith("U")):
                nodata_value = "#"
            
            # Get input raster as an in-memory raster layer, since AddJoin requires a layer
            tmp_raster_layer = "tmp_rst"
            arcpy.management.MakeRasterLayer(input_raster, tmp_raster_layer)
            arcpy.management.AddJoin(tmp_raster_layer, "VALUE", weights_table, "CLASS")

            output_tmp_w_raster = arcpy.CreateScratchName(w_raster_name, "", "RasterDataset", arcpy.env.scratchWorkspace)
            output_tmp_std_raster = arcpy.CreateScratchName(std_raster_name, "", "RasterDataset", arcpy.env.scratchWorkspace)

            arcpy.management.CopyRaster(tmp_raster_layer, output_tmp_w_raster, "#", "#", nodata_value)
            arcpy.management.CopyRaster(tmp_raster_layer, output_tmp_std_raster, "#", "#", nodata_value)

            weight_lookup = arcpy.sa.Lookup(output_tmp_w_raster, "WEIGHT")
            std_lookup = arcpy.sa.Lookup(output_tmp_std_raster, "W_STD")

            # Note: the step in the old code where the mask gets used is the Lookup function
            # TODO: go properly through the old code to see if mask should be applied earlier
            # (it might affect the logic itself, and possibly the performance)
            weight_lookup.save(output_tmp_w_raster)
            std_lookup.save(output_tmp_std_raster)

            tmp_weights_rasters.append(output_tmp_w_raster)
            tmp_std_rasters.append(output_tmp_std_raster)

            if not is_ignore_missing_data_selected:
                clause = "Class = %s" % missing_data_value
                with arcpy.da.SearchCursor(weights_table, ["Class"], where_clause=clause) as cursor:
                    for row in cursor:
                        if row:
                            tmp_rasters_with_missing_data.append(output_tmp_w_raster)
                            break

            i += 1

        arcpy.AddMessage("Finished creating tmp rasters")

        arcpy.AddMessage("\Creating Post Probability Raster...\n" + "=" * 41)

        variable_names = [f'"e{i}"' for i in range(len(tmp_weights_rasters))]
        weights_sum_expression = " + ".join(i for i in variable_names)
        prior_logit = math.log(prior_probability / (1.0 - prior_probability))

        if len(tmp_weights_rasters) == 1:
            posterior_logit_expression = "%s + %s" % (prior_logit, weights_sum_expression)
        else:
            posterior_logit_expression = "%s + (%s)" % (prior_logit, weights_sum_expression)
        
        arcpy.AddMessage(f"Variables: {variable_names}")
        arcpy.AddMessage("Posterior logit expression: " + posterior_logit_expression)

        posterior_probability_expression = "Float(Exp(%s) / (1.0 + Exp(%s)))" % (posterior_logit_expression, posterior_logit_expression)
        arcpy.AddMessage(f"Posterior probability expression: {posterior_probability_expression}")

        # NOTE: Conflicting info about the use of RasterCalculator in Esri's documentation
        # According to a how-to article on raster calculation, RasterCalculator isn't intended for use in scripting environments.
        # (See: https://support.esri.com/en-us/knowledge-base/how-to-perform-raster-calculation-using-arcpy-000022418)
        # But the arcpy RasterCalculator page does not mention anything about this.
        # (See: https://pro.arcgis.com/en/pro-app/latest/arcpy/spatial-analyst/raster-calculator.htm)
        posterior_probability_result = arcpy.sa.RasterCalculator(tmp_weights_rasters, variable_names, posterior_probability_expression, extent_type="UnionOf", cellsize_type="MinOf")
        posterior_probability_result.save(output_pprb_raster)

        # Due to bit depth issue, the resulting raster has pixel depth 64 Bit
        # See: https://community.esri.com/t5/arcgis-spatial-analyst-questions/controling-bit-depth-of-raster-in-fgdb-format/td-p/125850

        arcpy.AddMessage("\nCreating Post Probability STD Raster...\n" + "=" * 41)
        
        if len(tmp_std_rasters) == 1:
            # TODO: Just save the Std raster as is
            pass
        else:
            std_input_rasters = [arcpy.Describe(output_pprb_raster).catalogPath] + [arcpy.Describe(s).catalogPath for s in tmp_std_rasters]

            std_variable_names = [f'"e{i}"' for i in range(len(tmp_std_rasters))]
            variable_names = ['"pprob"'] + std_variable_names
            std_sum_expression = " + ".join(f'SQR({i})' for i in std_variable_names)
            constant = 1.0 / training_point_count
            std_expression = 'SQRT(SQR("pprob") * (%s + SUM(%s)))' % (constant, std_sum_expression)
            
            arcpy.AddMessage(f"Posterior probability STD expression: {std_expression}")
            arcpy.AddMessage(f"Input rasters: {std_input_rasters}")
            arcpy.AddMessage(f"Variable names: {variable_names}")

            # TODO: Continue
            # Creating the Std raster isn't working at the moment.
            # Possibly the issue is with the pixel depth and/or cell alignment of the posterior probability raster
            # Complains about missing VAT, but that is unlikely to be the real issue.
            std_result = arcpy.sa.RasterCalculator(std_input_rasters, variable_names, std_expression, extent_type="UnionOf", cellsize_type="MinOf")
            std_result.save(output_std_raster)

        arcpy.AddMessage("Deleting tmp rasters...")
        for raster in tmp_weights_rasters:
            arcpy.management.Delete(raster)
        
        for raster in tmp_std_rasters:
            arcpy.management.Delete(raster)
        
    except arcpy.ExecuteError:
        arcpy.AddError(arcpy.GetMessages(2))
    except Exception:
        tb = sys.exc_info()[2]
        tbinfo = traceback.format_tb(tb)[0]

        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
            str(sys.exc_info()) + "\n"
        msgs = "GP ERRORS:\n" + arcpy.GetMessages(2) + "\n"

        arcpy.AddError(msgs)
        arcpy.AddError(pymsg)