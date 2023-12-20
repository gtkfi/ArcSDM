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
        output_features = parameters[5].valueAsText

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

            if mask_type in ["FeatureLayer", "FeatureClass"]:
                arcpy.management.SelectLayerByLocation(input_features, "COMPLETELY_WITHIN", mask)
            elif mask_type in ["RasterLayer", "RasterDataset"]:
                # We need to convert the raster to a feature, since SelectLayerByLocation requires features
                tmp_mask = str(mask) + "_tmp_feature"

                # If the conversion seems slow with more complicated rasters, set max_vertices_per_feature
                arcpy.conversion.RasterToPolygon(mask, tmp_mask)
                arcpy.management.SelectLayerByLocation(input_features, "COMPLETELY_WITHIN", tmp_mask)
                # Delete the temporary layer
                arcpy.management.Delete(tmp_mask)
            else:
                raise arcpy.ExecuteError(f"Mask has forbidden data type: {mask_type}!")

        else:
            arcpy.management.SelectLayerByAttribute(input_features)

        identifier = get_identifier(input_features)

        training_sites = dict()

        with arcpy.da.SearchCursor(input_features, identifier) as features:
            for feature in features:
                training_sites[random.random()] = feature[0]

        training_sites = dict(sorted(training_sites.items()))

        # Clear selection
        arcpy.management.SelectLayerByAttribute(input_features, "CLEAR_SELECTION")

        if is_random_selection_selected:
            partition = percentage / 100.0
            cutoff = int(partition * len(training_sites)) # Rounded down to ensure within index
            
            clause = f"{identifier} = "

            selected_count = 0
            for key in training_sites:
                if selected_count == cutoff:
                    break

                if selected_count == 0:
                    clause += str(training_sites[key])
                else:
                    clause += f" or {identifier} = {training_sites[key]}"

                selected_count += 1

            arcpy.management.SelectLayerByAttribute(input_features, "ADD_TO_SELECTION", clause)
        
        if output_features:
                arcpy.management.CopyFeatures(input_features, output_features)

        # Finally, clear selection
        arcpy.management.SelectLayerByAttribute(input_features, "CLEAR_SELECTION")

        arcpy.SetParameterAsText(5, output_features)

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


def get_identifier(input_features):
    field_names = [f.name for f in arcpy.ListFields(input_features)]

    if "OBJECTID" in field_names:
        return "OBJECTID"
    
    if "FID" in field_names:
        return "FID"
    
    raise arcpy.ExecuteError("Check input feature attributes! The training points have no OBJECTID or FID.")
