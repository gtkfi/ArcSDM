"""
    ArcSDM 6 ToolBox for ArcGIS Pro

    Conversion and tool development for ArcGIS Pro by Geological Survey of Finland (GTK), 2024.

    Training sites reduction tool

    History:

    Previous version by Unknown (ArcSDM)
    7.9.2016 Fixes
    13.4.2016 Recoded for ArcSDM 5 / ArcGis pro
    30.6.2016 As python toolbox tool module
    8.8.2016 AG Desktop compatibility TR
    13.7.2017 Check for non raster masks TR
    29.12.2023 Convert to work on ArcGIS Pro 3.x, clarify code and implement fixes and improvements:
    support more mask types and don't remove mask layer from map, clear selection after tool is run
"""

import sys, traceback
import arcpy
import math
import random

from arcsdm.common import select_features_by_mask

def ReduceSites(self, parameters, messages):
    arcpy.AddMessage("Starting sites reduction")
    arcpy.AddMessage("------------------------------")

    try:
        input_features = parameters[0].valueAsText
        output_features = parameters[5].valueAsText

        is_thinning_selection_selected = parameters[1].value if parameters[1].value is not None else False
        unit_area_sq_km = parameters[2].value if parameters[2].value is not None else 0.0

        is_random_selection_selected = parameters[3].value if parameters[3].value is not None else False
        percentage = parameters[4].value if parameters[4].value is not None else 100

        arcpy.AddMessage(f"Training points: {input_features}")

        # Select the features, applying mask if it is set
        select_features_by_mask(input_features)

        identifier = get_identifier(input_features)

        training_sites = dict()

        # Go through the selected features: apply thinning, if selected, and collect
        # attributes that are used for random selection into training_sites
        with arcpy.da.SearchCursor(input_features, [identifier, "SHAPE@XY"]) as features:
            if is_thinning_selection_selected:
                thinned_points = []
                first_feature = next(features)
                thinned_points.append(Point(first_feature[1]))

                clause = f"{identifier} = {first_feature[0]}"

                if is_random_selection_selected:
                    # A random number is used as key to allow making a random selection
                    training_sites[random.random()] = first_feature[0]

                min_dist_m = math.sqrt(unit_area_sq_km * 1000000.0 / math.pi)
                arcpy.AddMessage(f"Minimum distance between neighboring training sites (m): {min_dist_m}")

                for feature in features:
                    point = Point(feature[1])
                    if check_distance_to_other_points_is_above_limit(thinned_points, point, min_dist_m):
                        thinned_points.append(point)
                        clause += f" or {identifier} = {feature[0]}"
                        if is_random_selection_selected:
                            training_sites[random.random()] = feature[0]
                
                if not is_random_selection_selected:
                    # Create the final selection that will be copied
                    arcpy.management.SelectLayerByAttribute(input_features, "CLEAR_SELECTION")
                    arcpy.management.SelectLayerByAttribute(input_features, "ADD_TO_SELECTION", clause)
            else:
                for feature in features:
                    # A random number is used as key to allow making a random selection
                    training_sites[random.random()] = feature[0]

        if is_random_selection_selected:
            # Clear selection
            arcpy.management.SelectLayerByAttribute(input_features, "CLEAR_SELECTION")

            training_sites = dict(sorted(training_sites.items()))            
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

            # Create the final selection that will be copied
            arcpy.management.SelectLayerByAttribute(input_features, "ADD_TO_SELECTION", clause)
        
        # Copy the selected features to a new layer
        if output_features:
            arcpy.management.CopyFeatures(input_features, output_features)

        # Finally, clear selection
        arcpy.management.SelectLayerByAttribute(input_features, "CLEAR_SELECTION")

        arcpy.SetParameterAsText(5, output_features)

    except arcpy.ExecuteError:
        e = sys.exc_info()[1]
        arcpy.AddError(f"Training sites reduction caught arcpy.ExecuteError: {e.args[0]}")

    except:
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


class Point:
    def __init__(self, point):
        self.x = point[0]
        self.y = point[1]

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __repr__(self):
        return f"{self.x}, {self.y}"


def distance(point_1, point_0):
    return math.hypot(point_1.x - point_0.x, point_1.y - point_0.y)


def check_distance_to_other_points_is_above_limit(points, new_point, limit):
    for point in points:
        d = distance(point, new_point)
        if d < limit:
            return False
    return True
