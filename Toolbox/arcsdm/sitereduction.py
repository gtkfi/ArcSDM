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

        # Select the features, applying mask if it is set
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

        # Go through the selected features and collect attributes that are used later into training sites
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
                arcpy.AddMessage(f"Minimum distance (m): {min_dist_m}")

                for feature in features:
                    point = Point(feature[1])
                    if check_if_point_is_far_enough(thinned_points, min_dist_m, point):
                        thinned_points.append(point)
                        clause += f" or {identifier} = {feature[0]}"
                        if is_random_selection_selected:
                            training_sites[random.random()] = feature[0]
                
                if not is_random_selection_selected:
                    # Create the selection
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


class Point:
    def __init__(self, point):
        self.x = point[0]
        self.y = point[1]

    def __eq__(self, otherpnt):
        return self.x == otherpnt.x and self.y == otherpnt.y
    
    # the __cmp__() special method is no longer honored in Python 3
    # TODO: remove/replace
    def __cmp__(self, otherpnt):
        return 0 if (self.x == otherpnt.x and self.y == otherpnt.y) else 1
        # if self.x == otherpnt.x and self.y == otherpnt.y:
        #     return 0
        # else:
        #     return 1
    
    def __repr__(self):
        return f"{self.x}, {self.y}"


def rowgen(searchcursor_rows):
    """ Convert gp searchcursor to a generator function """
    rows = searchcursor_rows
    row = rows.next()        
    while row:
        yield row
        row = rows.next()


def distance(pnt1, pnt0):
    return math.hypot(pnt1.x - pnt0.x, pnt1.y - pnt0.y)


def check_if_point_is_far_enough(points, limit, new_point):
    """
        1. Add first point to saved list.
        2. Check if next point is within Unit radius of saved points.
        3. If not, add it to saved list.
        4. Go to 2.
        
        The number tried is n/2 on average, because saved list grows from 1.
        nTrials = Sigma(x,x=(1,n)) = n/2 + n*n/4 or O(n*n)
        nTrials(10) = 5 + 25 = 30
        nTrials(100) = 50 + 2500 = 2550
        nTrials(1000) = 500 + 250,000 = 250,500
        nTrials(10,000) = 5000 + 25,000,000 = 25,050,000
    """
    for point in points:
        d = distance(point, new_point)
        if d < limit:
            return False
    return True
