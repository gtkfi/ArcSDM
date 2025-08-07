import sys
import traceback
import arcpy
import math
from Toolbox.utils.site_reduction_functions import (
    create_output_layer,
    get_area_unit,
    get_identifier,
    convert_area_to_sq_m,
    convert_distance_to_layer_units,
    geodesic_distance,
)

def ThinSites(self, parameters, messages):
    """
    Thinning tool. Selects a subset of points such that no two points are closer than a minimum distance,
    calculated from a user-specified unit area (with unit selection) or directly as a minimum distance.
    """
    try:
        # Parameters
        input_features = parameters[0].valueAsText
        unit_area = parameters[1].value
        area_unit = parameters[2].valueAsText
        min_distance = parameters[3].value
        output_layer = parameters[4].valueAsText

        # Validation
        if input_features is None or not arcpy.Exists(input_features):
            arcpy.AddError("Input training sites layer is required.")
            return
        if (unit_area is None or unit_area <= 0) and (min_distance is None or min_distance <= 0):
            arcpy.AddError("Either a positive unit area or minimum distance must be provided.")
            return

        converted_area_unit = get_area_unit(area_unit)

        # Get spatial reference and units
        desc = arcpy.Describe(input_features)
        spatial_ref = desc.spatialReference
        linear_unit = spatial_ref.linearUnitName if hasattr(spatial_ref, "linearUnitName") else "Unknown"
        arcpy.AddMessage(f"Input layer linear unit: {linear_unit}")

        # Calculate minimum distance if not provided
        if min_distance is None or min_distance <= 0:
            area_sq_m = convert_area_to_sq_m(unit_area, converted_area_unit)
            if area_sq_m is None:
                return
            min_distance = math.sqrt(area_sq_m / math.pi)
            arcpy.AddMessage(f"Calculated minimum distance (meters): {min_distance:.2f}")
        else:
            arcpy.AddMessage(f"User-specified minimum distance: {min_distance}")

        # Convert minimum distance from meters to layer units
        min_dist_layer_units = convert_distance_to_layer_units(min_distance, "Meters", linear_unit)
        if min_dist_layer_units is None:
            return
        arcpy.AddMessage(f"Minimum distance in layer units: {min_dist_layer_units:.2f} {linear_unit}")

        # Get identifier field
        identifier = get_identifier(input_features)

        # Collect all points
        points = []
        with arcpy.da.SearchCursor(input_features, [identifier, "SHAPE@XY"]) as cursor:
            for row in cursor:
                points.append((row[0], row[1]))

        # Thinning algorithm
        kept_ids = []
        kept_points = []
        for oid, (x, y) in points:
            pt = (x, y)
            if all(geodesic_distance(pt, kp, spatial_ref) >= min_dist_layer_units for kp in kept_points):
                kept_ids.append(oid)
                kept_points.append(pt)

        arcpy.AddMessage(f"Selected {len(kept_ids)} thinned points out of {len(points)} input points.")

        create_output_layer(kept_ids, input_features, output_layer, identifier)

    except arcpy.ExecuteError:
        arcpy.AddError(arcpy.GetMessages(2))
    except Exception as e:
        tb_info = traceback.format_tb(sys.exc_info()[2])[0]
        arcpy.AddError(f"PYTHON ERRORS:\nTraceback Info:\n{tb_info}\nError Info:\n{type(e).__name__}: {e}")
