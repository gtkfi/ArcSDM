import arcpy


def get_area_unit(user_selected):
    area_unit_mapping = {
        "Square Kilometers": "SquareKilometers",
        "Square Meters": "SquareMeters",
        "Square Miles": "SquareMilesInt", 
        "Square Yards": "SquareYardsInt",
        "Square Feet": "SquareFeetInt",
        "Acres": "AcresInt",
        "Hectares": "Hectares"
    }

    arcpy_unit = area_unit_mapping.get(user_selected)

    return arcpy_unit


def create_output_layer(features, input_features, output_layer, identifier):
    """Create an output layer from the selected features."""
    arcpy.AddMessage(f"Creating output layer: {output_layer}")
    temp_layer = "temp_layer"
    arcpy.management.MakeFeatureLayer(input_features, temp_layer)

    # Get field type to handle strings/guids properly
    field_type = [f.type for f in arcpy.ListFields(input_features) if f.name == identifier][0]
    delim = arcpy.AddFieldDelimiters(temp_layer, identifier)

    if field_type in ["String", "Guid"]:
        clause = " OR ".join([f"{delim} = '{feature}'" for feature in features])
    else:
        clause = " OR ".join([f"{delim} = {feature}" for feature in features])

    # Select and export
    arcpy.management.SelectLayerByAttribute(temp_layer, "NEW_SELECTION", clause)
    arcpy.management.CopyFeatures(temp_layer, output_layer)
    arcpy.management.SelectLayerByAttribute(temp_layer, "CLEAR_SELECTION")
    arcpy.management.Delete(temp_layer)

    arcpy.AddMessage(f"Output layer created: {output_layer}")


def get_identifier(input_features):
    """Get the identifier field (OBJECTID or FID) from the input features."""
    try:
        for field in arcpy.ListFields(input_features):
            if field.name.upper() in ["OBJECTID", "FID"]:
                return field.name
    except Exception as e:
        arcpy.AddError(f"Check input feature attributes! The training points have no OBJECTID or FID: {e}")


def geodesic_distance(pt1, pt2, spatial_ref):
    """Geodesic distance in meters between two (x, y) points using spatial reference."""
    p1 = arcpy.PointGeometry(arcpy.Point(*pt1), spatial_ref)
    p2 = arcpy.PointGeometry(arcpy.Point(*pt2), spatial_ref)
    return p1.distanceTo(p2)


def convert_area_to_sq_m(area, from_unit):
    """Convert any supported area unit to square meters using ArcPy."""
    try:
        factor = arcpy.ArealUnitConversionFactor(from_unit, "SquareMeters")
    except Exception as e:
        arcpy.AddError(f"Unsupported area unit '{from_unit}': {e}")
        return None
    return area * factor


def convert_distance_to_layer_units(distance, from_unit, to_unit):
    """Convert between supported linear units using ArcPy."""
    try:
        factor = arcpy.LinearUnitConversionFactor(from_unit, to_unit)
    except Exception as e:
        arcpy.AddError(f"Unsupported linear unit conversion from '{from_unit}' to '{to_unit}': {e}")
        return None
    return distance * factor
