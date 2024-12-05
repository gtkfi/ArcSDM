import arcpy

def check_input_data(evidence_rasters, training_point_feature):
    # Check that all rasters have the same coordinate system

    # TODO: read multiband rasters

    # TODO: check that all rasters have int-like datatype

    if len(evidence_rasters) > 0:
        raster_description = arcpy.Describe(evidence_rasters[0])
        raster_coord_sys = raster_description.spatialReference.name.strip()

        if len(evidence_rasters) > 1:
            i = 1
            while i < len(evidence_rasters):
                coord_sys = arcpy.Describe(evidence_rasters[i]).spatialReference.name.strip()
                if coord_sys != raster_coord_sys:
                    raise ValueError(f"Not all evidence rasters share the same coordinate system! Raster {evidence_rasters[i]} has spatial reference {coord_sys}, when expected spatial reference was {raster_coord_sys}.")
                i += 1
            
            # TODO: check that there isn't more than one nodata value used among the evidence rasters

        feature_description = arcpy.Describe(training_point_feature)
        feature_coord_sys = feature_description.spatialReference.name.strip()

        if feature_coord_sys != raster_coord_sys:
            raise ValueError(f"Coordinate system of training feature and evidence raster(s) do not match! Training data is in {feature_coord_sys} and evidence data is in {raster_coord_sys}.")


# Note: same purpose as ExtractValuesToPoints has in workarounds_93
# But that one uses shapefile instead of gdb feature class
def get_evidence_values_at_training_points(evidence_raster, training_point_feature):
    raise ValueError("Testing!")
    
    output_tmp_feature = arcpy.CreateScratchName("Extr", "Tmp", "FeatureClass", arcpy.env.scratchWorkspace)
    arcpy.sa.ExtractValuesToPoints(training_point_feature, evidence_raster, output_tmp_feature)

    return output_tmp_feature
