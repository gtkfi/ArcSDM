import arcpy

def check_input_data(evidence_rasters, training_point_feature):
    # Check that all rasters have the same coordinate system

    # TODO: read multiband rasters


    if len(evidence_rasters) > 0:
        raster_description = arcpy.Describe(evidence_rasters[0])
        raster_coord_sys = raster_description.spatialReference.name.strip()
        
        raster_info = arcpy.Raster(evidence_rasters[0])
        raster_nodata_value = raster_info.noDataValue

        if not raster_info.isInteger:
            raise ValueError(f"The evidence raster {evidence_rasters[0]} does not have integer type!")

        if not raster_info.hasRAT:
            raise ValueError(f"The evidence raster {evidence_rasters[0]} does not have an attribute table. Use 'Build Raster Attribute Table' tool to add it.")

        if len(evidence_rasters) > 1:
            i = 1
            while i < len(evidence_rasters):
                coord_sys = arcpy.Describe(evidence_rasters[i]).spatialReference.name.strip()
                if coord_sys != raster_coord_sys:
                    raise ValueError(f"Not all evidence rasters share the same coordinate system! Raster {evidence_rasters[i]} has spatial reference {coord_sys}, when expected spatial reference was {raster_coord_sys}.")
                
                raster = arcpy.Raster(evidence_rasters[i])
                if not raster.hasRAT:
                    raise ValueError(f"The evidence raster {evidence_rasters[i]} does not have an attribute table. Use 'Build Raster Attribute Table' tool to add it.")

                nodata_value = raster.noDataValue
                if nodata_value != raster_nodata_value:
                    raise ValueError(f"Not all evidence rasters share the same NoData value! Raster {evidence_rasters[i]} has {nodata_value}, when expected NoData value was {raster_nodata_value}.")

                i += 1

        feature_description = arcpy.Describe(training_point_feature)
        feature_coord_sys = feature_description.spatialReference.name.strip()

        if feature_coord_sys != raster_coord_sys:
            raise ValueError(f"Coordinate system of training feature and evidence raster(s) do not match! Training data is in {feature_coord_sys} and evidence data is in {raster_coord_sys}.")

        # TODO: check that the training feature has point geometry


# Note: same purpose as ExtractValuesToPoints has in workarounds_93
# But that one uses shapefile instead of gdb feature class
def get_evidence_values_at_training_points(evidence_raster, training_point_feature):
    output_tmp_feature = arcpy.CreateScratchName("Extr", "Tmp", "FeatureClass", arcpy.env.scratchWorkspace)
    arcpy.sa.ExtractValuesToPoints(training_point_feature, evidence_raster, output_tmp_feature)

    return output_tmp_feature
