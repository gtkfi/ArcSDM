import arcpy
import os
import sys
import traceback

from arcsdm.common import log_arcsdm_details, select_features_by_mask


def execute(self, parameters, messages):
    try:
        training_site_feature = parameters[0].valueAsText
        unit_cell_area_sq_km = parameters[1].value
        
        log_arcsdm_details()
        log_wofe(unit_cell_area_sq_km, training_site_feature)
        arcpy.AddMessage("\n" + "=" * 40)
    except arcpy.ExecuteError:
        arcpy.AddError(arcpy.GetMessages(2))
    except Exception:
        e = sys.exc_info()[1]
        arcpy.AddError(e.args[0])


def get_study_area_size_sq_km():
    if not arcpy.env.mask:
        raise ValueError("Mask doesn't exist! Set Mask under Analysis/Environments.")

    desc = arcpy.Describe(arcpy.env.mask)
    coord_sys = desc.spatialReference
    spatial_unit = coord_sys.linearUnitName if coord_sys.linearUnitName is not None else ""

    # TODO implement conversion
    if not spatial_unit.lower().strip() == "meter":
        raise ValueError(f"Output spatial unit should be meter, but was {spatial_unit}. Check output coordinate system in Environments!")

    masked_area_sq_m = 0.0

    if desc.dataType in ["RasterLayer", "RasterDataset", "RasterBand"]:
        raster_info = arcpy.Raster(desc.catalogPath)
        raster_nodata_value = raster_info.noDataValue
        nodata_value = raster_nodata_value if raster_nodata_value is not None else -9999

        raster_array = arcpy.RasterToNumPyArray(desc.catalogPath, nodata_to_value=nodata_value)

        element_count = (raster_array != nodata_value).sum()
        
        cell_size_sq_m = desc.MeanCellWidth * desc.MeanCellHeight
        masked_area_sq_m = element_count * cell_size_sq_m

    elif desc.dataType in ["FeatureLayer", "FeatureClass", "ShapeFile"]:
        geometry_field = desc.shapeFieldName

        with arcpy.da.SearchCursor(desc.catalogPath, ["SHAPE@AREA"]) as cursor:
            for row in cursor:
                masked_area_sq_m += row[0]
    else:
        raise ValueError(f"Incorrect mask data type: {desc.dataType}")
    
    conversion_factor = 0.000001
    masked_area_sq_km = masked_area_sq_m * conversion_factor

    arcpy.AddMessage(f"Mask area (km^2): {masked_area_sq_km}")

    return masked_area_sq_km


def get_study_area_unit_cell_count(unit_cell_size_sq_km):
    masked_area_sq_km = get_study_area_size_sq_km()
    return masked_area_sq_km / unit_cell_size_sq_km


def get_selected_point_count(point_feature):
    # Clear any existing selection
    arcpy.management.SelectLayerByAttribute(point_feature, "CLEAR_SELECTION")

    select_features_by_mask(point_feature)
    point_count = arcpy.management.GetCount(point_feature)

    arcpy.management.SelectLayerByAttribute(point_feature, "CLEAR_SELECTION")

    return point_count


def log_wofe(unit_cell_area_sq_km, training_points):
    """
    Log WofE parameters to Geoprocessor History.
    """
    try:
        arcpy.AddMessage("\n" + "=" * 10 + " WofE parameters " + "=" * 10)

        total_area_sq_km = get_study_area_size_sq_km()
        arcpy.AddMessage("%-20s %s" % ("Mask size (km^2):", str(total_area_sq_km)))
        arcpy.AddMessage("%-20s %s" % ("Unit Cell Size:", unit_cell_area_sq_km))
        
        unit_cell_area_sq_km = float(unit_cell_area_sq_km)
        unit_cells_count = float(total_area_sq_km) / unit_cell_area_sq_km

        training_point_count = get_selected_point_count(training_points)

        arcpy.AddMessage("%-20s %s" % ("# Training Sites:", training_point_count))
        arcpy.AddMessage("%-20s %s" % ("Unit Cell Area:", "{} km^2, Cells in area: {} ".format(unit_cell_area_sq_km, unit_cells_count)))

        if unit_cells_count == 0:
            arcpy.AddError("ERROR: 0 Cells in Area!")

        priorprob = float(str(training_point_count)) / float(unit_cells_count)

        if not (0 < priorprob <= 1.0):
            arcpy.AddError('Incorrect no. of training sites or unit cell area. TrainingPointsResult {}'.format(priorprob))

        arcpy.AddMessage("%-20s %0.6f" % ("Prior Probability:", priorprob))
        arcpy.AddMessage("%-20s %s" % ("Training Points:", arcpy.Describe(training_points).catalogPath))
        arcpy.AddMessage("%-20s %s" % ("Study Area Raster:", arcpy.Describe(arcpy.env.mask).catalogPath))
        arcpy.AddMessage("%-20s %s" % ("Study Area Area:", str(total_area_sq_km) + " km^2"))
        arcpy.AddMessage("")

        return total_area_sq_km, training_point_count
    except arcpy.ExecuteError as e:
        if not all(e.args):
            arcpy.AddMessage("Calculate weights caught arcpy.ExecuteError: ")
            args = e.args[0]
            args.split('\n')
            arcpy.AddError(args)
        arcpy.AddMessage("-------------- END EXECUTION ---------------")
        raise
    except:
        tb = sys.exc_info()[2]
        tbinfo = traceback.format_tb(tb)[0]
        arcpy.AddError(tbinfo)
        if len(arcpy.GetMessages(2)) > 0:
            msgs = "SDM GP ERRORS:\n" + arcpy.GetMessages(2) + "\n"
            arcpy.AddError(msgs)
        raise


def check_input_data(evidence_rasters, training_point_feature):
    """
    Check that the coordinate system of all inputs match.

    Verify that each raster input is of integer type and has an attribute table.

    Check that training feature has point geometry.
    """
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

        if feature_description.shapeType != 'Point':
            raise ValueError(f"Training data should have point geometry! Current geometry is {feature_description.shapeType}.")


# Note: same purpose as ExtractValuesToPoints has in workarounds_93
# But that one uses shapefile instead of gdb feature class
def get_evidence_values_at_training_points(evidence_raster, training_point_feature):
    output_tmp_feature = arcpy.CreateScratchName("Extr", "Tmp", "FeatureClass", arcpy.env.scratchWorkspace)
    arcpy.sa.ExtractValuesToPoints(training_point_feature, evidence_raster, output_tmp_feature)

    return output_tmp_feature


def get_training_point_statistics(evidence_raster, training_point_feature):
    values_at_training_points_tmp_feature = get_evidence_values_at_training_points(evidence_raster, training_point_feature)

    arcpy.management.Delete(os.path.join(arcpy.env.scratchWorkspace, "WtsStatistics"))
    output_tmp_table = arcpy.management.CreateTable(arcpy.env.scratchWorkspace, "WtsStatistics")

    arcpy.analysis.Statistics(values_at_training_points_tmp_feature, output_tmp_table, "rastervalu Sum", "rastervalu")
    # The rastervalu field has type SmallInteger, which is inconvenient
    # Create a new field with type Integer
    arcpy.management.CalculateField(
        in_table=output_tmp_table,
        field="class_category",
        expression="int(!rastervalu!)",
        expression_type="PYTHON3",
        field_type="LONG")

    arcpy.management.Delete(values_at_training_points_tmp_feature)

    return output_tmp_table, "class_category", "frequency"


def apply_mask_to_raster(evidence_raster, nodata_value=None):
    mask = arcpy.env.mask
    if mask:
        if not arcpy.Exists(mask):
            raise ValueError("Mask doesn't exist! Set Mask under Analysis/Environments.")

    mask_descr = arcpy.Describe(mask)
    masked_evidence_raster = arcpy.sa.ExtractByMask(evidence_raster, mask_descr.catalogPath)

    if nodata_value is not None:
        masked_evidence_descr = arcpy.Describe(masked_evidence_raster)
        temp_nodata_mask = arcpy.sa.IsNull(masked_evidence_descr.catalogPath)

        # Set nodata value to nodata areas within the mask
        masked_evidence_raster = arcpy.sa.Con(temp_nodata_mask, nodata_value, evidence_raster, "VALUE = 1")
    
        arcpy.management.Delete(masked_evidence_descr.catalogPath)
    
    return masked_evidence_raster
