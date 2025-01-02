import arcpy
import os
import sys

from arcsdm.common import log_arcsdm_details, select_features_by_mask


class WofeInputError(Exception):
    """Exception class for WofE Calculate Response errors."""


def execute(self, parameters, messages):
    try:
        training_site_feature = parameters[0].valueAsText
        unit_cell_area_sq_km = parameters[1].value
        
        log_arcsdm_details()
        get_study_area_parameters(unit_cell_area_sq_km, training_site_feature)
        arcpy.AddMessage("\n" + "=" * 40)
    except arcpy.ExecuteError:
        arcpy.AddError(arcpy.GetMessages(2))
    except Exception:
        e = sys.exc_info()[1]
        arcpy.AddError(e.args[0])


def get_study_area_size_sq_km():
    if not arcpy.env.mask:
        raise WofeInputError("Mask doesn't exist! Set Mask under Analysis/Environments.")

    masked_area_sq_km = get_area_size_sq_km(arcpy.env.mask)

    arcpy.AddMessage(f"Mask area (km^2): {masked_area_sq_km}")

    return masked_area_sq_km


def get_area_size_sq_km(layer):
    desc = arcpy.Describe(layer)
    coord_sys = desc.spatialReference
    spatial_unit = coord_sys.linearUnitName if coord_sys.linearUnitName is not None else ""

    # TODO implement conversion
    if not spatial_unit.lower().strip() == "meter":
        raise WofeInputError(f"Output spatial unit should be meter, but was {spatial_unit}. Check output coordinate system in Environments!")

    area_sq_m = 0.0

    if desc.dataType in ["RasterLayer", "RasterDataset", "RasterBand"]:
        raster_info = arcpy.Raster(desc.catalogPath)
        raster_nodata_value = raster_info.noDataValue
        nodata_value = raster_nodata_value if raster_nodata_value is not None else -9999

        raster_array = arcpy.RasterToNumPyArray(desc.catalogPath, nodata_to_value=nodata_value)

        element_count = (raster_array != nodata_value).sum()
        
        cell_size_sq_m = desc.MeanCellWidth * desc.MeanCellHeight
        area_sq_m = element_count * cell_size_sq_m

    elif desc.dataType in ["FeatureLayer", "FeatureClass", "ShapeFile"]:
        with arcpy.da.SearchCursor(desc.catalogPath, ["SHAPE@AREA"]) as cursor:
            for row in cursor:
                area_sq_m += row[0]
    else:
        raise WofeInputError(f"Incorrect mask data type: {desc.dataType}")
    
    conversion_factor = 0.000001
    area_sq_km = area_sq_m * conversion_factor

    return area_sq_km


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


# Similar to old sdmvalues.appendSDMValues, but with return values,
# and less risk of calculating the study area size wrong
def get_study_area_parameters(unit_cell_area_sq_km, training_points):
    """
    Use the mask from the geoprocessing environment to calculate the total study area in km^2
    and the total number of training points. Log WofE parameters to the geoprocessor history.

    Args:
        unit_cell_area_sq_km:
            Unit cell area in km^2.
        training_points:
            Point geometry feature of training points to use for WofE.
    
    Returns:
        A tuple of:
            Total area of the study area in km^2. 
            Number of training points within the study area.

    """
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
        raise WofeInputError("ERROR: 0 Cells in Area!")

    priorprob = float(str(training_point_count)) / float(unit_cells_count)

    if not (0 < priorprob <= 1.0):
        raise WofeInputError(f"Incorrect no. of training sites or unit cell area. TrainingPointsResult {priorprob}")

    arcpy.AddMessage("%-20s %0.6f" % ("Prior Probability:", priorprob))
    arcpy.AddMessage("%-20s %s" % ("Training Points:", arcpy.Describe(training_points).catalogPath))
    arcpy.AddMessage("%-20s %s" % ("Study Area Raster:", arcpy.Describe(arcpy.env.mask).catalogPath))
    arcpy.AddMessage("%-20s %s" % ("Study Area Area:", str(total_area_sq_km) + " km^2"))
    arcpy.AddMessage("")

    return total_area_sq_km, float(str(training_point_count))


def check_wofe_inputs(evidence_rasters, training_point_feature):
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
            raise WofeInputError(f"The evidence raster {evidence_rasters[0]} does not have integer type!")

        if not raster_info.hasRAT:
            raise WofeInputError(f"The evidence raster {evidence_rasters[0]} does not have an attribute table. Use 'Build Raster Attribute Table' tool to add it.")

        if len(evidence_rasters) > 1:
            i = 1
            while i < len(evidence_rasters):
                coord_sys = arcpy.Describe(evidence_rasters[i]).spatialReference.name.strip()
                if coord_sys != raster_coord_sys:
                    raise WofeInputError(f"Not all evidence rasters share the same coordinate system! Raster {evidence_rasters[i]} has spatial reference {coord_sys}, when expected spatial reference was {raster_coord_sys}.")
                
                raster = arcpy.Raster(evidence_rasters[i])
                if not raster.hasRAT:
                    raise WofeInputError(f"The evidence raster {evidence_rasters[i]} does not have an attribute table. Use 'Build Raster Attribute Table' tool to add it.")

                nodata_value = raster.noDataValue
                if nodata_value != raster_nodata_value:
                    raise WofeInputError(f"Not all evidence rasters share the same NoData value! Raster {evidence_rasters[i]} has {nodata_value}, when expected NoData value was {raster_nodata_value}.")

                i += 1

        feature_description = arcpy.Describe(training_point_feature)
        feature_coord_sys = feature_description.spatialReference.name.strip()

        if feature_coord_sys != raster_coord_sys:
            raise WofeInputError(f"Coordinate system of training feature and evidence raster(s) do not match! Training data is in {feature_coord_sys} and evidence data is in {raster_coord_sys}.")

        if feature_description.shapeType != 'Point':
            raise WofeInputError(f"Training data should have point geometry! Current geometry is {feature_description.shapeType}.")


# Note: same purpose as workarounds_93.ExtractValuesToPoints
# But instead of a shapefile, use a gdb feature class
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
            raise WofeInputError("Mask doesn't exist! Set Mask under Analysis/Environments.")

    mask_descr = arcpy.Describe(mask)
    masked_evidence_raster = arcpy.sa.ExtractByMask(evidence_raster, mask_descr.catalogPath)

    if nodata_value is not None:
        masked_evidence_descr = arcpy.Describe(masked_evidence_raster)
        temp_nodata_mask = arcpy.sa.IsNull(masked_evidence_descr.catalogPath)

        # Set nodata value to nodata areas within the mask
        masked_evidence_raster = arcpy.sa.Con(temp_nodata_mask, nodata_value, evidence_raster, "VALUE = 1")
    
        arcpy.management.Delete(masked_evidence_descr.catalogPath)
    
    return masked_evidence_raster


def extract_layer_from_raster_band(evidence_layer, evidence_descr):
    if evidence_descr.dataType == "RasterBand":
        # Try to change RasterBand to RasterDataset
        evidence1 = os.path.split(evidence_layer)
        evidence2 = os.path.split(evidence1[0])
        if (evidence1[1] == evidence2[1]) or (evidence1[1][:4] == "Band"):
            new_evidence_layer = evidence1[0]
            new_evidence_descr = arcpy.Describe(evidence_layer)
            arcpy.AddMessage(f"Evidence Layer is now {new_evidence_layer} and its data type is {new_evidence_descr.dataType}")
            return new_evidence_layer, new_evidence_descr
        else:
            raise WofeInputError("ERROR: Data Type of Evidence Layer cannot be RasterBand, use Raster Dataset.")
    else:
        return evidence_layer, evidence_descr


# Replaces the function MissingDataVariance from missingdatavar_func.py
def create_missing_data_variance_layer(nodata_value, study_area_size_sq_km, weights_rasters_list, post_probability_raster, md_variance_output_name):
    # TODO: Note! Input rasters should be masked already? - possibly add check?
    # TODO: decide whether the weights rasters need to already be masked in a way that missing data has been combined to nodata?
    
    for raster in weights_rasters_list:
        arcpy.AddMessage(f"Missing data Variance for: {raster}")
        weights_raster = arcpy.Describe(raster).catalogPath

        # Start MD Variance raster
        # Get posterior probability at MD cells
        R1 = os.path.join(arcpy.env.scratchWorkspace, "R1")
        if arcpy.Exists(R1):
            arcpy.management.Delete(R1)

        temp_nodata_mask = arcpy.sa.IsNull(weights_raster)
        pprb_at_nodata_cells = arcpy.sa.Con(temp_nodata_mask, post_probability_raster, 0.0, "VALUE = 1")

        pprb_at_nodata_cells.save(R1)
        
        # Get PostODDs at MD cells
        R2 = os.path.join(arcpy.env.scratchWorkspace, "R2")
        if arcpy.Exists(R2):
            arcpy.management.Delete(R2)

        # Exp = "%s / (1.0 - %s)" % (R1, R1)

        variable_names = ['"r1"']
        post_odds_expression = "%s / (1.0 - %s)" % ('"r1"', '"r1"')
        arcpy.AddMessage(f"R2 = {post_odds_expression}")
        post_odds = arcpy.sa.RasterCalculator([R1], variable_names, post_odds_expression, extent_type="UnionOf", cellsize_type="MinOf")
        post_odds.save(R2)
        arcpy.AddMessage(f"R2 exists: {arcpy.Exists(R2)}")

        # Get Total Variance of MD cells
        # Create total class variances list
        ClsVars = []

        # TODO: for each class, calculate the variance of the posterior probability due to missing data

        # TODO: 1: calculate for present pattern as if missing pattern is known:
        # (p(D|Ej)-p(D))^2 * p(Ej)
        # TODO: 2: calculate for absent pattern as if missing pattern is known:
        # (p(D|nEj)-p(D))^2 * p(nEj)
        # (ie. use the full mask area as the total area!)

        # Need:
        # p(Ej): area of predictor pattern j / total area
        # p(nEj): area of missing data for pattern j / total area
        # p(D): posterior probability calculated for the redion where Ej is missing -> pprb_at_nodata_cells
        # p(D|Ej): posterior probability in the presence of pattern Ej
        # p(D|nEj): posterior probability in the absence of pattern Ej
        # 

    return None
