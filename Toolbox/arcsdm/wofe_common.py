import arcpy
import numpy as np
import os
import sys
import traceback



def get_output_map_units(silent=False):
    """
    Get the map units from the output coordinate system.
    """
    try:
        ocs = arcpy.env.outputCoordinateSystem
        if not ocs:
            if not silent:
                arcpy.AddWarning("Output coordinate system not set - defaulting mapunit to meter")
            return "meter"
        if ocs.type == 'Projected':
            return ocs.linearUnitName
        elif ocs.type == 'Geographic':
            return ocs.angularUnitName
        else:
            return None
    except arcpy.ExecuteError as error:
        if not all(error.args):
            arcpy.AddMessage("SDMValues caught arcpy.ExecuteError: ")
            args = error.args[0]
            args.split('\n')
            arcpy.AddError(args)
        raise
    except:
        tb = sys.exc_info()[2]
        errors = traceback.format_exc()
        arcpy.AddError(errors)



def get_study_area_size_sq_km(self, parameters, messages):
    if not arcpy.env.mask:
        raise ValueError("Mask doesn't exist! Set Mask under Analysis/Environments.")

    desc = arcpy.Describe(arcpy.env.mask)
    coord_sys = desc.spatialReference
    spatial_unit = coord_sys.linearUnitName if coord_sys.linearUnitName is not None else ""

    # TODO implement conversion
    if not spatial_unit.lower().strip() == "meter":
        raise ValueError(f"Output spatial unit should be meter, but was {spatial_unit}. Check output coordinate system in Environments!")

    masked_area_sq_m = 0.0

    if desc.dataType == "RasterLayer" or desc.dataType == "RasterBand":
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



def get_mask_area_in_km(mapUnits):
    """
    Return the mask size in square kilometers.
    """
    try:
        desc = arcpy.Describe(arcpy.env.mask)

        arcpy.AddMessage(f"mask datatype: {desc.dataType}")

        if desc.dataType in ["RasterLayer", "RasterBand", "RasterDataset"]:
            if not str(arcpy.env.cellSize).replace('.', '', 1).replace(',', '', 1).isdigit():
                arcpy.AddMessage("*" * 50)
                arcpy.AddError("ERROR: Cell Size must be numeric when mask is raster. Check Environments!")
                arcpy.AddMessage("*" * 50)
                raise SDMError

            arcpy.AddMessage("Counting raster size")
            arcpy.AddMessage("File: " + desc.catalogPath)
            rows = int(arcpy.GetRasterProperties_management(desc.catalogPath, "ROWCOUNT").getOutput(0))
            columns = int(arcpy.GetRasterProperties_management(desc.catalogPath, "COLUMNCOUNT").getOutput(0))
            raster_array = arcpy.RasterToNumPyArray(desc.catalogPath, nodata_to_value=-9999)
            area_sq_m = 0
            count = 0
            arcpy.AddMessage("Iterating through mask in numpy..." + str(columns) + "x" + str(rows))
            for i in range(rows):
                for j in range(columns):
                    if raster_array[i][j] != -9999:
                        count += 1
            arcpy.AddMessage("count:" + str(count))
            cellsize = float(str(arcpy.env.cellSize).replace(",", "."))
            area_sq_m = count * (cellsize * cellsize)

        elif desc.dataType in ["FeatureLayer", "FeatureClass", "ShapeFile"]:
            maskrows = arcpy.SearchCursor(desc.catalogPath)
            shapeName = desc.shapeFieldName
            maskrow = maskrows.next()
            area_sq_m = 0
            while maskrow:
                feat = maskrow.getValue(shapeName)
                area_sq_m += feat.area
                maskrow = maskrows.next()
            arcpy.AddMessage("count:" + str(area_sq_m))

        else:
            raise arcpy.ExecuteError(desc.dataType + " is not allowed as Mask!")

        mapUnits = mapUnits.lower().strip()
        if not mapUnits.startswith('meter'):
            arcpy.AddError('Incorrect output map units: Check units of study area.')
        conversion = getMapConversion(mapUnits)
        unit_cell_count = area_sq_m * conversion
        return unit_cell_count
    except arcpy.ExecuteError as e:
        raise
    except:
        tb = sys.exc_info()[2]
        tbinfo = traceback.format_tb(tb)[0]
        arcpy.AddError(tbinfo)
        if len(arcpy.GetMessages(2)) > 0:
            msgs = "SDM GP ERRORS:\n" + arcpy.GetMessages(2) + "\n"
            arcpy.AddError(msgs)
        raise


def log_wofe(unit_cell_area_sq_km, TrainPts):
    """
    Log WofE parameters to Geoprocessor History.
    """
    try:
        arcpy.AddMessage("\n" + "=" * 10 + " WofE parameters " + "=" * 10)

        mapUnits = get_output_map_units()
        mapUnits = mapUnits.lower().strip()
        if not mapUnits.startswith("meter"):
            arcpy.AddError("Output map unit should be meter: Check output coordinate system & units of study area.")

        arcpy.AddMessage("%-20s %s" % ("Map Units:", mapUnits))

        total_area = get_mask_area_in_km(mapUnits)

        if not arcpy.env.mask:
            arcpy.AddError("Study Area mask not set. Check Environments!")
        else:
            if not arcpy.Exists(arcpy.env.mask):
                arcpy.AddError(f"Mask {arcpy.env.mask} not found!")
            
            mask_descr = arcpy.Describe(arcpy.env.mask)
            arcpy.AddMessage("%-20s %s" % ("Mask:", "\"" + mask_descr.name + "\" and it is of type " + mask_descr.dataType))
            
            if mask_descr.dataType in ["FeatureLayer", "FeatureClass"]:
                arcpy.AddWarning("Warning: You should only use single value raster type masks!")
            
            arcpy.AddMessage("%-20s %s" % ("Mask size (km^2):", str(total_area)))

        if not arcpy.env.cellSize:
            arcpy.AddError("Study Area cellsize not set")
        if arcpy.env.cellSize == "MAXOF":
            arcpy.AddWarning("Cellsize should have definitive value?")

        cellsize = arcpy.env.cellSize
        arcpy.AddMessage("%-20s %s" % ("Cell Size:", cellsize))
        
        unit_cell_area_sq_km = float(unit_cell_area_sq_km)
        unit_cells_count = float(total_area) / unit_cell_area_sq_km

        # Note! GetCount does not care about the mask. Ie., it's assumed that
        # the mask has been already applied at training sites reduction.
        training_point_count = arcpy.management.GetCount(TrainPts)
        arcpy.AddMessage("%-20s %s" % ("# Training Sites:", training_point_count))
        arcpy.AddMessage("%-20s %s" % ("Unit Cell Area:", "{} km^2, Cells in area: {} ".format(unit_cell_area_sq_km, unit_cells_count)))

        if unit_cells_count == 0:
            arcpy.AddError("ERROR: 0 Cells in Area!")

        priorprob = float(str(training_point_count)) / float(unit_cells_count)

        if not (0 < priorprob <= 1.0):
            arcpy.AddError('Incorrect no. of training sites or unit cell area. TrainingPointsResult {}'.format(priorprob))

        arcpy.AddMessage("%-20s %0.6f" % ("Prior Probability:", priorprob))
        arcpy.AddMessage("%-20s %s" % ("Training Points:", arcpy.Describe(TrainPts).catalogPath))
        arcpy.AddMessage("%-20s %s" % ("Study Area Raster:", arcpy.Describe(arcpy.env.mask).catalogPath))
        arcpy.AddMessage("%-20s %s" % ("Study Area Area:", str(total_area) + " km^2"))
        arcpy.AddMessage("")
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
