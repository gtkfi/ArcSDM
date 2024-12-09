# -*- coding: utf-8 -*-

import arcpy
import importlib
import math
import os
import sys
import traceback

import arcsdm.common
import arcsdm.sdmvalues
import arcsdm.wofe_common

from arcsdm.common import log_arcsdm_details
from arcsdm.sdmvalues import log_wofe
from arcsdm.wofe_common import check_input_data, get_evidence_values_at_training_points, get_training_point_statistics

ASCENDING = "Ascending"
DESCENDING = "Descending"
CATEGORICAL = "Categorical"
UNIQUE = "Unique"


def extract_layer_from_raster_band(evidence_layer, evidence_descr):
    if evidence_descr.dataType == "RasterBand":
    # Try to change RasterBand to RasterDataset
        evidence1 = os.path.split(evidence_layer)
        evidence2 = os.path.split(evidence1[0])
        if (evidence1[1] == evidence2[1]) or (evidence1[1][:4] == "Band"):
            new_evidence_layer = evidence1[0]
            new_evidence_descr = arcpy.Describe(evidence_layer)
            arcpy.AddMessage("Evidence Layer is now " + new_evidence_layer + " and its data type is " + new_evidence_descr.dataType)
            return new_evidence_layer, new_evidence_descr
        else:
            arcpy.ExecuteError("ERROR: Data Type of Evidence Layer cannot be RasterBand, use Raster Dataset.")
    else:
        return evidence_layer, evidence_descr


def calculate_weights(pattern_tp_count, pattern_area_sq_km, unit_area_sq_km, tp_count, total_area_sq_km, class_category):
    if pattern_tp_count > tp_count:
        arcpy.AddWarning(f"Unable to calculate weights: more than one training point per unit cell in study area for class {class_category}!")
        return tuple([0.0] * 7)

    return calculate_weights_bonham_carter(pattern_tp_count, pattern_area_sq_km, unit_area_sq_km, tp_count, total_area_sq_km)


def calculate_weights_bonham_carter(pattern_tp_count, pattern_area_sq_km, unit_area_sq_km, tp_count, total_area_sq_km):
    """
    Calculate weights for a binary pattern.

    Based on 'Fortran Program for Calculating Weights of Evidence' in Appendix II of Bonham-Carter (1994).

    Args:
        pattern_tp_count: <int>
            Number of training points in the pattern.
        pattern_area_sq_km: <float>
            Area of binary pattern.
        unit_area_sq_km: <float>
            Area of unit cell.
        tp_count: <int>
            Number of training points in the area of study.
        total_area_sq_km: <float>
            Size of the area of study.

    Returns:
        A tuple of:
            W+
            Standard deviation of W+
            W-
            Standard deviation of W-
            Contrast
            Standard deviation of contrast
            Studentized contrast

    References:
        Bonham-Carter, Graeme F. (1994). Geographic Information Systems for Geoscientists: Modelling with GIS. Pergamon. Oxford. 1st Edition.
    """
    # Area of study region
    s = total_area_sq_km

    # Area of binary map pattern
    b = pattern_area_sq_km

    # TODO: Implement various traps to handle acceptable data anomalies
    # TODO: Implement non-zero check for denominators and logs

    # Area of unit cell
    unit = unit_area_sq_km

    # No of deposits on pattern
    db = pattern_tp_count

    # Total no of deposits
    ds = tp_count

    # Total area (unit cells)
    s = s / unit

    # Total area of binary pattern (unit cells)
    b = b / unit

    pbd = db / ds
    pbdb = (b - db) / (s - ds)

    # Odds ratio
    or_ = db * (s - b - ds + db) / (b - db) / (ds - db)

    # Sufficiency ratio LS
    ls = pbd / pbdb

    # W+
    wp = math.log(ls)

    vp = 1.0 / db + 1.0 / (b - db)
    # Standard deviation of W+
    sp = math.sqrt(vp)

    pbbd = (ds - db) / ds
    pbbdb = (s - b - ds + db) / (s - ds)

    # Necessity ratio LN
    ln = pbbd / pbbdb

    # W-
    wm = math.log(ln)

    vm = 1.0 / (ds - db) + 1.0 / (s - b - ds + db)
    # Standard deviation of W-
    sm = math.sqrt(vm)

    # Contrast
    c = wp - wm

    # Standard deviation of contrast
    sc = math.sqrt(vp + vm)

    # Prior probability
    priorp = ds / s

    vprip = priorp / s
    # Standard deviation of prior probability
    sprip = math.sqrt(vprip)

    # Standard deviation of prior log odds
    sprilo = sprip / priorp

    prioro = priorp / (1 - priorp)
    # Prior log odds
    prilo = math.log(prioro)

    # Conditional probability of deposit given pattern
    cpp = math.exp(prilo + wp)
    cpp = cpp / (1.0 + cpp)

    # Conditional probability of deposit given no pattern
    cpm = math.exp(prilo + wm)
    cpm = cpm / (1.0 + cpm)

    # Bonham-Carter's (1994) Fortran program additionally outputs the Studentized Contrast: c / sc
    stud_c = c / sc

    return (wp, sp, wm, sm, c, sc, stud_c)


def Calculate(self, parameters, messages):
    # TODO: make relevant checks to input parameters
    # TODO: convert evidence feature to raster
    # TODO: make sure mask is used in all steps that it needs to be used in
    # TODO: calculate weights based on weights type

    # TODO: Remove this after testing is done!
    # Make sure imported modules are refreshed if the toolbox is refreshed.
    importlib.reload(arcsdm.wofe_common)
    importlib.reload(arcsdm.sdmvalues)
    importlib.reload(arcsdm.common)

    arcpy.AddMessage("Starting weight calculation")
    arcpy.AddMessage("------------------------------")
    try:
        arcpy.env.overwriteOutput = True
        arcpy.AddMessage("overwriteOutput set to True")

        # Input parameters are as follows:
        # 0: EvidenceRasterLayer
        # 1: EvidenceRasterCodefield (what is this?)
        # 2: TrainingPoints
        # 3: Type
        # 4: OutputWeightsTable
        # 5: ConfidenceLevelOfStudentizedContrast
        # 6: UnitAreaKm2
        # 7: MissingDataValue
        # 8: Success

        evidence_raster = parameters[0].valueAsText
        code_name =  parameters[1].valueAsText

        # TODO: make sure the mask is applied to the features
        training_sites_feature = parameters[2].valueAsText
        selected_weight_type =  parameters[3].valueAsText
        output_weights_table = parameters[4].valueAsText
        studentized_contrast_threshold = parameters[5].value
        unit_area_sq_km = parameters[6].value
        nodata_value = parameters[7].value

        # Test data type of Evidence Layer
        evidence_descr = arcpy.Describe(evidence_raster)
        evidence_coord = evidence_descr.spatialReference

        common_coord_sys = evidence_coord
        if (arcpy.env.outputCoordinateSystem is not None) and (arcpy.env.outputCoordinateSystem.name.strip() != evidence_coord.name.strip()):
            common_coord_sys = arcpy.env.outputCoordinateSystem

        # arcpy.AddMessage(f"Coordinate system of output will be: {common_coord_sys.name}")
        # arcpy.AddMessage(f"Type: {common_coord_sys.type}")
        # arcpy.AddMessage(f"Linear unit: {common_coord_sys.linearUnitName}")
        # arcpy.AddMessage(f"Angular unit name: {common_coord_sys.angularUnitName}")

        evidence_raster, evidence_descr = extract_layer_from_raster_band(evidence_raster, evidence_descr)

        check_input_data([evidence_raster], training_sites_feature)
        
        # If using non gdb database, lets add .dbf
        # If using GDB database, remove numbers and underscore from the beginning of the Weights table name (else block)
        workspace_descr = arcpy.Describe(arcpy.env.workspace)
        if workspace_descr.workspaceType == "FileSystem":
            if not(output_weights_table.endswith(".dbf")):
                output_weights_table += ".dbf"
        else:
            wtsbase = os.path.basename(output_weights_table)
            while len(wtsbase) > 0 and (wtsbase[:1] <= "9" or wtsbase[:1] == "_"):
                wtsbase = wtsbase[1:]
            output_weights_table = os.path.dirname(output_weights_table) + "\\" + wtsbase

        mask = arcpy.env.mask
        if mask:
            if not arcpy.Exists(mask):
                raise arcpy.ExecuteError("Mask doesn't exist! Set Mask under Analysis/Environments.")
            
            evidence_raster = arcpy.sa.ExtractByMask(evidence_raster, mask)


        # TODO: project the evidence raster to have cell size equal to the unit area defined by the user
        # TODO: check map units, express unit cell size in map units
        # (arcpy.management.Resample can be used for this)



        # Note: arcpy.conversion.PointToRaster honors the following environments:
        # Auto Commit, Cell Size, Cell Size Projection Method, Compression, Current Workspace, 
        # Extent, Geographic Transformations, Output CONFIG Keyword, Output Coordinate System, 
        # Pyramid, Scratch Workspace, Snap Raster, Tile Size



        log_arcsdm_details()
        log_wofe(unit_area_sq_km, training_sites_feature)
        arcpy.AddMessage("=" * 10 + " Calculate weights " + "=" * 10)

        arcpy.AddMessage("%-20s %s (%s)" % ("Creating table: ", output_weights_table, selected_weight_type))

        # Calculate number of training sites in each class
        statistics_table, class_column_name, count_column_name = get_training_point_statistics(evidence_raster, training_sites_feature)

        # TODO: in both categorical cases:
        # TODO: get both the evidence and the training data as raster
        # TODO: in case of training data: set default value as 1 & fill value as 0
        # TODO: get the unique values in the evidence raster (should already be classified, so give a warning if there are a lot of classes)
        # TODO: -> those are the classes
        # TODO: for each class, get the weights
        # TODO: (probably faster to read the data into a numpy array instead of trying to work with cursors)
        # TODO: (note: probably easier to work with)
        
        codename_field = [] if (code_name is None or code_name == "") else ["CODE","text","5","#","#","Symbol"]

        base_fields = [["Class", "LONG"]] + codename_field + [
            ["Count", "LONG"], # Evidence count (temp)
            ["Frequency", "LONG"], # Training point count (temp)
        ]
        base_fields = base_fields
        base_field_names = [i[0] for i in base_fields]

        weight_fields = [
            ["WPLUS", "DOUBLE", "10", "4", "#", "W+"],
            ["S_WPLUS", "DOUBLE", "10", "4", "#", "W+ Std"],
            ["WMINUS", "DOUBLE", "10", "4", "#", "W-"],
            ["S_WMINUS", "DOUBLE", "10", "4", "#", "W- Std"],
            ["CONTRAST", "DOUBLE", "10", "4", "#", "Contrast_"],
            ["S_CONTRAST", "DOUBLE", "10", "4", "#", "Contrast_Std"],
            ["STUD_CNT","DOUBLE", "10", "4", "#", "Studentized_Contrast"]
        ]
        weight_field_names = [i[0] for i in weight_fields]

        generalized_weight_fields = [] if (selected_weight_type == UNIQUE) else [
            ["GEN_CLASS", "LONG", "#", "#", "#", "Generalized_Class"],
            ["WEIGHT", "DOUBLE", "10", "6", "#", "Generalized_Weight"],
            ["W_STD", "DOUBLE", "10", "6", "#", "Generalized_Weight_Std"]
        ]

        all_fields = base_fields + [
            ["Area", "DOUBLE"], # Area in km^2 (temp)
            ["AreaUnits", "DOUBLE"], # Area in unit cells (temp)
            ["AREA_SQ_KM", "DOUBLE"], # Area in km^2
            ["AREA_UNITS", "DOUBLE"], # Area in unit cells
            ["NO_POINTS", "LONG"], # Training point count
        ] + weight_fields + generalized_weight_fields

        # Generalized weights are for all but unique weights

        evidence_fields = ["VALUE", "COUNT"] + codename_field
        stats_fields = [class_column_name, count_column_name]

        arcpy.management.CreateTable(os.path.dirname(output_weights_table), os.path.basename(output_weights_table))
        # arcpy.management.AddFields doesn't allow setting field precision or scale, so add the fields individually
        for field_details in all_fields:
            arcpy.management.AddField(output_weights_table, *field_details)
        for field_name in weight_field_names:
            arcpy.management.AssignDefaultToField(output_weights_table, field_name, 0.0)

        arcpy.AddMessage("Created output weights table")

        evidence_attribute_table = evidence_raster
        if evidence_descr.dataType == "RasterLayer":
            evidence_attribute_table = evidence_descr.catalogPath

        order = "DESC" if (selected_weight_type == DESCENDING) else "ASC"
        order_clause = f"ORDER BY VALUE {order}"

        with arcpy.da.InsertCursor(output_weights_table, base_field_names) as cursor_weights:
            with arcpy.da.SearchCursor(evidence_attribute_table, evidence_fields, sql_clause=(None, order_clause)) as cursor_evidence:
                for row_evidence in cursor_evidence:
                    if (code_name is None or code_name == ""):
                        evidence_class, evidence_count = row_evidence
                    else:
                        evidence_class, evidence_count, code_name_field = row_evidence
                    site_count = 0

                    expression = f"{class_column_name} = {evidence_class}"

                    with arcpy.da.SearchCursor(statistics_table, stats_fields, where_clause=expression) as cursor_stats:
                        for row_stats in cursor_stats:
                            # Find the first training point row to have matching class (there should be just one)
                            if row_stats:
                                _, site_count = row_stats
                                break
                    
                    if (code_name is None or code_name == ""):
                        weights_row = (evidence_class, evidence_count, site_count)
                    else:
                        weights_row = (evidence_class, code_name_field, evidence_count, site_count)
                    cursor_weights.insertRow(weights_row)

        evidence_cellsize = evidence_descr.MeanCellWidth
        # Assuming linear units are in meters
        # Note! The logged WofE values are calculated with the output cellsize from the env settings
        # But actual calculation is done with the evidence cellsize
        # TODO: Do something about this
        arcpy.CalculateField_management(output_weights_table, "Area",  "!Count! * %f / 1000000.0" % (evidence_cellsize ** 2), "PYTHON_9.3")
        arcpy.CalculateField_management(output_weights_table, "AreaUnits",  "!Area! / %f" % unit_area_sq_km, "PYTHON_9.3")

        temp_fields = ["Frequency", "Area", "AreaUnits"]
        fields_to_update = ["NO_POINTS", "AREA_SQ_KM", "AREA_UNITS"]

        area_field_names = ["Class"] + temp_fields + fields_to_update
    
        training_point_count = 0
        total_area_sq_km = 0.0

        with arcpy.da.UpdateCursor(output_weights_table, area_field_names) as cursor_weights:
            frequency_tot = 0
            area_tot = 0.0
            area_units_tot = 0.0

            for weights_row in cursor_weights:
                class_category, frequency, area, area_units_temp, no_points, area_sq_km, area_units = weights_row

                # TODO: nodata value should be considered earlier as well?
                if (selected_weight_type in [ASCENDING, DESCENDING]) and (class_category != nodata_value):
                    frequency_tot += frequency
                    area_tot += area
                    area_units_tot += area_units_temp
                    no_points = frequency_tot
                    area_sq_km = area_tot
                    area_units = area_units_tot
                else:
                    if class_category != nodata_value:
                        training_point_count += frequency
                        total_area_sq_km += area

                    no_points = frequency
                    area_sq_km = area
                    area_units = area_units_temp

                updated_row = (class_category, frequency, area, area_units_temp, no_points, area_sq_km, area_units)
                cursor_weights.updateRow(updated_row)

            if selected_weight_type in [ASCENDING, DESCENDING]:
                training_point_count = frequency_tot
                total_area_sq_km = area_tot

        arcpy.AddMessage(f"Total number of training points: {training_point_count}, total area (km^2): {total_area_sq_km}")

        # temp_fields_to_delete = ["Count", "Frequency", "Area", "AreaUnits"]
        temp_fields_to_delete = ["Frequency", "Area", "AreaUnits"]
        arcpy.management.DeleteField(output_weights_table, temp_fields_to_delete)

        # Calculate weights
        # Required fields:
        # all the weights fields
        # "NO_POINTS", "AREA_SQ_KM"
        # also need: unit_area_sq_km, training_point_count, total_area

        fields_to_update = ["Class", "NO_POINTS", "AREA_SQ_KM"] + weight_field_names
        
        with arcpy.da.UpdateCursor(output_weights_table, fields_to_update) as cursor_weights:
            for weights_row in cursor_weights:
                class_category, no_points, area_sq_km, wplus, s_wplus, wminus, s_wminus, contrast, s_contrast, stud_cnt = weights_row

                if class_category != nodata_value:
                    wplus, s_wplus, wminus, s_wminus, contrast, s_contrast, stud_cnt = calculate_weights(no_points, area_sq_km, unit_area_sq_km, training_point_count, total_area_sq_km, selected_weight_type)

                    updated_row = (class_category, no_points, area_sq_km, wplus, s_wplus, wminus, s_wminus, contrast, s_contrast, stud_cnt)
                    cursor_weights.updateRow(updated_row)


    except arcpy.ExecuteError:
        arcpy.AddError(arcpy.GetMessages(2))
    except Exception:
        e = sys.exc_info()[1]
        print(e.args[0])
        arcpy.AddError(e.args[0])
