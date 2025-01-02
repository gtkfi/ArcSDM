"""
    ArcSDM 6 ToolBox for ArcGIS Pro

    Conversion and tool development for ArcGIS Pro by Geological Survey of Finland (GTK), 2025.

    Calculate Weights - ArcSDM 5 for ArcGis pro 
    Recode from the original by Tero Rönkkö / Geological survey of Finland
    Update by Arianne Ford, Kenex Ltd. 2018
   
    History:
    2025 Update to ArcGIS 3.x
    6.10.2020 If using GDB database, remove numbers and underscore from the beginning of the Weights table name / Arto Laiho, GTK/GFS
    21-23.7.2020 combined with Unicamp fixes (made 8.8.2018) / Arto Laiho, GTK/GFS
    9.6.2020 If Evidence Layer has not attribute table, execution is stopped / Arto Laiho, GTK/GSF
    3.6.2020 Evidence Raster cannot be RasterBand (ERROR 999999 at rows = gp.SearchCursor(EvidenceLayer)) / Arto Laiho, GTK/GSF
    15.5.2020 Added Evidence Layer and Training points coordinate system checking / Arto Laiho, GTK/GSF
    27.4.2020 Database table field name cannot be same as alias name when ArcGIS Pro with File System Workspace is used. / Arto Laiho, GTK/GSF
    09/01/2018 Bug fixes for 10.x, fixed perfect correlation issues, introduced patch for b-db<=0 - Arianne Ford, Kenex Ltd.
    3.11.2017 Updated categorical calculations when perfect correlation exists as described in issue 66
    27.9.2016 Calculate weights output cleaned
    23.9.2016 Goes through
    12.8.2016 First running version for pyt. Shapefile training points and output?
    1.8.2016 Python toolbox version started
    12.12.2016 Fixes
    
    Spatial Data Modeller for ESRI* ArcGIS 9.3
    Copyright 2009
    Gary L Raines, Reno, NV, USA: production and certification
    Don L Sawatzky, Spokane, WA, USA: Python software development
    Ascending or Descending:  Calculates accumulative Wts for accumulative row areas and num points
        for ascending or descending classes both with ascending counts.
    
    Categorical: Calculates Wts for each row, then reports those Wts for rows having >= Confidence.
        For rows having < Confidence, Wts are produced from sum of row areas and num points of
        classes < Confidence.
    Required Input(0): Integer raster dataset
    Optional Input(1): Attribute field name
    Required Input(2): Points feature class
    Required Input - Output type(3): Ascending, Descending, Categorical, Unique
    Required Output (4): Weights table
    Required Input(5): Confident_Contrast
    Required Input(6):  Unitarea
    Required Input(7): MissingDataValue
    Derived Output(8) - Success of calculation, whether Valid table: True or False
"""

import arcpy
import math
import os
import sys
import traceback

from arcsdm.common import log_arcsdm_details
from arcsdm.wofe_common import (
    apply_mask_to_raster,
    check_wofe_inputs,
    extract_layer_from_raster_band,
    get_study_area_parameters,
    get_training_point_statistics
)


ASCENDING = "Ascending"
DESCENDING = "Descending"
CATEGORICAL = "Categorical"
UNIQUE = "Unique"


def calculate_weights_sq_km(pattern_tp_count, pattern_area_sq_km, unit_area_sq_km, tp_count, total_area_sq_km, class_category):
    # Total area (unit cells)
    total_cell_count = total_area_sq_km / unit_area_sq_km

    # Total area of binary pattern (unit cells)
    pattern_cell_count = pattern_area_sq_km / unit_area_sq_km

    return calculate_weights(pattern_tp_count, pattern_cell_count, tp_count, total_cell_count, class_category)


def calculate_weights(pattern_tp_count, pattern_cell_count, tp_count, total_cell_count, class_category):
    if pattern_tp_count > tp_count:
        arcpy.AddWarning(f"Unable to calculate weights for class {class_category}: more than one training point per unit cell in study area!")
        return tuple([0.0] * 7)
    
    if pattern_tp_count == tp_count:
        # Fix the pattern deposit count so that the studentized contrast behaves better in the case of perfect correlation
        # See Issue #66 for details
        # Note: The old code had a comment about this not working when total_cell_count - pattern_cell_count < tp_count - pattern_tp_count
        pattern_tp_count -= 0.99
    
    if (pattern_tp_count == 0) or (tp_count == 0):
        return tuple([0.0] * 7)
    
    if (pattern_tp_count < 0) or (tp_count < 0):
        arcpy.AddWarning(f"Unable to calculate weights for class {class_category}: encountered non-positive training point count!")
        return tuple([0.0] * 7)
    
    if (pattern_cell_count < 0) or (total_cell_count <= 0):
        arcpy.AddWarning(f"Unable to calculate weights for class {class_category}: encountered non-positive area!")
        return tuple([0.0] * 7)

    if total_cell_count - tp_count <= 0.0:
        # TODO: fix, will result in division by zero
        pass

    if pattern_cell_count - pattern_tp_count <= 0.0:
        pattern_cell_count = pattern_tp_count + 1

    if (total_cell_count - pattern_cell_count) <= (tp_count - pattern_tp_count):
        pattern_cell_count = total_cell_count + pattern_tp_count - tp_count - 0.99

    return calculate_weights_bonham_carter(pattern_tp_count, pattern_cell_count, tp_count, total_cell_count)


def calculate_weights_bonham_carter(pattern_tp_count, pattern_area_cells, tp_count, total_area_cells):
    """
    Calculate weights for a binary pattern.

    Based on 'Fortran Program for Calculating Weights of Evidence' in Appendix II of Bonham-Carter (1994).

    Args:
        pattern_tp_count: <int>
            Number of training points in the pattern.
        pattern_area_cells: <float>
            Area of binary pattern in unit cells.
        tp_count: <int>
            Number of training points in the area of study.
        total_area_cells: <float>
            Size of the area of study in unit cells.

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
    s = total_area_cells

    # Area of binary map pattern
    b = pattern_area_cells

    # No of deposits on pattern
    db = pattern_tp_count

    # Total no of deposits
    ds = tp_count

    # The probability of binary pattern B being present, given the presence of a deposit
    pbd = db / ds

    # The probability of B being present, given the absence of a deposit
    pbdb = (b - db) / (s - ds)

    # Odds ratio
    # or_ = db * (s - b - ds + db) / (b - db) / (ds - db)

    # Sufficiency ratio LS
    ls = pbd / pbdb

    # W+
    wp = math.log(ls)

    vp = 1.0 / db + 1.0 / (b - db)
    # Standard deviation of W+
    sp = math.sqrt(vp)

    # The probability of B being absent, given the presence of a deposit
    pbbd = (ds - db) / ds

    # The probability of B being absent, given the absence of a deposit
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

    # vprip = priorp / s
    # Standard deviation of prior probability
    # sprip = math.sqrt(vprip)

    # Standard deviation of prior log odds
    # sprilo = sprip / priorp

    prioro = priorp / (1.0 - priorp)
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
    arcpy.AddMessage("Starting weight calculation")
    arcpy.AddMessage("------------------------------")
    try:
        arcpy.env.overwriteOutput = True
        arcpy.AddMessage("Setting overwriteOutput to True")

        evidence_raster = parameters[0].valueAsText
        code_name = parameters[1].valueAsText

        # TODO: make sure the mask is applied to the features
        training_sites_feature = parameters[2].valueAsText
        selected_weight_type =  parameters[3].valueAsText
        output_weights_table = parameters[4].valueAsText
        studentized_contrast_threshold = parameters[5].value
        unit_cell_area_sq_km = parameters[6].value
        nodata_value = parameters[7].value

        evidence_descr = arcpy.Describe(evidence_raster)
        evidence_raster, evidence_descr = extract_layer_from_raster_band(evidence_raster, evidence_descr)

        check_wofe_inputs([evidence_raster], training_sites_feature)
        
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

        # TODO: differentiate between original nodata value and missing data from applying study area mask?
        masked_evidence_raster = apply_mask_to_raster(evidence_raster, nodata_value)
        masked_evidence_descr = arcpy.Describe(masked_evidence_raster)
        # Evidence raster preparation is now done

        log_arcsdm_details()
        total_area_sq_km_from_mask, training_point_count = get_study_area_parameters(unit_cell_area_sq_km, training_sites_feature)

        arcpy.AddMessage("=" * 10 + " Calculate weights " + "=" * 10)
        arcpy.AddMessage("%-20s %s (%s)" % ("Creating table: ", output_weights_table, selected_weight_type))

        # Calculate number of training sites in each class
        statistics_table, class_column_name, count_column_name = get_training_point_statistics(masked_evidence_raster, training_sites_feature)
        
        codename_field = [] if (code_name is None or code_name == "") else ["CODE", "text", "5", "#", "#", "Symbol"]

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
        generalized_weight_field_names = [i[0] for i in generalized_weight_fields]

        all_fields = base_fields + [
            ["Area", "DOUBLE"], # Area in km^2 (temp)
            ["AreaUnits", "DOUBLE"], # Area in unit cells (temp)
            ["AREA_SQ_KM", "DOUBLE"], # Area in km^2
            ["AREA_UNITS", "DOUBLE"], # Area in unit cells
            ["NO_POINTS", "LONG"], # Training point count
        ] + weight_fields

        # Generalized weights are for all but unique weights
        if selected_weight_type != UNIQUE:
            all_fields = all_fields + generalized_weight_fields

        evidence_fields = ["VALUE", "COUNT"] + codename_field
        stats_fields = [class_column_name, count_column_name]

        arcpy.management.CreateTable(os.path.dirname(output_weights_table), os.path.basename(output_weights_table))
        # arcpy.management.AddFields doesn't allow setting field precision or scale, so add the fields individually
        for field_details in all_fields:
            arcpy.management.AddField(output_weights_table, *field_details)
        for field_name in weight_field_names:
            arcpy.management.AssignDefaultToField(output_weights_table, field_name, 0.0)

        arcpy.AddMessage("Created output weights table")

        evidence_attribute_table = masked_evidence_descr.catalogPath

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

        arcpy.management.Delete(statistics_table)

        evidence_cellsize = masked_evidence_descr.MeanCellWidth
        # TODO: 
        # Assumes linear units of evidence raster is in meters
        # (The mask unit is checked in get_study_area_parameters, but this should be done for the evidence layer as well.)
        arcpy.CalculateField_management(output_weights_table, "Area",  "!Count! * %f / 1000000.0" % (evidence_cellsize ** 2), "PYTHON_9.3")
        arcpy.CalculateField_management(output_weights_table, "AreaUnits",  "!Area! / %f" % unit_cell_area_sq_km, "PYTHON_9.3")

        temp_fields = ["Frequency", "Area", "AreaUnits"]
        fields_to_update = ["NO_POINTS", "AREA_SQ_KM", "AREA_UNITS"]

        area_field_names = ["Class"] + temp_fields + fields_to_update
    
        training_point_count = 0
        total_area_sq_km = 0.0

        with arcpy.da.UpdateCursor(output_weights_table, area_field_names) as cursor:
            frequency_tot = 0
            area_tot = 0.0
            area_units_tot = 0.0

            for row in cursor:
                class_category, frequency, area, area_units_temp, no_points, area_sq_km, area_units = row

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
                cursor.updateRow(updated_row)

            if selected_weight_type in [ASCENDING, DESCENDING]:
                training_point_count = frequency_tot
                total_area_sq_km = area_tot

        temp_fields_to_delete = ["Count", "Frequency", "Area", "AreaUnits"]
        arcpy.management.DeleteField(output_weights_table, temp_fields_to_delete)

        fields_to_update = ["Class", "NO_POINTS", "AREA_SQ_KM"] + weight_field_names
        
        with arcpy.da.UpdateCursor(output_weights_table, fields_to_update) as cursor:
            for row in cursor:
                class_category, no_points, area_sq_km, wplus, s_wplus, wminus, s_wminus, contrast, s_contrast, stud_cnt = row

                if class_category != nodata_value:
                    wplus, s_wplus, wminus, s_wminus, contrast, s_contrast, stud_cnt = calculate_weights_sq_km(no_points, area_sq_km, unit_cell_area_sq_km, training_point_count, total_area_sq_km, selected_weight_type)

                    updated_row = (class_category, no_points, area_sq_km, wplus, s_wplus, wminus, s_wminus, contrast, s_contrast, stud_cnt)
                    cursor.updateRow(updated_row)

        # Required fields for generalization
        fields_to_read = ["OBJECTID", "CLASS", "AREA_UNITS", "NO_POINTS"] + weight_field_names
        fields_to_update = ["OBJECTID", "CLASS"] + generalized_weight_field_names

        # Generalize weights for non-unique weight selections
        if selected_weight_type in [ASCENDING, DESCENDING]:
            # Find the row with the maximum contrast value that has a studentized contrast that satisfies the threshold condition
            max_contrast_OID = -1
            max_contrast = -9999999.0
            tp_count = 0
            area_cell_count = 0.0
            max_wplus = -9999.0
            max_s_wplus = -9999.0
            max_wminus = -9999.0
            max_s_wminus = -9999.0
            max_std_contrast = -9999.0

            threshold_clause = f"STUD_CNT >= {studentized_contrast_threshold}"
            with arcpy.da.SearchCursor(output_weights_table, fields_to_read, where_clause=threshold_clause) as cursor_weights:
                for row in cursor_weights:
                    oid, class_category, area_units, no_points, wplus, s_wplus, wminus, s_wminus, contrast, s_contrast, stud_cnt = row
                    
                    if class_category != nodata_value:
                        if contrast > max_contrast:
                            max_contrast_OID = oid
                            max_contrast = contrast
                            tp_count = no_points
                            area_cell_count = area_units
                            max_wplus = wplus
                            max_s_wplus = s_wplus
                            max_wminus = wminus
                            max_s_wminus = s_wminus

            if max_contrast_OID >= 0:
                update_clause = f"OBJECTID <= {max_contrast_OID}"
                
                with arcpy.da.UpdateCursor(output_weights_table, fields_to_update) as cursor_generalized:
                    for row in cursor_generalized:
                        oid, class_category, gen_class, weight, w_std = row
                        
                        if class_category == nodata_value:
                            gen_class = nodata_value
                            weight = 0.0
                            w_std = 0.0
                        else:
                            if oid <= max_contrast_OID:
                                gen_class = 2
                                weight = max_wplus
                                w_std = max_s_wplus
                            else:
                                gen_class = 1
                                weight = max_wminus
                                w_std = max_s_wminus

                        updated_row = (oid, class_category, gen_class, weight, w_std)
                        cursor_generalized.updateRow(updated_row)
            else:
                arcpy.AddWarning(f"Unable to generalize weights! No contrast for type {selected_weight_type} satisties the user-defined confidence level {studentized_contrast_threshold}")
                arcpy.AddWarning(f"Table {output_weights_table} is incomplete.")

        elif selected_weight_type == CATEGORICAL:
            # Reclassify
            reclassified = False

            tp_count_99 = 0
            unit_cell_count_99 = 0.0
            tp_count = 0
            unit_cell_count = 0.0

            with arcpy.da.UpdateCursor(output_weights_table, ["Class", "AREA_UNITS", "NO_POINTS"] + weight_field_names + generalized_weight_field_names) as cursor:
                for row in cursor:
                    class_category, area_units, no_points, wplus, s_wplus, wminus, s_wminus, contrast, s_contrast, stud_cnt, gen_class, weight, w_std = row

                    # TODO: Verify if the missing data class should be generalized to the outside class as well
                    if (class_category != nodata_value) and (abs(stud_cnt) < studentized_contrast_threshold):
                        gen_class = 99
                        tp_count_99 += no_points
                        unit_cell_count_99 += area_units
                        reclassified = True
                    else:
                        gen_class = class_category

                    tp_count += no_points
                    unit_cell_count += area_units

                    # Set generalized weights to defaults (will be updated for class 99)
                    weight = wplus
                    w_std = s_wplus
                    
                    updated_row = (class_category, area_units, no_points, wplus, s_wplus, wminus, s_wminus, contrast, s_contrast, stud_cnt, gen_class, weight, w_std)
                    cursor.updateRow(updated_row)

            if not reclassified:
                arcpy.AddWarning(f"Unable to generalize classes with the given studentized contrast threshold!")
            else:
                gen_weight_99, gen_w_std_99, _, _, _, _, _ = calculate_weights(tp_count_99, unit_cell_count_99, tp_count, unit_cell_count, 99)

                arcpy.AddMessage(f"Generalized weight: {gen_weight_99}, STD of generalized weight: {gen_w_std_99}")

                categorical_clause = f"GEN_CLASS = 99"

                with arcpy.da.UpdateCursor(output_weights_table, generalized_weight_field_names, where_clause=categorical_clause) as cursor_generalized:
                    for row in cursor_generalized:
                        gen_class, weight, w_std = row

                        weight = gen_weight_99
                        w_std = gen_w_std_99

                        updated_row = (gen_class, weight, w_std)
                        cursor_generalized.updateRow(updated_row)

    except arcpy.ExecuteError:
        arcpy.AddError(arcpy.GetMessages(2))
    except Exception:
        tb = sys.exc_info()[2]
        tbinfo = traceback.format_tb(tb)[0]
        
        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
            str(sys.exc_info()) + "\n"
        msgs = "GP ERRORS:\n" + arcpy.GetMessages(2) + "\n"

        arcpy.AddError(msgs)
        arcpy.AddError(pymsg)
