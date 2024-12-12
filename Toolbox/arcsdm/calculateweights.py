import arcpy
import math
import os
import sys
import traceback

from arcsdm.common import log_arcsdm_details
from arcsdm.sdmvalues import log_wofe
from arcsdm.wofe_common import check_input_data, get_training_point_statistics

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
        code_name =  parameters[1].valueAsText

        # TODO: make sure the mask is applied to the features
        training_sites_feature = parameters[2].valueAsText
        selected_weight_type =  parameters[3].valueAsText
        output_weights_table = parameters[4].valueAsText
        studentized_contrast_threshold = parameters[5].value
        unit_area_sq_km = parameters[6].value
        nodata_value = parameters[7].value

        evidence_descr = arcpy.Describe(evidence_raster)
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

        log_arcsdm_details()
        log_wofe(unit_area_sq_km, training_sites_feature)
        arcpy.AddMessage("=" * 10 + " Calculate weights " + "=" * 10)

        arcpy.AddMessage("%-20s %s (%s)" % ("Creating table: ", output_weights_table, selected_weight_type))

        # Calculate number of training sites in each class
        statistics_table, class_column_name, count_column_name = get_training_point_statistics(evidence_raster, training_sites_feature)
        
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

        arcpy.management.Delete(statistics_table)

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

        temp_fields_to_delete = ["Count", "Frequency", "Area", "AreaUnits"]
        arcpy.management.DeleteField(output_weights_table, temp_fields_to_delete)

        fields_to_update = ["Class", "NO_POINTS", "AREA_SQ_KM"] + weight_field_names
        
        with arcpy.da.UpdateCursor(output_weights_table, fields_to_update) as cursor_weights:
            for weights_row in cursor_weights:
                class_category, no_points, area_sq_km, wplus, s_wplus, wminus, s_wminus, contrast, s_contrast, stud_cnt = weights_row

                if class_category != nodata_value:
                    wplus, s_wplus, wminus, s_wminus, contrast, s_contrast, stud_cnt = calculate_weights_sq_km(no_points, area_sq_km, unit_area_sq_km, training_point_count, total_area_sq_km, selected_weight_type)

                    updated_row = (class_category, no_points, area_sq_km, wplus, s_wplus, wminus, s_wminus, contrast, s_contrast, stud_cnt)
                    cursor_weights.updateRow(updated_row)

        # Required fields for generalization
        fields_to_read = ["OBJECTID", "CLASS", "AREA_UNITS", "NO_POINTS"] + weight_field_names
        fields_to_update = ["OBJECTID", "CLASS"] + generalized_weight_field_names

        # Generalize weights for non-unique weight selections
        if selected_weight_type in [ASCENDING, DESCENDING]:
            # Find the row with the maximum contrast value
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

            with arcpy.da.UpdateCursor(output_weights_table, ["Class", "AREA_UNITS", "NO_POINTS"] + weight_field_names + generalized_weight_field_names) as cursor_categorical:
                for row in cursor_categorical:
                    class_category, area_units, no_points, wplus, s_wplus, wminus, s_wminus, contrast, s_contrast, stud_cnt, gen_class, weight, w_std = row

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
                    cursor_categorical.updateRow(updated_row)

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
        e = sys.exc_info()[1]
        print(e.args[0])
        arcpy.AddError(e.args[0])
