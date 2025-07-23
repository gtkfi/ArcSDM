""" ArcSDM 6 ToolBox for ArcGIS Pro

Conversion and tool development for ArcGIS Pro by Geological Survey of Finland (GTK), 2025.

Area frequency tool 

Spatial Data Modeller for ESRI* ArcGIS 9.3
Copyright 2009
Gary L Raines, Reno, NV, USA: production and certification
Don L Sawatzky, Spokane, WA, USA: Python software development

Input raster is a floating raster that has no VAT.
However, can extract raster values to points.

Can get the frequency>0 of raster values from extracted points.
Because the floating raster values in the frequency table are
only those with a frequency of 1 or more,CANNOT get the remainder
of float raster values because they have no frequency.

Modified to accept an integer raster. Cannot substitute another attribute as Value
in this operation, because cannot ExtractValueToPoints for an attribute, only Value.
In case of NN output raster, user must generate a new floating raster for RBFLN, PNN,
FZZYCLSTR attributes, so proper extraction can be done.

Creates six lists, each of length equal to the VAT count.  This algorithm is designed to
operate on, say, less than 1000 training sites and 10,000 raster values.  Grids with real data
are desirable.  Images won't do.

Creates another dictionary of length equal to the Statistics table generated from raster and
training sites.

Makes five passes through the output table.  So, table cannot be long.

The floating raster VAT is a dictionary defined in floatingrasterclass.py;
with VAT[Value] = COUNT, like a VAT of an integer raster.
VAT.next() returns (ID, VALUE, COUNT)

"""
import arcpy
import os

from arcsdm.floatingrasterarray import FloatRasterVAT
from arcsdm.wofe_common import apply_mask_to_raster, get_study_area_parameters, get_training_point_statistics


class UserException(Exception):
    pass


def Execute(self, parameters, messages):
    arcpy.CheckOutExtension("Spatial")
    arcpy.env.overwriteOutput = True

    training_point_feature = parameters[0].valueAsText
    evidence_raster = parameters[1].valueAsText
    value_field = parameters[2].valueAsText
    unit_cell_area_sq_km = parameters[3].value
    output_table = parameters[4].valueAsText
    
    # Log WofE values
    _, total_training_point_count = get_study_area_parameters(unit_cell_area_sq_km, training_point_feature)

    arcpy.AddMessage("\n" + "=" * 10 + " Starting area frequency " + "=" * 10)

    evidence_info = arcpy.Raster(evidence_raster)

    if evidence_info.hasRAT and not value_field.strip().lower() in [n.baseName.strip().lower() for n in arcpy.ListFields(evidence_raster)]:
        raise UserException(f"The evidence raster {evidence_raster} does not have a field {value_field}!")
    
    # NoData value to use internally
    nodata_value = -9999
    masked_evidence_raster = apply_mask_to_raster(evidence_raster, nodata_value)

    # Create a summary statistics table
    statistics_table, class_column_name, count_column_name = get_training_point_statistics(masked_evidence_raster, training_point_feature)

    # Calculate the number of sites on the pattern
    missing_pattern_training_point_count = 0
    pattern_training_point_count = 0

    stats_fields = [class_column_name, count_column_name]
    with arcpy.da.SearchCursor(statistics_table, stats_fields) as cursor:
        for row in cursor:
            class_category, count = row
            if class_category == nodata_value:
                missing_pattern_training_point_count += count
            else: pattern_training_point_count += count
    
    # Check if the counts match the number of training sites
    if pattern_training_point_count != (total_training_point_count - missing_pattern_training_point_count):
        arcpy.AddWarning("Stats count and number of training sites in data area do not compare: training points exists outside of the evidence pattern")
    if missing_pattern_training_point_count > 0:
        arcpy.AddWarning(f"{missing_pattern_training_point_count} training points where evidence pattern is missing in the study area.")

    arcpy.AddMessage(f"Training sites within the study area that fall on the evidence pattern: {pattern_training_point_count}")
    
    # Create the output table
    arcpy.AddMessage(f"Creating table: {output_table}")
    arcpy.management.CreateTable(os.path.dirname(output_table), os.path.basename(output_table))

    arcpy.management.AddField(output_table, "Frequency", "LONG")
    arcpy.management.AddField(output_table, "RASTERVALU", "DOUBLE", field_precision="18", field_scale="8")
    arcpy.management.AddField(output_table, "Area_sqkm", "DOUBLE", field_alias="Area_Sq_Kilometers")
    arcpy.management.AddField(output_table, "CAPP_CumAr", "DOUBLE", field_alias="CAPP_Cumulative_Area")
    arcpy.management.AddField(output_table, "Eff_CumAre", "DOUBLE", field_alias="Efficiency_Cumulative_Area")
    arcpy.management.AddField(output_table, "Cum_Sites", "DOUBLE", field_alias="Cumulative_Sites")
    arcpy.management.AddField(output_table, "I_CumSites", "DOUBLE", field_alias="Cumulative_Sites")
    arcpy.management.AddField(output_table, "Eff_AUC", "DOUBLE", field_alias="A_U_C")
    
    # Calculate the area factor
    # NOTE: Assumes meters - this should be updated if we start supporting non-meter rasters in the future
    evidence_descr = arcpy.Describe(masked_evidence_raster)
    cell_size_sq_m = evidence_descr.MeanCellWidth * evidence_descr.MeanCellHeight
    # NOTE: This was previously divided by the unit cell area, resulting in the Area_sqkm column having the area in unit cells, not sq. km!
    factor = cell_size_sq_m / 1000000.0
    arcpy.AddMessage(f"factor: {factor}")
    
    flt_ras = FloatRasterVAT(masked_evidence_raster)
    rasrows = flt_ras.FloatRasterSearchcursor()

    # Initialize variables for cumulative calculations
    pattern_area = 0.0
    cumulative_area = 0
    effective_area_by_class = []
    site_count_by_class = []

    # Update the output table with area & training point frequency
    with arcpy.da.InsertCursor(output_table, ["RASTERVALU", "Area_sqkm", "Frequency"]) as cursor:
        for rasrow in rasrows:
            class_category = int(rasrow.value)
            area_sqkm = rasrow.count * factor
            site_count = 0

            expression = f"{class_column_name} = {class_category}"

            with arcpy.da.SearchCursor(statistics_table, [class_column_name, count_column_name], where_clause=expression) as cursor_stats:
                for row_stats in cursor_stats:
                    if row_stats:
                        _, site_count = row_stats
                        break
            
            pattern_area += area_sqkm
            effective_area_by_class.append(area_sqkm)
            site_count_by_class.append(site_count)

            cursor.insertRow((rasrow.value, area_sqkm, site_count))
    
    arcpy.management.Delete(statistics_table)
    
    # Calculate cumulative areas and sites
    effective_area_by_class_rev = reversed(effective_area_by_class)
    site_count_by_class_rev = reversed(site_count_by_class)
    effective_cumulative_area = 0
    cumulative_site_count = 0
    effective_cumulative_area_list = []
    cumulative_site_count_list = []

    for i in range(len(site_count_by_class)):
        effective_cumulative_area += 100.0 * next(effective_area_by_class_rev) / pattern_area
        effective_cumulative_area_list.append(effective_cumulative_area)
        cumulative_site_count += 100.0 * next(site_count_by_class_rev) / pattern_training_point_count
        cumulative_site_count_list.append(cumulative_site_count)
    
    effective_cumulative_area_list_rev = reversed(effective_cumulative_area_list)
    cumulative_site_count_list_rev = reversed(cumulative_site_count_list)

    # Update the output table with cumulative values
    with arcpy.da.UpdateCursor(output_table, ["Area_sqkm", "CAPP_CumAr", "Eff_CumAre", "Cum_Sites", "I_CumSites"]) as cursor:
        for row in cursor:
            cumulative_area += 100.0 * row[0] / pattern_area
            row[1] = cumulative_area
            row[2] = next(effective_cumulative_area_list_rev)
            cumulative_site_count_by_class = next(cumulative_site_count_list_rev)
            row[3] = cumulative_site_count_by_class
            row[4] = 100.0 - cumulative_site_count_by_class
            cursor.updateRow(row)
    
    # Calculate efficiency and AUC values
    effective_cumulative_area_by_class = []
    cumulative_site_count_by_class = []
    with arcpy.da.SearchCursor(output_table, ["Eff_CumAre", "Cum_Sites"]) as cursor:
        next(cursor)  # Skip the first row
        for row in cursor:
            effective_cumulative_area_by_class.append(row[0])
            cumulative_site_count_by_class.append(row[1])
    
    sum_efficiency_AUC = 0.0
    with arcpy.da.UpdateCursor(output_table, ["Eff_CumAre", "Cum_Sites", "Eff_AUC"]) as cursor:
        for i, row in enumerate(cursor):
            if i < len(effective_cumulative_area_by_class):
                val = 0.5 * (row[0] - effective_cumulative_area_by_class[i]) * (row[1] + cumulative_site_count_by_class[i]) / (100.0 * 100.0)
                sum_efficiency_AUC += val
                row[2] = val
                cursor.updateRow(row)
            else:
                val = 0.5 * (row[0]) * (row[1]) / (100.0 * 100.0)
                sum_efficiency_AUC += val
                row[2] = val
                cursor.updateRow(row)
    
    arcpy.AddMessage("Efficiency: %.1f%%" % (sum_efficiency_AUC * 100.0))
