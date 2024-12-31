"""
    Spatial Data Modeller for ESRI* ArcGIS 9.2
    Copyright 2007
    Gary L Raines, Reno, NV, USA: production and certification
    Don L Sawatzky, Spokane, WA, USA: Python software development
    15.6.2020 added "arcsdm." to import FloatRasterSearchcursor. "except Exception,Msg:" was invalid syntax / Arto Laiho, GTK/GFS

# ---------------------------------------------------------------------------
# MissingDataVar.py
# Created on: Fri Sep 29 2006 04:17:38 PM
#   (generated by ArcGIS/ModelBuilder)
# Arguments: geoprocessing object, list of wts tables,post probability raster, output raster
# Usage: AreaStatsMdl <rclssb2_CD> <Expression> <Statistics_Field_s_> <tempstat03_dbf> <Case_field> 
# Description: 
# Get sum of Area_sq_km field for Case = GEN_CLASS for class areas or Case empty for Total Data Area
# ---------------------------------------------------------------------------
    Gary's suggestion:
    Cell A2: [R1] = CON([rclssb2_md_w] == 0.0, [crap_pprb], 0.0)
    Cell E2: [UPDPOSTODDS] = CON([R2] == 0.0, 0.0, Exp(Ln([R2]) + [rclssb2_md_w]))
"""
import arcpy
import os
import sys
import traceback

from arcsdm.floatingrasterarray import FloatRasterSearchcursor
from arcsdm.wofe_common import get_area_size_sq_km


# def TotalAreaFromCounts(gp, Input_Raster, CellSize):
#     IsNul_Wts = gp.createuniquename(("IsNul_" + os.path.basename(Input_Raster))[:11], gp.scratchworkspace)
#     gp.IsNull_sa(Input_Raster, IsNul_Wts)
#     rasrows = gp.SearchCursor(IsNul_Wts, 'Value = 0')
#     rasrow = rasrows.Next()
#     TotalCount = rasrow.Count
#     arcpy.management.Delete(arcpy.Describe(IsNul_Wts).catalogPath)
#     return float(TotalCount) * CellSize * CellSize / 1000000


def create_missing_data_variance_raster(gp, masked_weights_rasters, masked_post_probability_raster, output_raster_name):
    """
    Calculate a raster of the variance of the posterior probability due to missing predictor patterns.

    Args:
        gp:
            Arcgisscripting geoprocessing object.
        masked_weights_rasters:
            A list containing a raster of weight values for each evidence pattern.
            The study area mask should have already been applied to the rasters.
        masked_post_probability_raster:
            A post probability raster. The study area mask should have been applied to the raster.
        output_raster_name:
            A name for the missing data variance raster that will be created as output.
    """
    try:
        i = 0
        
        # Create Total Missing Data Variance list
        md_variance_for_all_patterns = []

        # Loop throught Wts Rasters
        for Wts_Raster0 in masked_weights_rasters:
            arcpy.AddMessage(f"Missing data Variance for: {Wts_Raster0}")
            Wts_Raster = arcpy.Describe(Wts_Raster0).catalogPath

            # TODO!: this should probably actually be the study area size,
            # not just the area of the pattern where data is present
            # (literature is unclear)
            # pattern_area_sq_km = TotalAreaFromCounts(gp, Wts_Raster, CellSize)
            
            # Get pattern area where data is not missing
            pattern_area_sq_km = get_area_size_sq_km(Wts_Raster0)
            arcpy.AddMessage('TotDataArea = %.0f' % pattern_area_sq_km)

            desc = arcpy.Describe(Wts_Raster0)
            cell_size_sq_m = desc.MeanCellWidth * desc.MeanCellHeight
            
            # Start MD Variance raster
            # Get PostProb raster of MD cells
            pprb_where_pattern_is_missing_data = os.path.join(arcpy.env.scratchWorkspace, "R1")
            if arcpy.Exists(pprb_where_pattern_is_missing_data):
                arcpy.management.Delete(pprb_where_pattern_is_missing_data)

            pprb_expression = "CON(%s == 0.0,%s,0.0)" % (Wts_Raster, masked_post_probability_raster)
            arcpy.AddMessage(f"R1={pprb_expression}")
            gp.SingleOutputMapAlgebra_sa(pprb_expression, pprb_where_pattern_is_missing_data)

            # Get PostODDs raster of MD cells
            post_odds_raster_r2 = os.path.join(arcpy.env.scratchWorkspace, "R2")
            if arcpy.Exists(post_odds_raster_r2):
                arcpy.management.Delete(post_odds_raster_r2)

            post_odds_expression = "%s / (1.0 - %s)" % (pprb_where_pattern_is_missing_data, pprb_where_pattern_is_missing_data)
            arcpy.AddMessage(f"R2={post_odds_expression}")
            gp.SingleOutputMapAlgebra_sa(post_odds_expression, post_odds_raster_r2)
            arcpy.AddMessage(f"R2 exists: {arcpy.Exists(post_odds_raster_r2)}")
            
            # Get Total Variance of MD cells
            # Create total class variances list

            # Calculate the variance raster of each class (W+ & W-)
            class_md_variances_for_pattern = []

            # FloatRasterSearchcursor will contain the frequencies of each unique value of the raster
            # Basically this should just be 3 unique value classes: W+ & W- & 0.0 (for where weights could not be calculated)
            pattern_weights = FloatRasterSearchcursor(Wts_Raster)
            j = 0
            """ Cannot be done by single raster; must generate a raster for each value """
            for weight in pattern_weights:
                # NOTE: Assumes that previously, if weights could not be calculated for a class, its weights have been replaced with 0.0
                if weight.value == 0.0:
                    continue
                temp_variance_raster = str(os.path.join(arcpy.env.scratchWorkspace, "ClsVar%s%s" % (i, j)))
                j += 1
                if arcpy.Exists(temp_variance_raster):
                    arcpy.management.Delete(temp_variance_raster)
                
                # Calculate posterior odds as if missing data in the pattern was known - compensate with weight value for the class
                Exp1 = 'CON(%s == 0.0,0.0,EXP(LN(%s) + %s))' % (post_odds_raster_r2, post_odds_raster_r2, weight.value)
                # Updated posterior probability
                Exp2 = "%s / (1 + %s)" % (Exp1, Exp1)
                class_area_sq_km = float(weight.Count) * cell_size_sq_m * 0.000001
                Exp3 = "SQR(%s - %s) * (%s / %s)" % (Exp2, pprb_where_pattern_is_missing_data, class_area_sq_km, pattern_area_sq_km)
                gp.SingleOutputMapAlgebra_sa(Exp3, temp_variance_raster)
                class_md_variances_for_pattern.append(str(temp_variance_raster)) # Save the class variance raster

            del pattern_weights

            # Sum the class variances to get the variance of the posterior probability due to the missing evidence pattern
            temp_total_pattern_MD_variance = os.path.join(arcpy.env.scratchWorkspace, "TotClsVar%s" % i)
            i += 1
            if arcpy.Exists(temp_total_pattern_MD_variance):
                arcpy.Delete(temp_total_pattern_MD_variance)
            
            post_odds_expression = "SUM%s" % str(tuple(class_md_variances_for_pattern))
            gp.SingleOutputMapAlgebra_sa(post_odds_expression, temp_total_pattern_MD_variance)
            md_variance_for_all_patterns.append(str(temp_total_pattern_MD_variance))

            for tmp_raster in class_md_variances_for_pattern:
                arcpy.management.Delete(arcpy.Describe(tmp_raster).catalogPath)
               
        # Create Total Missing Data Variance raster
        else:
            if len(masked_weights_rasters) > 0:
                TotVarMD = output_raster_name
                post_odds_expression = "SUM%s" % str(tuple(md_variance_for_all_patterns))
                gp.SingleOutputMapAlgebra_sa(post_odds_expression, TotVarMD)

        arcpy.management.Delete(arcpy.Describe(pprb_where_pattern_is_missing_data).catalogPath)
        arcpy.management.Delete(arcpy.Describe(post_odds_raster_r2).catalogPath)
        
        for tmp_raster in md_variance_for_all_patterns:
            arcpy.management.Delete(arcpy.Describe(tmp_raster).catalogPath)

    except Exception:
        tb = sys.exc_info()[2]
        tbinfo = traceback.format_tb(tb)[0]

        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
            str(sys.exc_info()[0])+ ": " + str(sys.exc_info()[1]) + "\n"
        msgs = "GP ERRORS:\n" + arcpy.GetMessages(2) + "\n"

        arcpy.AddError(msgs)
        arcpy.AddError(pymsg)
