# -*- coding: utf-8 -*-
""" ArcSDM 6 ToolBox for ArcGIS Pro

Conversion and tool development for ArcGIS Pro by Geological Survey of Finland (GTK), 2024.

Spatial Data Modeller for ESRI* ArcGIS 9.2

Copyright 2007
Gary L Raines, Reno, NV, USA: production and certification
Don L Sawatzky, Spokane, WA, USA: Python software development

Converted to ArcSDM toolbox to ArcMap by GTK 2016
"""
import arcpy
import math
import os
import sys
import traceback

# import arcpy.management

from arcsdm.floatingrasterarray import FloatRasterSearchcursor
from arcsdm.wofe_common import get_study_area_parameters


def Calculate(self, parameters, messages):
    messages.addMessage("Starting Agterberg-Cheng CI Test")
    try:
        arcpy.CheckOutExtension("Spatial")
        arcpy.env.overwriteOutput = True

        postprob_raster = parameters[0].valueAsText
        std_raster = parameters[1].valueAsText
        training_points_feature = parameters[2].valueAsText
        unit_cell_area_sq_km = parameters[3].value
        output_txt_file = parameters[4].valueAsText

        postprob_descr = arcpy.Describe(postprob_raster)
        postprob_raster_path = postprob_descr.catalogPath
        std_raster_path = arcpy.Describe(std_raster).catalogPath

        basename = os.path.basename(postprob_raster)
        sdmuc = basename.split("_")[0]

        cell_size_sq_m = postprob_descr.MeanCellWidth * postprob_descr.MeanCellHeight
        km_in_m = 0.000001
        conversion_factor = cell_size_sq_m * km_in_m / unit_cell_area_sq_km

        postprob_raster_stats = FloatRasterSearchcursor(postprob_raster_path)
        # Predicted frequency of deposits in the study area - sum of posterior probabilities
        PredT = 0.0

        for distinct_value in postprob_raster_stats:
            PredT += (distinct_value.value * distinct_value.count)

        PredT *= conversion_factor

        _, training_point_count = get_study_area_parameters(unit_cell_area_sq_km, training_points_feature)

        std_raster_stats = FloatRasterSearchcursor(std_raster_path)
        total_variance = 0.0

        for distinct_std_value in std_raster_stats:
            total_variance += (distinct_std_value.value * distinct_std_value.count * conversion_factor) ** 2

        total_std = math.sqrt(total_variance)
        TS = (PredT - training_point_count) / total_std

        # PostProb

        # Total number of discrete events
        n = training_point_count
        # Sum of posterior probabilities
        T = PredT
        # STD = TStd
        P = ZtoF(TS) * 100.0
        if P >= 50.0:
            overall_CI = 100.0 * (100.0 - P) / 50.0
        else:
            overall_CI = 100.0 * (100.0 - (50 + (50 - P))) / 50.0

        Text = """
        Overall CI: %(0).1f%%\r
        Conditional Independence Test: %(1)s\r
        Observed No. training pts, n = %(2)i\r
        Expected No. of training points, T = %(3).1f\r
        Difference, T-n = %(4).1f\r
        Standard Deviation of T = %(5).3f\r
        \r
        ------------------------------------------------\r
        Conditional Independence Ratio: %(6).2f <simply the ratio n/T>\r
        Values below 1.00 may indicate conditional dependence\r
        among two or more of your data sets.  <Bonham-Carter(1994,ch.9)\r
        suggest that values <0.85 may indicate a problem>\r
        \r
        ------------------------------------------------\r
        Agterberg & Cheng Conditional Independence Test\r
        <See Agterberg and Cheng, Natural Resources Research 11(4), 249-255, 2002>\r
        This is a one-tailed test of the null hypothesis that T-n=0.  The test\r
        statistic is (T-n)/standard deviation of T. Probability values greater\r
        than 95%% or 99%% indicate that the hypothesis of CI should be rejected,\r
        but any value greater than 50%% indicates that some conditional\r
        dependence occurs>\r
        \r
        Probability that this model is not conditionally independent with\r
        (T-n)/Tstd = %(7).6f is %(8).1f%%\r
        ------------------------------------------------\r
        \r
        Input Data:\r
        Post Probability: %(9)s\r
        Post Probability Std Deviation: %(10)s\r
        Training Sites: %(11)s
        \r
        """ % {'0': overall_CI, '1': sdmuc, '2': n, '3': T, '4': T-n, '5': total_std, '6': n/T, '7': TS, '8': ZtoF(TS)*100.0, '9': postprob_raster,
               '10': std_raster, '11': training_points_feature}

        messages.addMessage(Text)

        if output_txt_file:
            file = open(output_txt_file, "w")
            file.write(Text)
            messages.addMessage("Text File saved: %s" % output_txt_file)

    except arcpy.ExecuteError:
        arcpy.AddError(arcpy.GetMessages(2))
    except Exception:
        tb = sys.exc_info()[2]
        tbinfo = traceback.format_tb(tb)[0]
        
        pymsg = f"PYTHON ERRORS:\nTraceback Info:\n{tbinfo}\nError Info:\n{sys.exc_info()}\n"
        msgs = f"GP ERRORS:\n{arcpy.GetMessages(2)}\n"

        arcpy.AddError(msgs)
        arcpy.AddError(pymsg)


def ZtoF(Z):
    """
    Conversion from a normal probability density function to a normal probability function.

    Based on ZToF (source unknown), a 'subroutine to compute frequency from z'. 
    The original code references eq. 26.2.17 in Handbook Of Mathematical Functions
    (edited by Abramowitz & Stegun, 1964). This equation is part of the polynomial and rational
    approximations for P(x) and Z(x), where P(x) is the normal probability function
    and Z(x) is the normal probability density function.
    """
    CONST6 = 0.2316419
    B1 = 0.31938153
    B2 = -0.356563782
    B3 = 1.781477937
    B4 = -1.821255978
    B5 = 1.330274429

    Z = float(Z)
    X = Z

    if X < 0.0:
        X = -Z

    t = 1.0 / (CONST6 * X + 1.0)

    PID = 2.0 * math.pi

    XX = -X * X / 2.0
    XX = math.exp(XX) / math.sqrt(PID)

    PZ = (B1 * t) + (B2 * t * t) + (B3 * (t ** 3)) + (B4 * (t ** 4)) + (B5 * (t ** 5))
    PZ = 1.0 - (PZ * XX)

    if Z < 0:
        PZ = 1.0 - PZ

    return PZ

