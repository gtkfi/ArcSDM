# -*- coding: utf-8 -*-
""" ArcSDM 6 ToolBox for ArcGIS Pro

Conversion and tool development for ArcGIS Pro by Geological Survey of Finland (GTK), 2024.

Spatial Data Modeller for ESRI* ArcGIS 9.2

Copyright 2007
Gary L Raines, Reno, NV, USA: production and certification
Don L Sawatzky, Spokane, WA, USA: Python software development

Converted to ArcSDM toolbox to ArcMap by GTK 2016
"""
import sys, os, math, traceback, math
import arcpy
import arcpy.management

from arcsdm.floatingrasterarray import FloatRasterSearchcursor

def Calculate(self, parameters, messages):
    messages.addMessage("Starting Agterberg-Cheng CI Test")
    try:
        arcpy.CheckOutExtension("spatial")
        arcpy.env.overwriteOutput = True

        PostProb =  parameters[0].valueAsText
        PPStd =  parameters[1].valueAsText
        TrainSites =  parameters[2].valueAsText
        UnitArea =  parameters[3].value
        SaveFile =  parameters[4].valueAsText

        postprob_raster_path = arcpy.Describe(PostProb).catalogPath
        std_raster_path = arcpy.Describe(PPStd).catalogPath

        basename = os.path.basename(PostProb)
        sdmuc = basename.split("_")[0]
        CellSize = float(arcpy.env.cellSize)
        #ExpNumTP = arcpy.GetCount_management(TrainSites) #Num of Selected sites
        result = arcpy.management.GetCount(TrainSites)
        ExpNumTP = int(result.getOutput(0))
        ConvFac = (CellSize ** 2) / 1000000.0 / UnitArea
        PredT = 0.0

        postprob_raster = FloatRasterSearchcursor(postprob_raster_path)
        std_raster = FloatRasterSearchcursor(std_raster_path)

        for postprob_value in postprob_raster:
            PredT += (postprob_value.Value * postprob_value.Count)
        PredT *= ConvFac

        TVar = 0.0

        for std_value in std_raster:
            TVar += (std_value.Value * std_value.Count * ConvFac) ** 2
        TStd = math.sqrt(TVar)
        TS = (PredT - ExpNumTP) / TStd
        #PostProb
        n = ExpNumTP
        T = PredT
        #STD = TStd
        P = ZtoF(TS) * 100.0
        if P >= 50.0: overallCI = 100.0 * (100.0 - P) / 50.0
        else: overallCI = 100.0 * (100.0 - (50 + (50 - P))) / 50.0

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
        """ % {'0': overallCI, '1': sdmuc, '2': n, '3': T, '4': T-n, '5': TStd, '6': n/T, '7': TS, '8': ZtoF(TS)*100.0, '9': PostProb,
               '10': PPStd, '11': TrainSites}

        messages.addMessage(Text)

        if SaveFile:
            file = open(SaveFile, "w")
            file.write(Text)
            messages.addMessage("Text File saved: %s" % SaveFile)

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
##Function ZToF(Z As Double) As Double
##'      SUBROUTINE ZTOF(Z, PZ)
##'c
##'C ... SUBROUTINE TO COMPUTE FREQUENCY FROM Z.   EQ. 26.2.17   IN
##'c
##'C      ABRAMOWITZ, M. AND STEGUN, I.A.     (EDITORS)
##'c "HANDBOOK OF MATHEMATICAL FUNCTIONS"
##'C         WITH FORMULAS, GRAPHS, AND MATHEMATICAL TABLES"
##'C      PUB. BY NATIONAL BUREAU OF STANDARDS OF THE
##'C         U.S. DEPARTMENT OF COMMERCE,   1964.
##'c
##'C  CALLED BY SUBROUTINES  COMP, REORD, & WDIST
##'c
##'      DATA  PI/3.141592654/, CONST6/0.2316419/, B1/0.319381530/,
##'     +      B2/-0.356563782/, B3/1.781477937/, B4/-1.821255978/,
##'     +      B5/1.330274429/
    PI = 3.141592654
    CONST6 = 0.2316419
    B1 = 0.31938153
    B2 = -0.356563782
    B3 = 1.781477937
    B4 = -1.821255978
    B5 = 1.330274429
##'c
##'      x = Z
    Z = float(Z)
    X = Z
##'      IF (Z.LT.0.0) X = -Z
    if X < 0.0 : X = -Z
##'      t = 1# / (CONST6 * x + 1#)
    t = 1.0 / (CONST6 * X + 1.0)
##'      PID = 2# * PI
    PID = 2.0 * PI
##'      XX = -x * x / 2#
    XX = -X * X / 2.0
##'      XX = Exp(XX) / SQRT(PID)
    XX = math.exp(XX) / math.sqrt(PID)
##'      PZ = (B1 * T) + (B2 * T * T) + (B3 * T**3) + (B4 * T**4) +
##'     +     (B5 * T**5)
    PZ = (B1 * t) + (B2 * t * t) + (B3 * (t ** 3)) + (B4 * (t ** 4)) + (B5 * (t ** 5))
##'      PZ = 1# - (PZ * XX)
    PZ = 1.0 - (PZ * XX)
##'      IF (Z.LT.0.0) PZ = 1.0 - PZ
    if Z < 0:
        PZ = 1.0 - PZ
    return PZ
##'      Return
##'      End
