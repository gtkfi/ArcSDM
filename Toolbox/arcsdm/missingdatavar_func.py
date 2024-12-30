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


def TotalAreaFromCounts(gp, Input_Raster, CellSize):
    IsNul_Wts = gp.createuniquename(("IsNul_" + os.path.basename(Input_Raster))[:11], gp.scratchworkspace)
    gp.IsNull_sa(Input_Raster, IsNul_Wts)
    rasrows = gp.SearchCursor(IsNul_Wts, 'Value = 0')
    rasrow = rasrows.Next()
    TotalCount = rasrow.Count
    arcpy.management.Delete(arcpy.Describe(IsNul_Wts).catalogPath)
    return float(TotalCount) * CellSize * CellSize / 1000000


def MissingDataVariance(gp, Wts_Rasters, PostProb, OutputName):
    # TODO: Fix - env cell size may differ from actual cell size
    CellSize = float(gp.CellSize)

    try:
        i = 0
        
        # Create Total Missing Data Variance list
        TotClsVars = []

        # Loop throught Wts Rasters
        for Wts_Raster0 in Wts_Rasters:
            arcpy.AddMessage(f"Missing data Variance for: {Wts_Raster0}")
            Wts_Raster = arcpy.Describe(Wts_Raster0).catalogPath
            TotDataArea = TotalAreaFromCounts(gp, Wts_Raster, CellSize)
            arcpy.AddMessage('TotDataArea = %.0f' % TotDataArea)
            
            # Start MD Variance raster
            # Get PostProb raster of MD cells
            R1 = os.path.join(arcpy.env.scratchWorkspace, "R1")
            if arcpy.Exists(R1):
                arcpy.management.Delete(R1)
            Exp0 = "CON(%s == 0.0,%s,0.0)" % (Wts_Raster, PostProb)
            arcpy.AddMessage(f"R1={Exp0}")
            gp.SingleOutputMapAlgebra_sa(Exp0, R1)
            # Get PostODDs raster of MD cells
            R2 = os.path.join(arcpy.env.scratchWorkspace, "R2")
            if arcpy.Exists(R2):
                arcpy.management.Delete(R2)
            Exp = "%s / (1.0 - %s)" % (R1, R1)
            arcpy.AddMessage(f"R2={Exp}")
            gp.SingleOutputMapAlgebra_sa(Exp, R2)
            arcpy.AddMessage(f"R2 exists: {arcpy.Exists(R2)}")
            
            # Get Total Variance of MD cells
            # Create total class variances list
            ClsVars = []

            Wts_RasterRows = FloatRasterSearchcursor(Wts_Raster)
            j = 0
            """ Cannot be done by single raster; must generate a raster for each value """
            for Wts_RasterRow in Wts_RasterRows:
                if Wts_RasterRow.Value == 0.0:
                    continue
                ClsVar = str(os.path.join(arcpy.env.scratchWorkspace, "ClsVar%s%s" % (i, j)))
                j += 1
                if arcpy.Exists(ClsVar):
                    arcpy.management.Delete(ClsVar)
                Exp1 = 'CON(%s == 0.0,0.0,EXP(LN(%s) + %s))' % (R2, R2, Wts_RasterRow.Value)
                Exp2 = "%s / (1 + %s)" % (Exp1, Exp1)
                ClsArea = float(Wts_RasterRow.Count) * CellSize * CellSize / 1000000.0
                Exp3 = "SQR(%s - %s) * (%s / %s)" % (Exp2, R1, ClsArea, TotDataArea)
                gp.SingleOutputMapAlgebra_sa(Exp3, ClsVar)
                ClsVars.append(str(ClsVar)) # Save the class variance raster

            del Wts_RasterRows
            # Sum the class variances
            TotClsVar = os.path.join(arcpy.env.scratchWorkspace, "TotClsVar%s" % i)
            i += 1
            if arcpy.Exists(TotClsVar):
                arcpy.Delete(TotClsVar)
            
            Exp = "SUM%s" % str(tuple(ClsVars))
            gp.SingleOutputMapAlgebra_sa(Exp, TotClsVar)
            TotClsVars.append(str(TotClsVar))

            for tmp_raster in ClsVars:
                arcpy.management.Delete(arcpy.Describe(tmp_raster).catalogPath)
               
        # Create Total Missing Data Variance raster and list
        else:
            if len(Wts_Rasters) > 0:
                TotVarMD = OutputName
                Exp = "SUM%s" % str(tuple(TotClsVars))
                gp.SingleOutputMapAlgebra_sa(Exp, TotVarMD)

        arcpy.management.Delete(arcpy.Describe(R1).catalogPath)
        arcpy.management.Delete(arcpy.Describe(R2).catalogPath)
        
        for tmp_raster in TotClsVars:
            arcpy.management.Delete(arcpy.Describe(tmp_raster).catalogPath)

    except Exception:
        tb = sys.exc_info()[2]
        tbinfo = traceback.format_tb(tb)[0]

        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
            str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
        msgs = "GP ERRORS:\n" + gp.GetMessages(2) + "\n"

        gp.AddError(msgs)
        gp.AddError(pymsg)

