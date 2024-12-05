""" SDM Values / ArcSDM 6 ToolBox for ArcGIS Pro

Conversion and tool development for ArcGIS Pro by Geological Survey of Finland (GTK), 2024.

History: 
27.9.2016 Output cleaned and getMaskSize fixed
21.4-15.5.2020 added cell size numeric value check, RasterBand and some value checks / Arto Laiho, Geological survey of Finland
21.7.2020 combined with Unicamp fixes (made 24.10.2018) / Arto Laiho, GTK/GFS

A function to append Spatial Data Modeller parameters to Geoprocessor History
for those SDM tools that have the following values:
gp: geoprocessor object
unitCell: unit cell area in sq km
TrainPts: training sites Points feature class
"""

import traceback, sys
from arcsdm.exceptions import SDMError
import arcpy
import os
import numpy

# Conversion factors from various units to square kilometers
ToMetric = {
    'square meters to square kilometers': 0.000001,
    'square feet to square kilometers': 0.09290304 * 1e-6,
    'square inches to square kilometers': 0.00064516 * 1e-6,
    'square miles to square kilometers': 2.589988110647
}

# Debug level for logging
debuglevel = 0

# Global variable to store mask size, initially set to -1
# globalmasksize = -1

def testdebugfile():
    """
    Check if debugging is enabled by looking for a DEBUG file in the script directory.
    Returns 1 if debugging is enabled, otherwise returns 0.
    """
    returnvalue = 0  # This because python sucks in detecting outputs from functions
    import sys
    import os
    if debuglevel > 0:
        return 1
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if os.path.isfile(dir_path + "/DEBUG"):
        return 1
    return returnvalue

def dwrite(message):
    """
    Write a debug message if debugging is enabled.
    """
    debug = testdebugfile()
    if debuglevel > 0 or debug > 0:
        arcpy.AddMessage("Debug: " + message)

def execute(self, parameters, messages):
    """
    Obsolete function to execute the tool. Needs refactoring.
    """
    dwrite("Starting sdmvalues")
    TrainingSites = parameters[0].valueAsText
    Unitarea = float(parameters[1].value)
    appendSDMValues(Unitarea, TrainingSites)
    arcpy.AddMessage("\n" + "=" * 40)
    arcpy.AddMessage("\n")

# Old
def getPriorProb(TrainPts, unitCell, mapUnits):
    """
    Calculate the prior probability against mask/training points.
    """
    size = getMaskSize
    num_tps = arcpy.GetCount_management(TrainPts)
    total_area = getMaskSize(mapUnits)  # Now the getMaskSize returns it correctly in sqkm
    unitCell = float(unitCell)
    total_area = float(total_area)
    num_unit_cells = total_area / unitCell
    num_tps = count = int(num_tps.getOutput(0))
    priorprob = num_tps / num_unit_cells
    return priorprob


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


# Old
def getMaskSize(mapUnits):
    """
    Return the mask size in square kilometers.
    """
    try:
        global globalmasksize
        if globalmasksize > 0:
            return globalmasksize
        desc = arcpy.Describe(arcpy.env.mask)

        if desc.dataType == "RasterLayer" or desc.dataType == "RasterBand":
            if not str(arcpy.env.cellSize).replace('.', '', 1).replace(',', '', 1).isdigit():
                arcpy.AddMessage("*" * 50)
                arcpy.AddError("ERROR: Cell Size must be numeric when mask is raster. Check Environments!")
                arcpy.AddMessage("*" * 50)
                raise SDMError

            dwrite("Counting raster size")
            dwrite("File: " + desc.catalogPath)
            rows = int(arcpy.GetRasterProperties_management(desc.catalogPath, "ROWCOUNT").getOutput(0))
            columns = int(arcpy.GetRasterProperties_management(desc.catalogPath, "COLUMNCOUNT").getOutput(0))
            raster_array = arcpy.RasterToNumPyArray(desc.catalogPath, nodata_to_value=-9999)
            count = 0
            dwrite("Iterating through mask in numpy..." + str(columns) + "x" + str(rows))
            for i in range(rows):
                for j in range(columns):
                    if raster_array[i][j] != -9999:
                        count += 1
            dwrite("count:" + str(count))
            cellsize = float(str(arcpy.env.cellSize).replace(",", "."))
            count = count * (cellsize * cellsize)

        elif desc.dataType in ["FeatureLayer", "FeatureClass", "ShapeFile"]:
            maskrows = arcpy.SearchCursor(desc.catalogPath)
            shapeName = desc.shapeFieldName
            maskrow = maskrows.next()
            count = 0
            while maskrow:
                feat = maskrow.getValue(shapeName)
                count += feat.area
                maskrow = maskrows.next()
            dwrite("count:" + str(count))

        else:
            raise arcpy.ExecuteError(desc.dataType + " is not allowed as Mask!")

        mapUnits = mapUnits.lower().strip()
        if not mapUnits.startswith('meter'):
            arcpy.AddError('Incorrect output map units: Check units of study area.')
        conversion = getMapConversion(mapUnits)
        count = count * conversion
        globalmasksize = count
        return count
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

        mapUnits = getMapUnits()
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

# Old
def appendSDMValues(unitCell, TrainPts):
    """
    Append Spatial Data Modeller parameters to Geoprocessor History.
    """
    try:
        arcpy.AddMessage("\n" + "=" * 10 + " arcsdm values  " + "=" * 10)
        with open(os.path.join(os.path.dirname(__file__), "arcsdm_version.txt"), "r") as myfile:
            data = myfile.readlines()
        arcpy.AddMessage("%-20s %s" % ("", data[0]))
        installinfo = arcpy.GetInstallInfo()
        arcpy.AddMessage("%-20s %s (%s)" % ("Arcgis environment: ", installinfo['ProductName'], installinfo['Version']))

        if not arcpy.env.workspace:
            arcpy.AddError('Workspace not set')
            raise arcpy.ExecuteError("Workspace not set!")
        if not arcpy.Exists(arcpy.env.workspace):
            arcpy.AddError('Workspace %s not found' % (arcpy.env.workspace))
            raise arcpy.ExecuteError('Workspace %s not found' % (arcpy.env.workspace))
        desc = arcpy.Describe(arcpy.env.workspace)
        arcpy.AddMessage("%-20s %s (%s)" % ("Workspace: ", arcpy.env.workspace, desc.workspaceType))

        if not arcpy.env.scratchWorkspace:
            arcpy.AddError('Scratch workspace mask not set')
        wdesc = arcpy.Describe(arcpy.env.scratchWorkspace)
        arcpy.AddMessage("%-20s %s (%s)" % ("Scratch workspace:", arcpy.env.scratchWorkspace, wdesc.workspaceType))

        if wdesc.workspaceType != desc.workspaceType:
            arcpy.AddError("Workspace and scratch workspace must be of the same type!")
            raise arcpy.ExecuteError("Workspace type mismatch")

        mapUnits = getMapUnits()
        mapUnits = mapUnits.lower().strip()
        if not mapUnits.startswith('meter'):
            arcpy.AddError('Incorrect output map units: Check units of study area.')
        conversion = getMapConversion(mapUnits)
        arcpy.AddMessage("%-20s %s" % ('Map Units:', mapUnits))

        total_area = getMaskSize(mapUnits)

        if not arcpy.env.mask:
            arcpy.AddError('Study Area mask not set')
            raise arcpy.ExecuteError("Mask not set. Check Environments!")
        else:
            if not arcpy.Exists(arcpy.env.mask):
                arcpy.AddError("Mask " + arcpy.env.mask + " not found!")
                raise arcpy.ExecuteError("Mask not found")
            desc = arcpy.Describe(arcpy.env.mask)
            arcpy.AddMessage("%-20s %s" % ("Mask:", "\"" + desc.name + "\" and it is " + desc.dataType))
            if desc.dataType in ["FeatureLayer", "FeatureClass"]:
                arcpy.AddWarning('Warning: You should only use single value raster type masks!')
            arcpy.AddMessage("%-20s %s" % ("Mask size:", str(total_area)))

        if not arcpy.env.cellSize:
            arcpy.AddError('Study Area cellsize not set')
        if arcpy.env.cellSize == "MAXOF":
            arcpy.AddWarning("Cellsize should have definitive value?")

        cellsize = arcpy.env.cellSize
        arcpy.AddMessage("%-20s %s" % ("Cell Size:", cellsize))
        
        unitCell = float(unitCell)
        num_unit_cells = float(total_area) / unitCell
        num_tps = arcpy.management.GetCount(TrainPts)
        arcpy.AddMessage("%-20s %s" % ('# Training Sites:', num_tps))
        arcpy.AddMessage("%-20s %s" % ("Unit Cell Area:", "{}km^2, Cells in area: {} ".format(unitCell, num_unit_cells)))

        if num_unit_cells == 0:
            raise arcpy.ExecuteError("ERROR: 0 Cells in Area!")
        priorprob = float(str(num_tps)) / float(num_unit_cells)
        if not (0 < priorprob <= 1.0):
            arcpy.AddError('Incorrect no. of training sites or unit cell area. TrainingPointsResult {}'.format(priorprob))
            raise arcpy.ExecuteError
        arcpy.AddMessage("%-20s %0.6f" % ('Prior Probability:', priorprob))

        arcpy.AddMessage("%-20s %s" % ('Training Set:', arcpy.Describe(TrainPts).catalogPath))
        arcpy.AddMessage("%-20s %s" % ('Study Area Raster:', arcpy.Describe(arcpy.env.mask).catalogPath))
        arcpy.AddMessage("%-20s %s" % ('Study Area Area:', str(total_area) + "km^2"))
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

# Old
def getMapConversion(mapUnits):
    """
    Get the conversion factor from the map units to square kilometers.
    """
    pluralMapUnits = {'meter': 'meters', 'foot': 'feet', 'inch': 'inches', 'mile': 'miles'}
    conversion = ToMetric["square %s to square kilometers" % pluralMapUnits[mapUnits]]
    return conversion

# Old
def getMapUnits(silent=False):
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
