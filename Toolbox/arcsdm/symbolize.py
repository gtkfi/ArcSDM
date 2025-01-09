'''
    ArcSDM 5 
    Tero Ronkko / GTK 2016
    This can be used to symbolize raster layer with proper classes 
    TODO: Needs input from science group!
'''

#TODO: ArcGis pro compatibility!

import arcpy
import os
import sys
import traceback
from arcsdm.exceptions import SDMError


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
globalmasksize = -1

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


def getPriorProb(TrainPts, unitCell, mapUnits):
    """
    Calculate the prior probability against mask/training points.
    """
    num_tps = arcpy.GetCount_management(TrainPts)
    total_area = getMaskSize(mapUnits)  # Now the getMaskSize returns it correctly in sqkm
    unitCell = float(unitCell)
    total_area = float(total_area)
    num_unit_cells = total_area / unitCell
    num_tps = int(num_tps.getOutput(0))
    priorprob = num_tps / num_unit_cells
    return priorprob


def getMaskSize(mapUnits):
    """
    Return the mask size in square kilometers.
    """
    try:
        global globalmasksize
        if globalmasksize > 0:
            return globalmasksize
        desc = arcpy.Describe(arcpy.env.mask)

        if desc.dataType in ["RasterLayer", "RasterDataset", "RasterBand"]:
            if not str(arcpy.env.cellSize).replace('.', '', 1).replace(',', '', 1).isdigit():
                arcpy.AddMessage("*" * 50)
                arcpy.AddError("ERROR: Cell Size must be numeric when mask is raster. Check Environments!")
                arcpy.AddMessage("*" * 50)
                raise SDMError

            dwrite("Counting raster size")
            dwrite(f"File: {desc.catalogPath}")
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
            raise arcpy.ExecuteError(f"{desc.dataType} is not allowed as Mask!")

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
            msgs = f"SDM GP ERRORS:\n{arcpy.GetMessages(2)}\n"
            arcpy.AddError(msgs)
        raise


def getMapConversion(mapUnits):
    """
    Get the conversion factor from the map units to square kilometers.
    """
    pluralMapUnits = {'meter': 'meters', 'foot': 'feet', 'inch': 'inches', 'mile': 'miles'}
    conversion = ToMetric["square %s to square kilometers" % pluralMapUnits[mapUnits]]
    return conversion


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
        if ocs.type == "Projected":
            return ocs.linearUnitName
        elif ocs.type == "Geographic":
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


def execute(self, parameters, messages):
    rastername = parameters[0].valueAsText
    trainpts = parameters[1].valueAsText
    unitcell = parameters[2].value
    Temp_Symbology = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..\\raster_classified.lyr')
    desc = arcpy.Describe(rastername)
    
    arcpy.AddMessage(Temp_Symbology)
    arcpy.AddMessage("=" * 20 + " starting symbolize " + "=" * 20)
    arcpy.AddMessage("%-20s %s"% ("Raster name: ", desc.file))
    rastername = desc.file
    # TODO: need to replace with arcpy.mp. See: https://pro.arcgis.com/en/pro-app/latest/arcpy/mapping/migratingfrom10xarcpymapping.htm
    mxd = arcpy.mapping.MapDocument("CURRENT")
    df = arcpy.mapping.ListDataFrames(mxd, "")[0]
    lyr = arcpy.mapping.ListLayers(mxd, rastername, df)[0]
    prob = getPriorProb(trainpts, unitcell, getMapUnits())
    arcpy.AddMessage("%-20s %s" % ("Probability: ", str(prob)))
    arcpy.AddMessage("Applying raster symbology to classified values from lyr file... ")
    arcpy.ApplySymbologyFromLayer_management(lyr, Temp_Symbology)
    
    if lyr.symbologyType == "RASTER_CLASSIFIED":
        #lyr.symbology.classBreakValues = [1, 60, 118, 165, 255]
        arcpy.AddMessage("Setting values to prob priority values ... ")
        values = lyr.symbology.classBreakValues
        values[0] = 0
        values[1] = prob #TODO: Does this need rounding?
        lyr.symbology.classBreakValues = values
        #lyr.symbology.classBreakLabels = ["1 to 60", "61 to 118", 
        #                                "119 to 165", "166 to 255"]
        #lyr.symbology.classBreakDescriptions = ["Class A", "Class B",
        #                                      "Class C", "Class D"]
        lyr.symbology.reclassify()                                          
        
        #lyr.symbology.excludedValues = '0'
    arcpy.RefreshActiveView()
    arcpy.RefreshTOC()
    del mxd, df, lyr