"""A function to append Spatial Data Modeller parameters to Geoprocessor History
for those SDM tools that have the following values:
gp: geoprocessor object
unitCell: unit cell area in sq km
TrainPts: training sites Points feature class
"""

import traceback, sys

# Conversion factors from various units to square kilometers
ToMetric = {
    'square meters to square kilometers' : 0.000001,
    'square feet to square kilometers' : 0.09290304 * 1e-6,
    'square inches to square kilometers' : 0.00064516 * 1e-6,
    'square miles to square kilometers' : 2.589988110647
}

def appendSDMValues(gp, unitCell, TrainPts):
    """Append Spatial Data Modeller parameters to Geoprocessor History."""
    try:
        # Check if workspace is set
        if not gp.workspace:
            gp.adderror('Workspace not set')
        gp.addmessage("Workspace: %s" % gp.workspace)

        # Check if scratch workspace is set
        if not gp.scratchworkspace:
            gp.adderror('Scratch workspace mask not set')
        gp.addmessage("Scratch workspace: %s" % gp.scratchworkspace)

        # Check if study area mask is set
        if not gp.mask:
            gp.adderror('Study Area mask not set')

        # Count the number of rows in the mask
        maskrows = gp.SearchCursor(gp.describe(gp.mask).catalogpath)
        maskrow = maskrows.next()
        count = 0
        while maskrow:
            count += 1
            maskrow = maskrows.next()
        gp.AddMessage("Maskrowcount: " + str(count))

        # Get map units and check if they are in meters
        mapUnits = getMapUnits(gp).lower().strip()
        if not mapUnits.startswith('meter'):
            gp.addError('Incorrect output map units: Check units of study area.')

        # Get conversion factor for map units to square kilometers
        conversion = getMapConversion(gp, mapUnits)

        # Check if cell size is set
        if not gp.cellsize:
            gp.adderror('Study Area cellsize not set')
        cellsize = float(gp.cellsize)

        # Calculate total area and number of unit cells
        total_area = count * cellsize ** 2 * conversion
        num_unit_cells = total_area / unitCell

        # Get number of training points
        num_tps = gp.GetCount_management(TrainPts)

        # Calculate prior probability
        priorprob = num_tps / num_unit_cells
        if not (0 < priorprob <= 1.0):
            gp.adderror('Incorrect no. of training sites or unit cell area')

        # Log messages
        gp.addmessage('Prior Probability: %0.6f' % priorprob)
        gp.addmessage('Training Set: %s' % gp.describe(TrainPts).catalogpath)
        gp.addmessage('Number of Training Sites: %s' % num_tps)
        gp.addmessage('Study Area Raster: %s' % gp.describe(gp.mask).catalogpath)
        gp.addmessage('Study Area Area (sq km): %s' % total_area)
        gp.addmessage('Unit Cell Area (sq km): %s' % unitCell)
        gp.addmessage('Map Units: %s' % mapUnits)

    except:
        # Handle exceptions and log error messages
        tb = sys.exc_info()[2]
        tbinfo = traceback.format_tb(tb)[0]
        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
            str(sys.exc_type) + ": " + str(sys.exc_value) + "\n"
        if len(gp.GetMessages(2)) > 0:
            msgs = "GP ERRORS:\n" + gp.GetMessages(2) + "\n"
            gp.AddError(msgs)
        gp.AddError(pymsg)
        raise

def getMapConversion(gp, mapUnits):
    """Get conversion factor from map units to square kilometers."""
    pluralMapUnits = {'meter': 'meters', 'foot': 'feet', 'inch': 'inches', 'mile': 'miles'}
    conversion = ToMetric["square %s to square kilometers" % pluralMapUnits[mapUnits]]
    return conversion    

def getMapUnits(gp):
    """Get document map units from g.outputcoordinatesystem."""
    try:
        ocs = gp.outputcoordinatesystem
        if not ocs:
            gp.adderror('Output Coordinate System not set')
            raise Exception
        ocs = ocs.replace("'", '"')
        prjfile = gp.createuniquename('coordsys', gp.scratchworkspace) + '.prj'
        with open(prjfile, 'w') as fdout:
            fdout.write(ocs)
            fdout.write('\n')
        spatref = gp.createobject('spatialreference')
        spatref.createfromfile(prjfile)
        if spatref.type == 'Projected':
            return spatref.linearunitname
        elif spatref.type == 'Geographic':
            return spatref.angularunitname
        else:
            return None        
    except:
        tb = sys.exc_info()[2]
        tbinfo = traceback.format_tb(tb)[0]
        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
            str(sys.exc_type) + ": " + str(sys.exc_value) + "\n"
        msgs = "GP ERRORS:\n" + gp.GetMessages(2) + "\n"
        gp.AddError(msgs)
        gp.AddError(pymsg)
        print(pymsg)
        print(msgs)

if __name__ == '__main__':
    import arcpy
    gp = arcpy.create()
    training_sites = gp.getParameterAsText(0)
    unit_area = gp.getparameter(1)
    appendSDMValues(gp, unit_area, training_sites)
