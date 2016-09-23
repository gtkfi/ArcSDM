"""A function to append Spatial Data Modeller parameters to Geoprocessor History
    for those SDM tools that have the following values:
    gp: geoprocessor object
    unitCell: unit cell area in sq km
    TrainPts: training sites Points feature class
"""
import traceback, sys, arcsdm.exceptions;
import arcpy;

ToMetric = {
    'square meters to square kilometers' : 0.000001,
    'square feet to square kilometers' : 0.09290304 * 1e-6,
    'square inches to square kilometers' : 0.00064516 * 1e-6,
    'square miles to square kilometers' : 2.589988110647
    }
    
    
    
#Return mask size in square km
def getMaskSize ():
    desc = arcpy.Describe(arcpy.env.mask);
    #arcpy.AddMessage( "getMaskSize()");
    if (desc.dataType == "RasterLayer"):
        #arcpy.AddMessage( " Counting raster size");                       
        maskrows = gp.SearchCursor(desc.catalogpath)        
        maskrow = maskrows.next()
        count =  0
        while maskrow:
            count += maskrow.count
            maskrow = maskrows.next()
    if (desc.dataType == "FeatureLayer"):
        #arcpy.AddMessage( " Calculating mask size");           
        maskrows = arcpy.SearchCursor(desc.catalogpath)
        shapeName = desc.shapeFieldName                
        maskrow = maskrows.next()
        count =  0
        while maskrow:
            feat = maskrow.getValue(shapeName)
            count += feat.area;
            maskrow = maskrows.next()
       
        mapUnits = getMapUnits().lower().strip()
        if not mapUnits.startswith('meter'):
                arcpy.addError('Incorrect output map units: Check units of study area.')
        conversion = getMapConversion( mapUnits)
        count = count * conversion;
    
    return count 

    
    
    
def appendSDMValues(gp, unitCell, TrainPts):
    try:
        arcpy.AddMessage("\n" + "="*30 + " arcsdm values  " + "=" *30);
        if not gp.workspace:
            gp.adderror('Workspace not set')
        gp.addmessage("Workspace: %s"%gp.workspace)
        if not gp.scratchworkspace:
            gp.adderror('Scratch workspace mask not set')
        gp.addmessage("Scratch workspace: %s"%gp.scratchworkspace)
        # TODO: These should be moved to common CHECKENV class/function TR
        if not gp.mask:
            gp.adderror('Study Area mask not set');
            raise arcpy.ExecuteError;
        else:
            #gp.AddMessage("Mask set");
            desc = gp.describe(gp.mask);
            gp.addMessage( "Mask name is \"" + desc.name + "\" and it is " + desc.dataType);           
            #gp.AddMessage("Masksize: " + str(getMaskSize()));            
        mapUnits = getMapUnits().lower().strip()
        if not mapUnits.startswith('meter'):
            gp.addError('Incorrect output map units: Check units of study area.')            
        conversion = getMapConversion(mapUnits)                
        #gp.addMessage("Conversion from map units to km^2: " + str(conversion));

        if not gp.cellsize:        
            gp.adderror('Study Area cellsize not set')
        if (gp.cellsize == "MAXOF"):
            gp.AddError("Cellsize must have value!");
            raise arcpy.ExecuteError
            #raise arcsdm.exceptions.SDMError ("");
        
        cellsize = float(gp.cellsize)
        gp.addmessage('Cell Size: %s'%cellsize)
        #gp.addMessage("Debug: " + str(conversion));
        total_area = getMaskSize() * cellsize **2 * conversion
        #gp.addMessage("Debug));
        
        #gp.addMessage("Debug: Total_area=" + str(total_area));
        #gp.addMessage("Debug: Unitcell=" + str((unitCell)));
        unitCell = float(unitCell)#.replace(",", ".")); # Python: "Commas, gtfo"
        num_unit_cells = total_area / unitCell
        num_tps = gp.GetCount_management(TrainPts)
        #gp.AddMessage("Debug: num_tps = {} num_unit_cells = {}".format(num_tps, num_unit_cells));
        gp.addmessage('Number of Training Sites: %s' %num_tps)
        gp.addmessage('Unit Cell Area (sq km): {}  Cells in area: {} '.format(unitCell,num_unit_cells))
        
        priorprob = num_tps / num_unit_cells
        if not (0 < priorprob <= 1.0):
            gp.adderror('Incorrect no. of training sites or unit cell area. TrainingPointsResult {}'.format(priorprob));
            raise arcpy.ExecuteError;
        gp.addmessage('Prior Probability: %0.6f' %priorprob)
        gp.addmessage('Training Set: %s'%gp.describe(TrainPts).catalogpath)
        gp.addmessage('Study Area Raster: %s'%gp.describe(gp.mask).catalogpath)
        gp.addmessage('Study Area Area (sq km): %s'%total_area)
        gp.addmessage('Map Units: %s'%mapUnits)
        #gp.addmessage('Map Units to Square Kilometers Conversion: %f'%conversion)
       
    except arcpy.ExecuteError as e:
        #TODO: Clean up all these execute errors in final version
        arcpy.AddError("\n");
        if not all(e.args):
            arcpy.AddMessage("Calculate weights caught arcpy.ExecuteError: ");
            args = e.args[0];
            args.split('\n')
            arcpy.AddError(args);
                    
        arcpy.AddMessage("-------------- END EXECUTION ---------------");        
        raise
  
    except:
        # get the traceback object
        tb = sys.exc_info()[2]
        #gp.addError("sdmvalues.py excepted:");
        # tbinfo contains the line number that the code failed on and the code from that line
        tbinfo = traceback.format_tb(tb)[0]
        gp.addError ( tbinfo );
        # concatenate information together concerning the error into a message string
        #pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
        #    str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
        # generate a message string for any geoprocessing tool errors
        if len(gp.GetMessages(2)) > 0:
            msgs = "SDM GP ERRORS:\n" + gp.GetMessages(2) + "\n"
            gp.AddError(msgs)

        # return gp messages for use with a script tool
        #gp.AddError(pymsg)

        raise

def getMapConversion(mapUnits):
    pluralMapUnits = {'meter':'meters', 'foot':'feet', 'inch':'inches', 'mile':'miles'}
    conversion = ToMetric["square %s to square kilometers"%pluralMapUnits[mapUnits]]
    return conversion    

def getMapUnits():
    """ Get document map units from g.outputcoordinatesystem """
    try:
        #Get spatial reference of geoprocessor
        ocs = arcpy.env.outputCoordinateSystem
        if not ocs:
            arcpy.adderror('Output Coordinate System not set')
            raise arcpy.ExecuteError
        #else:
        #arcpy.AddMessage("Outputcoordinate system ok");
        ##Replace apostrophes with quotations
        #ocs = ocs.replace("'",'"')
        ##Open scratch file for output
        #prjfile = arcpy.createuniquename('coordsys', gp.scratchFolder) + '.prj'
        ##Write spatial reference string to scratch file
        #fdout = open(prjfile,'w')
        #fdout.write(ocs)
        #fdout.write('\n')
        #fdout.close()
        ##Create spatial reference object
        #spatref = gp.createobject('spatialreference')
        #Populate it by parsing of scratch file
        #spatref.createfromfile(prjfile)
        #Return map units value
        #spatial_ref = arcpy.Describe(dataset).spatialReference
        if ocs.type == 'Projected':
            #arcpy.AddMessage("Projected system");
            return ocs.linearUnitName
            
        elif ocs.type == 'Geographic':
            #arcpy.AddMessage("Geographics system");
            return ocs.angularUnitName
        else:
            return None        
    except arcpy.ExecuteError as error:
        #gp.AddError(gp.GetMessages(2))
        #gp.AddMessage("Debug SDMVAlues exception");
        raise
        #pass;
    except:
        import traceback, sys
        # get the traceback object
        tb = sys.exc_info()[2]
        # tbinfo contains the line number that the code failed on and the code from that line
        tbinfo = traceback.format_tb(tb)[0]
        # concatenate information together concerning the error into a message string
        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
            str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
        # generate a message string for any geoprocessing tool errors
        msgs = "SDMVALUES + GP ERRORS:\n" + arcpy.GetMessages(2) + "\n"
        arcpy.AddError(msgs)

        # return gp messages for use with a script tool
        if (len(arcpy.GetMessages(2)) < 1):
            arcpy.AddError(pymsg)
            print (pymsg)
        
        

        # print messages for use in Python/PythonWin
        print (msgs)
        raise

if __name__ == '__main__':
    import arcgisscripting
    gp = arcgisscripting.create()
    training_sites = gp.getParameterAsText(0)
    unit_area = gp.getparameter(1)
    appendSDMValues(gp, unit_area, training_sites)
    
