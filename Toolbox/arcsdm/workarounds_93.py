""" ArcSDM 6 ToolBox for ArcGIS Pro

Conversion and tool development for ArcGIS Pro by Geological Survey of Finland (GTK), 2024.

- sys.exc_type and exc_value are deprecated, replaced by sys.exc_info()

Spatial Data Modeller for ESRI* ArcGIS 9.3
Copyright 2009
Gary L Raines, Reno, NV, USA: production and certification
Don L Sawatzky, Spokane, WA, USA: Python software development
    
Work-arounds for 9.3 version of ArcGIS 
"""
import sys, os, traceback, arcpy

def rowgen(searchcursor):
    """ 
    Wrapper for searchcursor to permit its use in Python for statement.
    This function yields rows from the search cursor one by one.
    """
    with searchcursor as rows:
        for row in rows:
            yield row

def GetIDField(table):
    """
    Returns the ID field name of the given table.
    Checks for the presence of 'FID' or 'OBJECTID' fields.
    
    Args:
        table (str): The table to check for ID fields.
    
    Returns:
        str: The name of the ID field.
    """
    field_names = [field.name for field in arcpy.ListFields(table)]
    if "FID" in field_names:
        return "FID"
    else:
        return "OBJECTID" 
    
def ExtractValuesToPoints(inputRaster, inputFeatures, siteFIDName):
    """
    Extracts values to points for selected features in ArcGIS 9.3.
    This routine generates the extracted feature class from only selected input features.
    If the selected subset is very large, a very large query string will be created in memory.
    
    Args:
        inputRaster (str): The input raster dataset.
        inputFeatures (str): The input feature class.
        siteFIDName (str): The field name to store the FID values.
    
    Returns:
        str: The path to the temporary extracted shapefile.
    """
    try:
        assert siteFIDName in ('TPFID','NDTPFID')
        if siteFIDName not in [field.name for field in arcpy.ListFields(inputFeatures)]:
            arcpy.AddField_management(inputFeatures, siteFIDName, 'LONG')            

        idfield = GetIDField(inputFeatures)
        arcpy.CalculateField_management(inputFeatures, siteFIDName, "!{}!".format(idfield), "PYTHON_9.3")

        tempExtrShp = arcpy.CreateScratchName('Extr', 'Tmp', 'shapefile', arcpy.env.scratchWorkspace)
        arcpy.sa.ExtractValuesToPoints(inputFeatures, inputRaster, tempExtrShp)
        return tempExtrShp
    except:
        tb = sys.exc_info()[2]
        tbinfo = traceback.format_tb(tb)[0]
        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
            str(sys.exc_info()) + "\n"
        msgs = "GP ERRORS:\n" + arcpy.GetMessages(2) + "\n"
        arcpy.AddError(msgs)
        arcpy.AddError(pymsg)
        print(pymsg)
        print(msgs)

def BandCollectionStats(inputRasters, BandCol_Stats, state="BRIEF"):
    """
    Computes band collection statistics for input rasters.
    This routine creates masked copies of all input rasters, temporarily turns off
    the environment mask, and runs the Band Collection Statistics tool on them.
    
    Args:
        inputRasters (list): List of input raster datasets.
        BandCol_Stats (str): The output band collection statistics file.
        state (str): The state of the statistics, default is "BRIEF".
    """
    try:
        saved_mask = arcpy.env.mask
        arcpy.env.mask = ''
        arcpy.sa.BandCollectionStats(inputRasters, BandCol_Stats, state)
        arcpy.env.mask = saved_mask
    except:
        msgs = arcpy.GetMessages(0)
        msgs += arcpy.GetMessages(2)
        arcpy.AddError(msgs)
        raise

__all__ = ["ExtractValuesToPoints","BandCollectionStats"]

def testBCS():
    """
    Test function for BandCollectionStats.
    Gets input parameters from the user, runs BandCollectionStats, and sets the output parameter.
    """
    inputrasters = arcpy.GetParameterAsText(0)
    outputfile = arcpy.GetParameterAsText(1)
    BandCollectionStats(inputrasters, outputfile)
    arcpy.SetParameterAsText(1, outputfile)
    
if __name__ == '__main__':
    testBCS()
