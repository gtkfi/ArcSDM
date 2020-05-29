"""

    Fixed for ArcSDM 5 for ArcGis pro
    history: 
    12.8.2016 started fixing TR
    5.5.2020 Arto Laiho, Geological survey of Finland: 
    - sys.exc_type and exc_value are deprecated, replaced by sys.exc_info()

    Spatial Data Modeller for ESRI* ArcGIS 9.3
    Copyright 2009
    Gary L Raines, Reno, NV, USA: production and certification
    Don L Sawatzky, Spokane, WA, USA: Python software development
    
    Work-arounds for 9.3 version of ArcGIS 
"""
import sys, os, traceback, arcpy


def rowgen( searchcursor ):
    """ wrapper for searchcursor to permit its use in Python for statement """
    rows = searchcursor
    rows.reset()
    row = rows.next()
    while row:
        yield row
        row = rows.next()
    del rows

    
    
#This function needs to be GENERAL    

def GetIDField(table):
    field_names = []
    fields = arcpy.ListFields(table,"")
    for field in fields:
        field_names.append(field.name)
    if "FID" in field_names:
        return "FID"
    else:
        return "OBJECTID" 
    
def ExtractValuesToPoints(gp, inputRaster, inputFeatures, siteFIDName):
    """ ExtractValuesToPoints tool in ArcGIS 9.3 now selects ALL features.
           This routine generates the extracted feature class from only selected input features.
           If the selected subset is very large, a very large query string will be created in memory.
    """

    try:
        #arcpy.AddMessage("Debug: workarounds93 Extracting values to points...")
        assert siteFIDName in ('TPFID','NDTPFID')
        if siteFIDName not in [field.name for field in rowgen(gp.ListFields(inputFeatures))]:
            gp.AddField_management(inputFeatures, siteFIDName, 'LONG')            
            #gp.AddMessage("Debug: workarounds93 Added new FID field");           
        #else:
            #arcpy.AddMessage("Debug: workarounds93 SiteFIDName = " + siteFIDName );
         
        #gp.CalculateField_management(inputFeatures, siteFIDName, "!FID!", "PYTHON_9.3", None)
        idfield = GetIDField(inputFeatures); 
        gp.CalculateField_management(inputFeatures, siteFIDName, "!{}!".format(idfield) , "PYTHON_9.3")

        tempExtrShp = gp.CreateScratchName ('Extr', 'Tmp', 'shapefile', gp.scratchworkspace)
        #tempSelectedExtrShp = gp.CreateScratchName ('SelExtr', 'Tmp', 'shapefile', gp.scratchworkspace)
        gp.ExtractValuesToPoints_sa(inputFeatures, inputRaster, tempExtrShp)
        #gp.addwarning(str(gp.getcount(inputFeatures)))
##        if gp.GetCount_management(inputFeatures) < gp.GetCount_management(tempExtrShp):
##            #Not going to happen in 9.2, but will in 9.3
##            rows = rowgen(gp.searchcursor(inputFeatures)) # Gets selected rows
##            FIDs = (str(row.FID) for row in rows)
##            query = '"FID" IN (' + ','.join(FIDs) + ')'
##            #gp.addwarning(query)
##            TempShp = 'tempshp'
##            gp.MakeFeatureLayer_management(tempExtrShp, TempShp)
##            gp.addwarning(str(gp.GetCount(TempShp)))
##            gp.SelectLayerByAttribute_management(tempExtrShp, "NEW_SELECTION", query)
##            #gp.addwarning(str(gp.GetCount(TempShp)))
##            return tempExtrShp
##        else:
##            return inputFeatures
        return tempExtrShp
    except:
        # get the traceback object
        tb = sys.exc_info()[2]
        # tbinfo contains the line number that the code failed on and the code from that line
        tbinfo = traceback.format_tb(tb)[0]
        # concatenate information together concerning the error into a message string
        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
            str(sys.exc_info()) + "\n"    #AL 050520
        #    str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
        # generate a message string for any geoprocessing tool errors
        msgs = "GP ERRORS:\n" + gp.GetMessages(2) + "\n"
        gp.AddError(msgs)

        # return gp messages for use with a script tool
        gp.AddError(pymsg)

        # print messages for use in Python/PythonWin
        print (pymsg)
        print (msgs)

def BandCollectionStats(gp, inputRasters, BandCol_Stats, state="BRIEF"):
    """ BandCollectionStatistics tool in ArcGIS 9.3 now does not use mask; in fact,
            the tool breaks when a mask is set in the environment.
           This routine not creates masked copies of all input rasters, temporarily turns off
           the environment mask, runs the Band Collection Statistics tool on them.
    """
    try:
##>>>>>>>>>>>>>  Code when using pre-masked data sets >>>>>>>>>>>>>
        saved_mask = gp.mask
        gp.mask = ''
        gp.BandCollectionStats_sa(inputRasters, BandCol_Stats, state)
        gp.mask = saved_mask
##>>>>>>>>>>>>>  Code when using unmasked data sets >>>>>>>>>>>>>
##        OutRasters = []
##        InMask = gp.mask
##        for InRaster in inputRasters.split(";"):
##            OutRaster = gp.CreateScratchName("BCS", "TMP", "RasterBand", gp.scratchworkspace)
##            gp.ExtractByMask_sa(InRaster, InMask, OutRaster)
##            OutRasters.append(OutRaster)
        
##        saved_mask = gp.mask
##        gp.mask = ''
##        gp.BandCollectionStats_sa(";".join(OutRasters), BandCol_Stats, state)
##        gp.mask = saved_mask
##        for OutRaster in OutRasters:
##            gp.Delete(OutRaster)
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<          
    except:
        # Get the geoprocessing error messages
        #
        msgs = gp.GetMessage(0)
        msgs += gp.GetMessages(2)

        # Return gp error messages for use with a script tool
        #
        gp.AddError(msgs)

        # Print gp error messages for use in Python/PythonWin
        #
        raise

__all__ = ["ExtractValuesToPoints","BandCollectionStats"]

def testBCS(gp):
    inputrasters = gp.GetParameterAsText(0)
    outputfile = gp.GetParameterAsText(1)
    BandCollectionStats(gp, inputrasters, outputfile)
    gp.SetParameterAsText(1, outputfile)
    
if __name__ == '__main__':
    import arcgisscripting
    gp = arcgisscripting.create()

    testBCS(gp)
