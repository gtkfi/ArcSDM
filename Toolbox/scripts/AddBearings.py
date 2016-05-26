"""
    ArcMPM Project recode for ArcGis Pro
    2016 Tero Rönkkö / GTK
    


    Add fields of azimuths and bearings to a featurelayer of lines.
    Azimuths can be based on map or geodgraphic coordinates.
    For segmented lines, azimuths are length-weighted averages of segments.
"""
# Import system modules
import sys, string, os, math, random, traceback, tempfile

# Create the Geoprocessor object
#gp = win32com.client.Dispatch("esriGeoprocessing.GpDispatch.1")
import arcgisscripting
gp = arcgisscripting.create()

### Check out any necessary licenses
##gp.CheckOutExtension("spatial")
##
gp.OverwriteOutput = 1

# Load required toolboxes...
#gp.AddToolbox("C:/Program Files/ArcGIS/ArcToolbox/Toolboxes/Data Management Tools.tbx")

# Script arguments...
try:
    #Get bearing type
    geodesic = True #gp.GetParameterAsText(1) == 'true'
    #Get evidence feature layer name
    out_feat_class = gp.GetParameterAsText(0)
    #gp.AddMessage(gp.describe(out_feat_class).shapetype)
    if gp.describe(out_feat_class).shapetype != "Polyline":
        raise Exception ('not a polyline-type feature class')
    #Check for Azimuth field
    fieldnames = []
    fields = gp.ListFields(out_feat_class)
    field = fields.next()
    while field:
        fieldnames.append(field.name)
        field = fields.next()
    if 'SDMBearing' not in fieldnames:
        gp.addfield_management(out_feat_class,'SDMBearing','long',5)
    if 'SDMAzimuth' not in fieldnames:
        gp.addfield_management(out_feat_class,'SDMAzimuth','long',5)

    #Open attribute table for update featureclass
    outrows = gp.updatecursor(out_feat_class)

#Define the bearing, azimuth algorithm
    def geodesic_azimuth(inshape):
        """ Map azimuth calculation from a polyline shape """
        sumazmlen = 0
        sumbrglen = 0
        sumlen = 0
        line = inshape.getpart(0)
        pnt = line.next()
        if pnt:
            pnt0 = pnt
            pnt = line.next()
            while pnt:
                delx = pnt.X-pnt0.X
                dely = pnt.Y-pnt0.Y
                length = math.hypot(delx, dely)
                #Transform math angle to azimuth angle
                sdmazimuth = int(90.0-math.degrees(math.atan2(dely,delx)))
                if sdmazimuth < 0: sdmazimuth += 360.0
                #Transform azimuth to bearing angle
                if 0 <= sdmazimuth <= 90:
                    sdmbearing = sdmazimuth
                elif 90 < sdmazimuth <= 270:
                    sdmbearing = sdmazimuth -180
                else: sdmbearing = sdmazimuth - 360
                #gp.AddMessage(str((sdmbearing,sdmazimuth)))
                sumbrglen += sdmbearing*length
                sumazmlen += sdmazimuth*length
                sumlen += length
                pnt0 = pnt
                pnt = line.next()
        return sumbrglen/sumlen,sumazmlen/sumlen

#Process the lines
    outrow = outrows.next()
    while outrow:
        if geodesic:
            outshape = outrow.shape
            sdmbearing,sdmazimuth = geodesic_azimuth(outshape)
            #gp.AddMessage('!: %s,%s'%(sdmbearing,sdmazimuth))
            outrow.sdmbearing = sdmbearing
            outrow.sdmazimuth = sdmazimuth
        else:
            pass #map_azimuth(inrow.shape)
        outrows.updaterow(outrow)
        outrow = outrows.next()
        #georow = georows.next()
    del outrow, outrows

except (Exception):
    # get the traceback object
    tb = sys.exc_info()[2]
    # tbinfo contains the line number that the code failed on and the code from that line
    tbinfo = traceback.format_tb(tb)[0]
    # concatenate information together concerning the error into a message string
    pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
        str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
    # generate a message string for any geoprocessing tool errors
    msgs = "GP ERRORS:\n" + gp.GetMessages(2) + "\n"
    gp.AddError(msgs)

    # return gp messages for use with a script tool
    gp.AddError(pymsg)

    # print messages for use in Python/PythonWin
    print (pymsg)
    print (msgs)
    raise
