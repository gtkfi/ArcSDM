"""
Calculate_bends tool

ArcSDM 5 for Arcgis Pro
Recode for Arcgis Pro - Copyright 2016 GTK/Tero Rönkkö
Original Work by Don Sawatzky 2008.

TODO: Update all copyright headers


"""


# Copyright (C) 2008, Don Sawatzky
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the copyright holder nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
Creates a point feature class from a polyline feature class and calculate the degrees of
turning at each point for the interior vertices of the polyline.

"""
import arcgisscripting
import math, os, sys, traceback

def rowgen(rows):
    row = rows.next()
    while row:
        yield row
        row = rows.next()

def partgen(parts):
    numparts = parts.PartCount
    for numpart in range(numparts):
        part = parts.getPart(numpart)
        yield part

def calculate_bend(pnt0, pnt1, pnt2):
    """
        Calculate bend angle at middle vertex
        Negative bend is right bend
        Positive bend is left bend
    """
    #Get trig angle of first line
    delx1 = pnt1.X - pnt0.X
    dely1 = pnt1.Y - pnt0.Y
    angle1 = math.atan2(dely1, delx1)
    #Get trig angle of next line
    delx2 = pnt2.X - pnt1.X
    dely2 = pnt2.Y - pnt1.Y
    angle2 = math.atan2(dely2, delx2)
    #Convert del angle to degrees
    bend = math.degrees(angle2 - angle1)
    if abs(bend) > 180:
        if bend > 0: bend -= 360
        else: bend +=360
    return bend

def LinearFeaturesToPoints(polylineIn, pointsOut):
    try:
        #Create output point feature dataset
        outdir = os.path.dirname(pointsOut)
        outbase = os.path.basename(pointsOut)
        spatref = gp.describe(polylineIn).spatialreference
        gp.CreateFeatureClass_management(outdir, outbase, 'POINT', '#', '#', 'ENABLED', spatref)
        gp.AddField_management(pointsOut, "BEND", "FLOAT")
        gp.AddField_management(pointsOut, "ID", "INTEGER")


        #Open output feature class
        outrows = gp.insertcursor(pointsOut)
        #Process input features
        for inrow in rowgen(gp.searchcursor(polylineIn)):
            pline = inrow.Shape
            #Insert output row
            for linepart in partgen(pline):
                #gp.addmessage("No. pnts: %d"%linepart.Count)
                #Process output features
                for pntno in range(linepart.Count):
                    #Create next new row
                    newrow = outrows.NewRow()
                    #Insert point shape in new row
                    pnt = linepart.GetObject(pntno)
                    newrow.shape = pnt
                    #Insert polyline FID in new row
                    field_names = [f.name for f in arcpy.ListFields(polylineIn)]

                    gp.addmessage(str(field_names))
                    if ('FID' in field_names):
                        newrow.ID = inrow.FID
                        gp.addmessage("Using FID")
                    else:
                        newrow.ID = inrow.OBJECTID
                        gp.addmessage("Using OBJECTID")
                    #Calculate degrees of turning
                    if pntno == 0 or pntno == linepart.Count-1:
                        #First and last points given false turning = 400
                        newrow.BEND = 400.0
                    else:
                        #Calculate degrees of turning
                        pntlast = linepart.GetObject(pntno - 1)
                        pntnext = linepart.GetObject(pntno + 1)
                        newrow.BEND = calculate_bend(pntlast, pnt, pntnext)
                    #Insert new row in db file
                    outrows.InsertRow(newrow)
        del outrows #Like closes the output db file

    except Exception as msg:
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

if __name__ == "__main__":
    #create geoprocessor
    gp = arcgisscripting.create()
        #Get input and output dataset names
    polylineIn = gp.getparameterastext(0)
    desc = gp.describe(polylineIn)
    #gp.addmessage(desc.shapetype)
    assert desc.shapetype == 'Polyline', 'Input features: %s are not Polyline type' % polylineIn
    pointsOut = gp.getparameterastext(1)
    LinearFeaturesToPoints(polylineIn, pointsOut)
    gp.SetParameterAsText(1, pointsOut)
