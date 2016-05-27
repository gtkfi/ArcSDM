# Commandline test for WEights of evidence
#
# Tero Rönkkö, Geological survey of Finland 2016
# TODO: Use demo dataset when it comes


import os
import arcpy
import os.path

#Change this to point whereever you wish to create test db...
gdbdir = "../../"
gdb = "arcsdmtest.gdb"

#TODO: Adjust datadirs to point to demo data when it comes ready
datadir = "../demo/"


print ("Checking if database exists...");
if (not (os.path.exists(gdbdir + gdb))):
    print (" it doesn't, creating it...");
    arcpy.CreateFileGDB_management( gdbdir, gdb );
    print ("done.");


arcpy.CheckOutExtension("spatial")
arcpy.ImportToolbox("./arcsdm.tbx")
print ("Testing Calculate bends -tool ...")


# This _SHOULD_ work against filegeodatabase
arcpy.env.workspace = gdbdir + gdb;
arcpy.env.scratchWorkspace = gdbdir + gdb;

arcpy.env.outputCoordinateSystem = arcpy.SpatialReference("../demodata/TrainGP.prj")
arcpy.env.cellSize = "../demodata/Geologia"


#arcpy.env.mask = "../demodata/geologia"
#arcpy.Delete_management("../demodata/results/as_rcl_CalculateWeights.dbf");

arcpy.ArcSDM.CalculateBends("../demodata/demodatabase.gdb/old/faults", r"../demodata/results/as_rcl_CalculateWeights.dbf")
print ("Test complete.");
