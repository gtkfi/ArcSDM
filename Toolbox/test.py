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
arcpy.ImportToolbox(".arcsdm.tbx")
print ("Testing Calculate_weights -tool (Wofe)")


# This _SHOULD_ work against filegeodatabase
arcpy.env.workspace = gdbdir + gdb;
arcpy.env.scratchWorkspace = gdbdir + gdb;

arcpy.env.outputCoordinateSystem = arcpy.SpatialReference("c:/SDM/NewWofE_beto/WofE/TrainGP.prj")
arcpy.env.cellSize = "c:/SDM/NewWofE_beto/WofE/Geologia"


arcpy.env.mask = "c:/SDM/NewWofE_beto/WofE/geologia"
arcpy.Delete_management(r"C:\Users\tronkko\OneDrive\Work\MPM\MPM\as_rcl_CalculateWeights.dbf");

arcpy.ArcSDM.CalculateWeights("C:\\SDM\\NewWofE_beto\\WofE\\as_rcl", "VALUE", "C:\\SDM\\NewWofE_beto\\WofE\\TrainGP.shp", "Ascending", r"C:\Users\tronkko\OneDrive\Work\MPM\MPM\as_rcl_CalculateWeights.dbf", 2, 1, -99)
print ("Ok?");
