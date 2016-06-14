# Commandline test for WEights of evidence toolbox
#
# Testing: Training sites reduction
#
# Tero Rönkkö, Geological survey of Finland 2016
#
# TODO: This doesn't test correctly - Works in ArcGis pro - (input is layer)




import os
import arcpy
import os.path

#Change this to point whereever you wish to create test db...
gdbdir = "../work/"
gdb = "database.gdb"

#TODO: Adjust datadirs to point to demo data when it comes ready
datadir = "../work/"


print ("Checking if database exists...");
if (not (os.path.exists(gdbdir + gdb))):
    print (" it doesn't, creating it...");
    arcpy.CreateFileGDB_management( gdbdir, gdb );
    print ("done.");
    

arcpy.CheckOutExtension("spatial")
arcpy.ImportToolbox("../toolbox/arcsdm.tbx")
print ("Testing Training sites reduction -tool (Wofe)...")


# This _SHOULD_ work against filegeodatabase
arcpy.env.workspace = gdbdir + gdb;
arcpy.env.scratchWorkspace = gdbdir + gdb;


dataset = "../work/database.gdb/gold_deposits"
arcpy.env.outputCoordinateSystem = arcpy.Describe(dataset).spatialReference
arcpy.env.cellSize = "../work/database.gdb/study_area"


arcpy.env.mask = "../work/database.gdb/study_area"




arcpy.ArcSDM.SiteReduction("../work/database.gdb/gold_deposits", True, 50, False, 50)
