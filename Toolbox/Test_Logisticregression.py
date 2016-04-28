# -*- coding: utf-8 -*-
# Commandline test for Logistic regression
#
# Tero Ronkko, Geological survey of Finland 2016
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
arcpy.ImportToolbox(r"./arcsdm.tbx")
print ("Testing Logistic regression -tool (Wofe)...")


# This _SHOULD_ work against filegeodatabase
arcpy.env.workspace = gdbdir + gdb;
arcpy.env.scratchWorkspace = gdbdir + gdb;

arcpy.env.outputCoordinateSystem = arcpy.SpatialReference("../demodata/TrainGP.prj")
arcpy.env.cellSize = "../demodata/Geologia"


arcpy.env.mask = "../demodata/geologia"

arcpy.ArcSDM.LogisticRegression("../demodata/as_rcl", "o", "../demodata/results/as_rcl_CalculateWeights.dbf", "../demodata/TrainGP.shp", -99, 1, r"C:\Users\tronkko\OneDrive\Work\MPM\MPM\MPM.gdb\LR_logpol", r"C:\Users\tronkko\OneDrive\Work\MPM\MPM\MPM.gdb\LR_Coeff", r"C:\Users\tronkko\OneDrive\Work\MPM\MPM\MPM.gdb\LR_pprb", r"C:\Users\tronkko\OneDrive\Work\MPM\MPM\MPM.gdb\LR_std", r"C:\Users\tronkko\OneDrive\Work\MPM\MPM\MPM.gdb\LR_conf")
print ("Test complete.");

