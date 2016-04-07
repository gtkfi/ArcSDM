import os
import arcpy

arcpy.CheckOutExtension("spatial")
arcpy.ImportToolbox("C:\\Users\\tronkko\\OneDrive\\Code\\ArcSDM\\Toolbox\\arcsdm.tbx")
print ("Tutkitaan ArcSDM")


#print (dir(arcpy.env))
# This _SHOULD_ work against filegeodatabase
arcpy.env.workspace = "C:/Users/tronkko/OneDrive/Work/MPM/MPM/MPM.gdb"
#arcpy.env.workspace = "C:/Users/tronkko/OneDrive/Work/MPM/MPM/"
#arcpy.env.scratchWorkspace = "C:/Users/tronkko/OneDrive/Work/MPM/MPM/MPM.gdb"
arcpy.env.scratchWorkspace = "C:/Users/tronkko/OneDrive/Work/MPM/MPM/"

arcpy.env.outputCoordinateSystem = arcpy.SpatialReference("c:/SDM/NewWofE_beto/WofE/TrainGP.prj")
arcpy.env.cellSize = "c:/SDM/NewWofE_beto/WofE/Geologia"


arcpy.env.mask = "c:/SDM/NewWofE_beto/WofE/geologia"
arcpy.Delete_management(r"C:\Users\tronkko\OneDrive\Work\MPM\MPM\as_rcl_CalculateWeights.dbf");
arcpy.ArcSDM.CalculateWeights("C:\\SDM\\NewWofE_beto\\WofE\\as_rcl", "VALUE", "C:\\SDM\\NewWofE_beto\\WofE\\TrainGP.shp", "Ascending", r"C:\Users\tronkko\OneDrive\Work\MPM\MPM\as_rcl_CalculateWeights.dbf", 2, 1, -99)
