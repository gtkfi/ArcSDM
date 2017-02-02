#
# Rescale raster - ArcSDM 5 
# 
# Rescales source raster to new floating raster - typically 0..1 to be used as SOM network input.
#
# Tero Ronkko, Geological survey of Finland, 2017
#
# TODO: 
# Cleanup
# Make negatives count/ignore negatives
#
# History:
# 2.2.2017 First version added to toolbox


import arcpy;
import numpy;
from minisom import MiniSom    
from datetime import datetime
import itertools;
import random;



def myprint (str):
    arcpy.AddMessage(str);
    print (str);

    
def addToDisplay(layer, name, position):
    result = arcpy.MakeRasterLayer_management(layer, name)
    lyr = result.getOutput(0)    
    product = arcpy.GetInstallInfo()['ProductName']
    if "Desktop" in product:
        mxd = arcpy.mapping.MapDocument("CURRENT")
        dataframe = arcpy.mapping.ListDataFrames(mxd)[0]
        arcpy.mapping.AddLayer(dataframe, lyr, position)
    elif "Pro" in product:
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        m = aprx.listMaps("Map")[0]
        m.addLayer(lyr, position)
        


def execute(self, parameters, messages):    
    rasteri = arcpy.Raster(parameters[0].valueAsText)
    arcpy.env.overwriteOutput = True
    lowervalue =parameters[1].value;
    uppervalue = parameters[2].value;

    output_rastername = parameters[3].valueAsText;


    raster_array = arcpy.RasterToNumPyArray (rasteri);


    myprint ("="*10 + " Rescale raster " + "="*10);
    myprint ("Starting rescale raster");

    

    #myprint (str(raster_array));

    diff = uppervalue - lowervalue;

    myprint (" max(raster_array):%s diff:%s" %( str( numpy.max(raster_array)), str(diff) ));



    raster_array = ( raster_array / (float((numpy.max(raster_array))) ) * diff );
    raster_array = raster_array + lowervalue;

    myprint("Calculation done.");
    #myprint (str(raster_array));



        
    mx = rasteri.extent.XMin + 0 * rasteri.meanCellWidth

    my = rasteri.extent.YMin + 0 * rasteri.meanCellHeight

    #myprint ( "Size of output raster: %s x %s"%( numpy.shape(numero4),numpy.shape(uusi)));

    #myprint (uusi[100]);

    # Overwrite
    arcpy.env.overwriteOutput = True
    myRasterBlock = arcpy.NumPyArrayToRaster(raster_array, arcpy.Point(mx, my),rasteri.meanCellWidth, rasteri.meanCellHeight);

    myprint ("Saving new raster...\n Output rastername: %s"% (output_rastername ));
    #myRasterBlock.save("d:\\arcgis\\database.gdb\\tulos");
    myRasterBlock.save(output_rastername);
    desc = arcpy.Describe(output_rastername)
    name = desc.file + "_layer";
    myprint (" Map layer name: " + name); 
    parameters[3].value =  myRasterBlock    ;
    addToDisplay(output_rastername, name , "TOP");
                

    
    #myprint ("Rescale raster complete!");
