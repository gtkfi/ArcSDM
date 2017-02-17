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
#
# History:
# 2.2.2017 First version added to toolbox


import arcpy;
import numpy;
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

    nodatathreshold = parameters[3].value;
    output_rastername = parameters[4].valueAsText;
   

    addtomap = parameters[5].value;
    ignorenegative = parameters[6].value;
   


    raster_array = arcpy.RasterToNumPyArray (rasteri)#, nodata_to_value=0);
    
    super_threshold_indices = raster_array < nodatathreshold
    raster_array[super_threshold_indices] = numpy.nan;
    
    #myprint (str(raster_array[0][0]))

    
    
    myprint ("\n" + "="*10 + " Rescale raster " + "="*10);
    myprint ("Starting rescale raster");
    
    minimi = numpy.nanmin(raster_array);
    maksimi = numpy.nanmax(raster_array);
    
    if (ignorenegative):
        myprint ("   Negatives will be changed to zero..."); 
        raster_array[raster_array < 0] = 0;
        raster_array[raster_array < 0] = 0;
        myprint ("      ...done");
    else:
        minimi = numpy.nanmin(raster_array);
        if (minimi < 0):
            myprint ("   Negatives will be spread to new raster Min: %s" % (str(minimi)) );
            raster_array +=  numpy.abs(minimi);
        
    
    #myprint (str(raster_array));

    diff = uppervalue - lowervalue;
    myprint ("   Rescaling array[%s - %s] -> array[%s .. %s] " % (str(minimi), str(maksimi),  str(lowervalue), str(uppervalue)) );
    myprint ("   max(raster_array):%s diff:%s" %( str( numpy.nanmax(raster_array)), str(diff) ));



    raster_array = ( raster_array / (float((numpy.nanmax(raster_array))) ) * diff );
    raster_array = raster_array + lowervalue;
    #myprint (str(raster_array[0][0]))

    myprint("Calculation done.");
    #myprint (str(raster_array));



        
    mx = rasteri.extent.XMin + 0 * rasteri.meanCellWidth

    my = rasteri.extent.YMin + 0 * rasteri.meanCellHeight

    #myprint ( "Size of output raster: %s x %s"%( numpy.shape(numero4),numpy.shape(uusi)));

    #myprint (uusi[100]);

    # Overwrite
    arcpy.env.overwriteOutput = True
    myRasterBlock = arcpy.NumPyArrayToRaster(raster_array, arcpy.Point(mx, my),rasteri.meanCellWidth, rasteri.meanCellHeight);

    myprint ("Saving new raster...\n   Output rastername: %s"% (output_rastername ));
    #myRasterBlock.save("d:\\arcgis\\database.gdb\\tulos");
    myRasterBlock.save(output_rastername);
    desc = arcpy.Describe(output_rastername)
    name = desc.file + "_layer";
    parameters[3].value =  myRasterBlock    ;
    if (addtomap):
        myprint ("   Adding layer to map with name: " + name); 
        addToDisplay(output_rastername, name , "TOP");
                

    
    #myprint ("Rescale raster complete!");
