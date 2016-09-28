'''
    ArcSDM 5 
    Tero Ronkko / GTK 2016
    This can be used to symbolize raster layer with proper classes 
    TODO: Needs input from science group!
'''

import arcpy



import os
import arcsdm.sdmvalues;
import importlib;

def execute(self, parameters, messages):
    rastername = parameters[0].valueAsText; 
    trainpts = parameters[1].valueAsText;
    unitcell = parameters[2].value;
    try:
        importlib.reload (arcsdm.sdmvalues)
        importlib.reload (arcsdm.workarounds_93);
    except :
        reload(arcsdm.sdmvalues);
        reload(arcsdm.workarounds_93);   
    Temp_Symbology = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..\\raster_classified.lyr')
    desc = arcpy.Describe(rastername);
    
    arcpy.AddMessage(Temp_Symbology);
    arcpy.AddMessage("="*20 + " starting symbolize " + "="*20);
    arcpy.AddMessage("%-20s %s"% ("Raster name: ",  desc.file));
    rastername = desc.file;
    mxd = arcpy.mapping.MapDocument("CURRENT")
    df = arcpy.mapping.ListDataFrames(mxd, "")[0]
    lyr = arcpy.mapping.ListLayers(mxd, rastername, df)[0]
    prob = arcsdm.sdmvalues.getPriorProb(trainpts, unitcell)
    arcpy.AddMessage("%-20s %s" % ("Probability: ",  str(prob)));
    arcpy.AddMessage("Applying raster symbology to classified values from lyr file... ");
    arcpy.ApplySymbologyFromLayer_management(lyr, Temp_Symbology)

    
    if lyr.symbologyType == "RASTER_CLASSIFIED":
        #lyr.symbology.classBreakValues = [1, 60, 118, 165, 255]
        arcpy.AddMessage("Setting values to prob priority values ... ");
        values = lyr.symbology.classBreakValues;
        values[0] = 0;
        values[1] = prob; #TODO: Does this need rounding?
        lyr.symbology.classBreakValues = values;
        #lyr.symbology.classBreakLabels = ["1 to 60", "61 to 118", 
        #                                "119 to 165", "166 to 255"]
        #lyr.symbology.classBreakDescriptions = ["Class A", "Class B",
        #                                      "Class C", "Class D"]
        lyr.symbology.reclassify()                                          
        
        #lyr.symbology.excludedValues = '0'
    arcpy.RefreshActiveView()
    arcpy.RefreshTOC()
    del mxd, df, lyr