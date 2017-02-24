import sys
import arcpy

import arcsdm.somtool
import arcsdm.rescale_raster

from arcsdm.common import execute_tool


import importlib
from imp import reload;


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        
        self.label = "Experimental SDM toolbox"
        self.alias = "experimentaltools" 

        # List of tool classes associated with this toolbox
        self.tools = [rastersom,rescaleraster]

class rescaleraster(object):
    def __init__(self):
        self.label = "Rescale raster values to new float raster"
        self.description = "Rescales raster values to a new [min .. max] float raster (typically preprocessing step for SOM calculations) "
        self.category = "Utilities"
        self.canRunInBackground = False

    def getParameterInfo(self):
        input_raster = arcpy.Parameter(
            displayName="Input raster",
            name="input_raster",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input")

        new_min = arcpy.Parameter(
            displayName="New minimum",
            name="new_min",
            datatype="GPLong",
            parameterType="Required",         
            direction="Input")
        new_min.value = 0;
        
        new_max = arcpy.Parameter(
            displayName="New maximum",
            name="new_max",
            datatype="GPLong",
            parameterType="Required",         
            direction="Input")
        new_max.value = 1;
        
        
        min_to_na = arcpy.Parameter(
            displayName="NoData threshold value (turn all below this to NoData)",
            name="min_to_na",
            datatype="GPLong",
            parameterType="Required",         
            direction="Input")
        min_to_na.value = -100000;
        
        
        
        
        output_raster = arcpy.Parameter(
            displayName="Output rastername",
            name="results_table",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output")
        output_raster.value = "%workspace%\\rescaled_raster";
        
        paramAddToMap = arcpy.Parameter(
        displayName="Add layer to map",
        name="addtomap",
        datatype="Boolean",
        parameterType="Optional",
        direction="Input")
        paramAddToMap.value= True;
        
        paramIgnoreNegative = arcpy.Parameter(
        displayName="Ignore negative values and replace them with zero",
        name="ignorenegative",
        datatype="Boolean",
        parameterType="Optional",
        direction="Input")
        paramIgnoreNegative.value= True;
        
        
        
        return [input_raster, new_min, new_max, min_to_na, output_raster, paramAddToMap, paramIgnoreNegative ]

    def isLicensed(self):
        return True
        
    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        if parameters[0].value:
            if parameters[0].altered:
                layer = parameters[0].valueAsText;
                desc = arcpy.Describe(layer)
                name = desc.file;
                type = parameters[4].valueAsText;
                filename = name + "_rescaled";
                #Update name accordingly
                resulttmp = "%WORKSPACE%\\" + name + "_rescaled"; #Output is _W + first letter of type
                lopullinen_nimi = arcpy.CreateUniqueName(filename)
                parameters[4].value =  lopullinen_nimi #.replace(".","");  #Remove illegal characters
        return

    def execute(self, parameters, messages):        
        #execute_tool(arcsdm.roctool.execute, self, parameters, messages)
        
        try:
            importlib.reload (arcsdm.rescale_raster)
        except :
            reload(arcsdm.rescale_raster);
        arcsdm.rescale_raster.execute (self, parameters, messages);
        return
                


        
class rastersom(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "SOM - Linear distance from BMU"
        self.description = "Returns raster of linear distance from SOM BMU of input raster"
        self.canRunInBackground = False
        self.category = "SOM"
        
    def getParameterInfo(self):
        """Define parameter definitions"""
        paramInput = arcpy.Parameter(
        displayName="Input data raster (composite bands)",
        name="inputdata",
        datatype="GPRasterLayer",
        parameterType="Required",
        direction="Input")
        
        paramSomsize = arcpy.Parameter(
        displayName="SOM size",
        name="somsize",
        datatype="GPLong",
        parameterType="Required",
        direction="Input")
        paramSomsize.value = "24";
        
        paramIterations = arcpy.Parameter(
        displayName="Training iterations",
        name="iterations",
        datatype="GPLong",
        parameterType="Required",
        direction="Input")
        paramIterations.value = "1";
        
        paramSigma = arcpy.Parameter(
        displayName="Sigma",
        name="sigma",
        datatype="GPDouble",
        parameterType="Required",
        direction="Input")
        paramSigma.value = "0.3";
        
        paramLearningrate = arcpy.Parameter(
        displayName="Learning rate",
        name="learningrate",
        datatype="GPDouble",
        parameterType="Required",
        direction="Input")
        paramLearningrate.value = "0.5";
        
        paramOutputraster = arcpy.Parameter(
        displayName="Output distance raster",
        name="outputraster",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output")
        paramOutputraster.value = "%Workspace%\SOMdistance"
        
        
        
        
        
        params = [paramInput, paramSomsize, paramIterations, paramLearningrate, paramSigma, paramOutputraster]
        return params

        
            
    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        #if parameters[0].value and parameters[3].value:
        #    if parameters[0].altered or paramaters[3].altered:
        #        layer = parameters[0].valueAsText;
        #        desc = arcpy.Describe(layer)
        #        name = desc.file;
        #        type = parameters[3].valueAsText;
        #        char = type[:1];
        #        if (char != 'U'):
        #            char = 'C' + char; #Output  _C + first letter of type unless it is U
                #Update name accordingly
        #        resulttmp = "%WORKSPACE%\\" + name + "_" + char; 
        #        parameters[4].value =  resulttmp.replace(".","");  #Remove illegal characters
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
     
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        execute_tool(arcsdm.somtool.execute, self, parameters, messages)
        return        
