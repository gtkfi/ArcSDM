import sys
import arcpy

import arcsdm;

from arcsdm import *
from arcsdm.sitereduction import ReduceSites
from arcsdm.calculateweights import Calculate
from arcsdm.categoricalmembership import Calculate
from arcsdm.logisticregression import Execute
from arcsdm.areafrequency import Execute


   

import importlib
from imp import reload;


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        
        self.label = "Area Frequency Toolbox (tmp)"
        self.alias = "areafreq" 

        # List of tool classes associated with this toolbox
        self.tools = [ArealFrequency]


class ArealFrequency(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Areal frequency"
        self.description = "TODO: Copy this from old toolbox"
        self.canRunInBackground = False
        #self.category = "Weights of Evidence"

    def getParameterInfo(self):
        """Define parameter definitions"""
        # TODO: Multiple rasters?
        
        paramTrainingSites = arcpy.Parameter(
        displayName="Training sites",
        name="training_sites",
        #datatype="DEFeatureClass",
        datatype="GPFeatureLayer",
        parameterType="Required",
        direction="Input")
                     
        
        paramRaster = arcpy.Parameter(
        displayName="Input Raster Layer",
        name="input_raster_layer",
        datatype="GPRasterLayer",
        parameterType="Required",
        direction="Input")
        
        
        paramField = arcpy.Parameter(
        displayName="Value field",
        name="valuefield_name",
        datatype="Field",
        parameterType="Required",
        direction="Input")
        #paramField.filter.list = [['Short', 'Long']]
        #paramField.parameterDependencies = [paramRaster.name];
        paramField.value = "VALUE";
        
        #param1.filter.type = "ValueList";
        #param1.filter.list = ["o", "c"];
        #param1.value = "o";
        
        paramUnitArea = arcpy.Parameter(
        displayName="Unit area (km2)",
        name="Unit_Area__sq_km_",
        datatype="GPDouble",
        parameterType="Required",
        direction="Input")
        
        
        
  
        
        
        paramOutputTable = arcpy.Parameter(
        displayName="Output table",
        name="Output_Table",
        datatype="DEDbaseTable",
        parameterType="Required",
        direction="Output")
        paramOutputTable.value = "%WorkspaceDir%\LR_logpol"
                
    
        
        
                                  
        params = [paramTrainingSites, paramRaster, paramField, paramUnitArea, paramOutputTable]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
     
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        #3.4
        try:
            importlib.reload (arcsdm.areafrequency)
        except :
            reload(arcsdm.areafrequency);
        # To list what functions does module contain
        #messages.addWarningMessage(dir(arcsdm.SiteReduction));
        #arcsdm.CalculateWeights.Calculate(self, parameters, messages);
        #messages.AddMessage("Waiting for debugger")
        #wait_for_debugger(15);
        #No do yet
        arcsdm.areafrequency.Execute(self, parameters, messages)
        return
        
        