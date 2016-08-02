import sys
import arcpy
import arcsdm.SiteReduction as SiteReduction;
import arcsdm.CalculateWeights as CalculateWeights;
import arcsdm;
import importlib;
from imp import reload;

class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        
        self.label = "ArcSDM python toolbox"
        self.alias = "ArcSDM python toolbox"
        # List of tool classes associated with this toolbox
        self.tools = [CalculateWeightsTool,SiteReductionTool]



class CalculateWeightsTool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Calculate Weights"
        self.description = "Calculate weight rasters from the inputs"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
        displayName="Evidence Raster Layer",
        name="evidence_raster_layer",
        datatype="GPRasterLayer",
        parameterType="Required",
        direction="Input")

        param1 = arcpy.Parameter(
        displayName="Evidence raster codefield",
        name="Evidence_Raster_Code_Field",
        datatype="Field",
        parameterType="Optional",
        direction="Input")

        paramTrainingPoints = arcpy.Parameter(
        displayName="Training points",
        name="Training_points",
        datatype="GPFeatureLayer",
        parameterType="Required",
        direction="Input")
        
        param2 = arcpy.Parameter(
        displayName="Type",
        name="Type",
        datatype="GPString",
        parameterType="Required",
        direction="Input")
        param2.filter.type = "ValueList";
        param2.filter.list = ["Descending", "Ascending", "Categorical", "Unique"];
        param2.value = "Descending";
        
        param3 = arcpy.Parameter(
        displayName="Output weights table",
        name="output_weights_table",
        datatype="DETable",
        parameterType="Required",
        direction="Output")

        param4 = arcpy.Parameter(
        displayName="Confidence Level of Studentized Contrast",
        name="Confidence_Level_of_Studentized_Contrast",
        datatype="GPDouble",
        parameterType="Required",
        direction="Input")
        param4.value = "2";
                           
        param5 = arcpy.Parameter(
        displayName="Unit area (km2)",
        name="Unit_Area__sq_km_",
        datatype="GPDouble",
        parameterType="Required",
        direction="Input")
        
        
        param6 = arcpy.Parameter(
        displayName="Missing data value",
        name="Missing_Data_Value",
        datatype="GPLong",
        parameterType="Required",
        direction="Input")
        param6.value = "-99";

                           
                                  
        params = [param0, param1, paramTrainingPoints, param2, param3, param4, param5, param6]
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
            importlib.reload (arcsdm.CalculateWeights)
        except :
            reload(arcsdm.CalculateWeights);
        # To list what functions does module contain
        #messages.addWarningMessage(dir(arcsdm.SiteReduction));
        arcsdm.CalculateWeights.Calculate(self, parameters, messages);
        return
        
        

        
        
        
class SiteReductionTool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Training sites reduction"
        self.description = "Selects subset of the training points"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
        displayName="Training sites layer",
        name="Training_Sites_layer",
        datatype="GPFeatureLayer",
        parameterType="Required",
        direction="Input")

        param1 = arcpy.Parameter(
        displayName="Thinning selection",
        name="Thinning_Selection",
        datatype="Boolean",
        parameterType="Optional",
        direction="Input")

        param2 = arcpy.Parameter(
        displayName="Unit area (sq km)",
        name="Unit_Area__sq_km_",
        datatype="GPLong",
        parameterType="Optional",
        direction="Input")
        
# Tämä vois olla hyvinkin valintalaatikko?
        param3 = arcpy.Parameter(
        displayName="Random selection",
        name="Random_selection",
        datatype="Boolean",
        parameterType="Optional",
        direction="Input")

        param4 = arcpy.Parameter(
        displayName="Random percentage selection",
        name="Random_percentage_selection",
        datatype="GPLong",
        parameterType="Optional",
        direction="Input")
                                            
        params = [param0, param1, param2, param3, param4]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        if parameters[1].value == True:
            parameters[2].enabled = True;
        else:
            parameters[2].enabled = False;            
            
        if parameters[3].value == True:
            parameters[4].enabled = True;
        
        else:
            parameters[4].enabled = False;  
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        l = 0;
        
        if parameters[1].value == True:
            #parameters[4].setErrorMessage("Random percentage value required!")
            l = l + 1;
            if parameters[2].value == '' or  parameters[2].value is None:
                parameters[2].setErrorMessage("Thinning value required!")
                
        if parameters[3].value == True:
            l = l + 1;            
            #parameters[4].setErrorMessage("Random percentage value required!")
            if parameters[4].value == '' or  parameters[4].value is None:
                parameters[4].setErrorMessage("Random percentage value required!")
            elif parameters[4].value > 100 or parameters[4].value < 0:
                parameters[4].setErrorMessage("Value has to between 0-100 %!")
            
        if (l < 1):
            parameters[1].setErrorMessage("You have to select either one!")
            parameters[3].setErrorMessage("You have to select either one!")
        
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        #3.4
        try:
            importlib.reload (arcsdm.SiteReduction)
        except :
            reload(arcsdm.SiteReduction);
        # To list what functions does module contain
        #messages.addWarningMessage(dir(arcsdm.SiteReduction));
        arcsdm.SiteReduction.ReduceSites(self, parameters, messages);
        return
        
        
        
        
        
        
        
