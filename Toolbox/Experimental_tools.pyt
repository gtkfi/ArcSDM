import sys
import arcpy

import arcsdm.somtool
import arcsdm.rescale_raster
import arcsdm.adaboost
import arcsdm.SelectRandomPoints
import arcsdm.EnrichPoints

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
        self.tools = [rastersom, rescaleraster, SelectRandomPoints, EnrichPoints, Adaboost]

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

class SelectRandomPoints(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Select Random Points"
        self.description = "Selects random points far from selected points inside areas with points"
        self.canRunInBackground = False
        self.category = "Utilities"

    def getParameterInfo(self):
        output_workspace = arcpy.Parameter(
            displayName="Output Workspace",
            name="output_workspace",
            datatype="DEWorkspace",
            parameterType="Optional",
            direction="Input")

        output_point = arcpy.Parameter(
            displayName="Output Point Feature Class",
            name="output_point",
            datatype="GPString",
            parameterType="Required",
            direction="Output")
        # output_point.filter.list = ["Point", "Multipoint"]
        # output_point.parameterDependencies = [output_workspace.name]

        number_points= arcpy.Parameter(
            displayName="Number of Random Points",
            name="number_points",
            datatype="GPLong",
            parameterType="Required",
            direction = "Input")

        constraining_area = arcpy.Parameter(
            displayName="Constraining Area",
            name="constraining_area",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")
        constraining_area.filter.list = ["Polygon"]

        data_rasters = arcpy.Parameter(
            displayName="Data Rasters",
            name="data_rasters",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input")
        data_rasters.columns = [['GPRasterLayer', 'Information Rasters']]

        excluding_points = arcpy.Parameter(
            displayName="Excluding Points",
            name="excluding_points",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input")
        excluding_points.filter.list = ["Point", "Multipoint"]

        excluding_distance = arcpy.Parameter(
            displayName="Excluding Distance",
            name="excluding_distance",
            datatype="GPLinearUnit",
            parameterType="Optional",
            direction="Input",
            enabled=False)

        minimum_distance = arcpy.Parameter(
            displayName="Minimum Distance",
            name="minimum_distance",
            datatype="GPLinearUnit",
            parameterType="Optional",
            direction="Input")

        params = [output_workspace, output_point, number_points, constraining_area, data_rasters, excluding_points, excluding_distance, minimum_distance ]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        if parameters[5].altered:
            parameters[6].enabled = (parameters[5].value is not None)

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        return execute_tool(arcsdm.SelectRandomPoints.execute, self, parameters, messages)


class EnrichPoints(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Enrich Points"
        self.description = 'Adds data to the attribute table of the underlying rasters as well as mark them as ' \
                           'Prospective or not and replaces/deletes missing data'
        self.canRunInBackground = False
        self.category = "Utilities"

    def getParameterInfo(self):

        deposit_points = arcpy.Parameter(
            displayName="Deposit Points",
            name="deposit_points",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input")
        deposit_points.filter.list = ["Point", "Multipoint"]

        non_deposit_points = arcpy.Parameter(
            displayName="Non Deposit Points",
            name="non_deposit_points",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input")
        non_deposit_points.filter.list = ["Point", "Multipoint"]

        information_rasters = arcpy.Parameter(
            displayName="Information Rasters",
            name="info_rasters",
            datatype="GPValueTable",
            parameterType="Optional",
            direction="Input")
        information_rasters.columns = [['GPRasterLayer', 'Information Rasters']]

        missing_mask = arcpy.Parameter(
            displayName="Missing Value",
            name="missing_value",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input")

        output = arcpy.Parameter(
            displayName="Output",
            name="output",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Output")
        output.filter.list = ["Point", "Multipoint"]

        params = [deposit_points, non_deposit_points, information_rasters, missing_mask, output ]
        return params

    def isLicensed(self):
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
        execute_tool(arcsdm.EnrichPoints.execute , self, parameters, messages)
        return


class Adaboost(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Adaboost"
        self.description = 'Performs Adaboost algorithm for supervised machine learning'
        self.canRunInBackground = False
        self.category = "Adaboost"

    def getParameterInfo(self):

        prospective_train_points = arcpy.Parameter(
            displayName="Prospective Training Points",
            name="prospective_train_points",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")
        prospective_train_points.filter.list = ["Point", "Multipoint"]

        non_prospective_train_points = arcpy.Parameter(
            displayName="Non Prospective Training Points",
            name="non_prospective_train_points",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")
        non_prospective_train_points.filter.list = ["Point", "Multipoint"]

        prospective_test_points = arcpy.Parameter(
            displayName="Prospective Test Points",
            name="prospective_test_points",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input",
            category="Test")
        prospective_test_points.filter.list = ["Point", "Multipoint"]

        non_prospective_test_points = arcpy.Parameter(
            displayName="Non Prospective Test Points",
            name="non_prospective_test_points",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input",
            category="Test")
        non_prospective_test_points.filter.list = ["Point", "Multipoint"]

        information_rasters = arcpy.Parameter(
            displayName="Information Rasters",
            name="info_rasters",
            datatype="GPValueTable",
            parameterType="Required",
            direction="Input")
        information_rasters.columns = [['GPRasterLayer', 'Information Rasters']]

        num_estimators = arcpy.Parameter(
            displayName="Number of Estimators",
            name="num_estimators",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
            category="Model")

        learning_rate = arcpy.Parameter(
            displayName="Learning Rate",
            name="learning_rate",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input",
            category="Model")

        missing_mask = arcpy.Parameter(
            displayName="Missing Value",
            name="missing_value",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input",
            category="Model")

        output_model = arcpy.Parameter(
            displayName="Output Model",
            name="output_model",
            datatype="DEFile",
            parameterType="Optional",
            direction="Output",
            category="Output")
        # TODO: Make this Optional
        # output_model.filter.list = ['pkl']

        output_map = arcpy.Parameter(
            displayName="Output Map",
            name="output_map",
            datatype="DERasterDataset",
            parameterType="Optional",
            direction="Output",
            category="Output")

        params = [prospective_train_points, non_prospective_train_points, prospective_test_points,
                  non_prospective_test_points, information_rasters, num_estimators, learning_rate, missing_mask,
                  output_model, output_map]
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        # if parameters[8].altered:
            # if "\\" not in parameters[8].ValueAsText or "/" not in parameters[8].ValueAsText:
            #     parameters[8].Value = arcpy.CreateScratchName(parameters[8].ValueAsText, ".pkl",
            #                                                   workspace=arcpy.env.scratchFolder)
            # if not parameters[8].ValueAsText.endswith('.pkl'):
            #     parameters[8].ValueAsText = parameters[8].ValueAsText + '.pkl'

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        execute_tool(arcsdm.adaboost.Execute, self, parameters, messages)
        return
