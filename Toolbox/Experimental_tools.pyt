import sys
import arcpy

import arcsdm.somtool
import arcsdm.rescale_raster
import arcsdm.adaboost
import arcsdm.SelectRandomPoints
import arcsdm.EnrichPoints
import arcsdm.AdaboostBestParameters
import arcsdm.AdaboostTrain

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
        self.tools = [rastersom, rescaleraster, SelectRandomPoints, EnrichPoints, AdaboostBestParameters, AdaboostTrain,
                       Adaboost]

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

class AdaboostBestParameters(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Adaboost Best Parameters"
        self.description = 'Makes a grid search for the paramethers with best score against test/train set'
        self.canRunInBackground = False
        self.category = "Adaboost"

    def getParameterInfo(self):

        train_points = arcpy.Parameter(
            displayName="Train Points",
            name="train_points",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")
        train_points.filter.list = ["Point", "Multipoint"]

        train_regressors = arcpy.Parameter(
            displayName="Train Regressors",
            name="train_regressors",
            datatype="Field",
            parameterType="Required",
            direction="Input",
            multiValue=True)
        train_regressors.parameterDependencies = [train_points.name]
        train_regressors.filter.list = ['Short', 'Long', 'Double', 'Float', 'Single']

        train_response = arcpy.Parameter(
            displayName="Train Response",
            name="train_response",
            datatype="Field",
            parameterType="Required",
            direction="Input")
        train_response.parameterDependencies = [train_points.name]
        train_response.filter.list = ['Short', 'Long', 'Double', 'Float', 'Single']

        test_response = arcpy.Parameter(
            displayName="Test Response",
            name="test_response",
            datatype="Field",
            parameterType="Optional",
            direction="Input",
            enabled=False)
        test_response.parameterDependencies = [train_points.name]
        test_response.filter.list = ['Short', 'Long', 'Double', 'Float', 'Single']

        num_estimators_min = arcpy.Parameter(
            displayName="Minimum number of Estimators",
            name="num_estimators_min",
            datatype="GPLong",
            parameterType="Required",
            direction="Input",
            category="Number of estimators")
        num_estimators_min.value = 1

        num_estimators_max = arcpy.Parameter(
            displayName="Maximum Number of Estimators",
            name="num_estimators_max",
            datatype="GPLong",
            parameterType="Required",
            direction="Input",
            category="Number of estimators")
        num_estimators_max.value = 20

        num_estimators_increment = arcpy.Parameter(
            displayName="Increment of number of Estimators",
            name="num_estimators_increment",
            datatype="GPLong",
            parameterType="Required",
            direction="Input",
            category="Number of estimators")
        num_estimators_increment.value = 2

        learning_rate_min = arcpy.Parameter(
            displayName="Minimum learning Rate",
            name="learning_rate_min",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="Learning Rate")
        learning_rate_min.value = 0.5

        learning_rate_max = arcpy.Parameter(
            displayName="Maximum learning Rate",
            name="learning_rate_max",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="Learning Rate")
        learning_rate_max.value = 1.5

        learning_rate_increment = arcpy.Parameter(
            displayName="Increment of learning Rate",
            name="learning_rate_increment",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="Learning Rate")
        learning_rate_increment.value = 0.2

        output_table = arcpy.Parameter(
            displayName="Output Table",
            name="output_table",
            datatype="DEFile",
            parameterType="Optional",
            direction="Output",
            category="Output")

        plot_file = arcpy.Parameter(
            displayName="Plot Results",
            name="plot_file",
            datatype="DEFile",
            parameterType="Optional",
            direction="Output",
            category="Output")

        params = [train_points, train_regressors, train_response, num_estimators_min,
                  num_estimators_max, num_estimators_increment, learning_rate_min, learning_rate_max,
                  learning_rate_increment, output_table, plot_file]
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        if parameters[10].altered and not parameters[10].valueAsText.endswith(".png"):
            parameters[10].value = parameters[10].valueAsText + ".png"
        if parameters[9].altered and not parameters[9].valueAsText.endswith(".dbf"):
            parameters[9].value = parameters[9].valueAsText + ".dbf"

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        if any([parameters[x].altered for x in xrange(3, 6)]):
            if parameters[3].value > parameters[4].value:
                parameters[3].setErrorMessage("Minimum value greater than maximum")
            elif (parameters[4].value - parameters[3].value) < parameters[5].value:
                parameters[5].setWarningMessage("Increment greater than the interval")
            elif parameters[3].value <= 0:
                parameters[3].setErrorMessage("Non positive values forbidden")
            elif parameters[5].value <= 0 and (parameters[4].value - parameters[3].value) > 0:
                parameters[5].setErrorMessage("Non positive values forbidden")

        if any([parameters[x].altered for x in xrange(6, 9)]):
            if parameters[6].value > parameters[7].value:
                parameters[6].setErrorMessage("Minimum value greater than maximum")
            elif (parameters[7].value - parameters[6].value) < parameters[8].value:
                parameters[8].setWarningMessage("Increment greater than the interval")
            elif parameters[6].value <= 0:
                parameters[6].setErrorMessage("Non positive values forbidden")
            elif parameters[8].value <= 0 and (parameters[7].value - parameters[6].value) > 0:
                parameters[8].setErrorMessage("Non positive values forbidden")
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        execute_tool(arcsdm.AdaboostBestParameters.execute, self, parameters, messages)
        return


class AdaboostTrain(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Adaboost Train"
        self.description = 'Trains a classificator using Adaboost'
        self.canRunInBackground = False
        self.category = "Adaboost"

    def getParameterInfo(self):

        train_points = arcpy.Parameter(
            displayName="Train Points",
            name="train_points",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")
        train_points.filter.list = ["Point", "Multipoint"]

        train_regressors = arcpy.Parameter(
            displayName="Train Regressors",
            name="train_regressors",
            datatype="Field",
            parameterType="Required",
            direction="Input",
            multiValue=True)
        train_regressors.parameterDependencies = [train_points.name]
        train_regressors.filter.list = ['Short', 'Long', 'Double', 'Float', 'Single']

        train_response = arcpy.Parameter(
            displayName="Train Response",
            name="train_response",
            datatype="Field",
            parameterType="Required",
            direction="Input")
        train_response.parameterDependencies = [train_points.name]
        train_response.filter.list = ['Short', 'Long', 'Double', 'Float', 'Single']

        num_estimators = arcpy.Parameter(
            displayName="Number of Estimators",
            name="num_estimators",
            datatype="GPLong",
            parameterType="Required",
            direction="Input")
        num_estimators.value = 20
        num_estimators.filter.type = "Range"
        num_estimators.filter.list = [1, 1000]

        learning_rate = arcpy.Parameter(
            displayName="Learning Rate",
            name="learning_rate",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input")
        learning_rate.value = 1
        learning_rate.filter.type = "Range"
        learning_rate.filter.list = [0.0000000000001 , 10]

        output_model = arcpy.Parameter(
            displayName="Output Model",
            name="output_model",
            datatype="DEFile",
            parameterType="Optional",
            direction="Output")

        leave_one_out = arcpy.Parameter(
            displayName="Leave-one-out cross validation",
            name="leave_one_out",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Output")
        leave_one_out.value = True

        params = [train_points, train_regressors, train_response, num_estimators, learning_rate, output_model,
                  leave_one_out]
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""

        output_model = parameters[5]
        if output_model.altered :
            if output_model == "":
                output_model = None
            elif not output_model.valueAsText.endswith(".pkl"):
                output_model.value = output_model.valueAsText + ".pkl"

        return


    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        train_regressors = parameters[1]
        train_response = parameters[2]
        if (train_response.altered or train_regressors.altered) and train_regressors.valueAsText is not None :
            for field in train_regressors.valueAsText.split(";"):
                if field == train_response.valueAsText:
                    train_response.setErrorMessage("{} can not be included in {}".format(train_response.displayName,
                                                                                         train_regressors.displayName))

        num_estimators = parameters[3]
        if num_estimators.altered and num_estimators.value < 1:
            num_estimators.value = 1

        learning_rate = parameters[4]
        if learning_rate.altered and learning_rate.value <= 0:
            learning_rate.value = 0.1

        return


    def execute(self, parameters, messages):
        """The source code of the tool."""
        execute_tool(arcsdm.AdaboostTrain.execute, self, parameters, messages)
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
