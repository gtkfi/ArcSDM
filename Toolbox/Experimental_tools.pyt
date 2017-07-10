import sys
import arcpy

import arcsdm.somtool
import arcsdm.rescale_raster
import arcsdm.SelectRandomPoints
import arcsdm.EnrichPoints
import arcsdm.AdaboostBestParameters
import arcsdm.ModelValidation
import arcsdm.ApplyModel
import arcsdm.MulticlassSplit
import arcsdm.ApplyFilter
import arcsdm.ModelTrain

from arcsdm.common import execute_tool
import arcsdm.general_func as general

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
                      ModelValidation, MulticlassSplit, ApplyModel, ApplyFilter, LogisticRegressionTrain, SVMTrain,
                      BrownBoostTrain]

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

        buffer_points = arcpy.Parameter(
            displayName="buffer Points",
            name="buffer_points",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input")
        buffer_points.filter.list = ["Point", "Multipoint"]

        buffer_distance = arcpy.Parameter(
            displayName="buffer Distance",
            name="buffer_distance",
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

        select_inside = arcpy.Parameter(
            displayName="Select inside buffer",
            name="select_inside",
            datatype="Boolean",
            parameterType="Optional",
            direction="Input",
            enabled=False)
        select_inside.value = False;

        params = [output_workspace, output_point, number_points, constraining_area, data_rasters, buffer_points,
                  buffer_distance, select_inside, minimum_distance ]
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
            parameters[7].enabled = (parameters[5].value is not None)

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

        class1_points = arcpy.Parameter(
            displayName="Class 1 Points (Deposit)",
            name="class1_points",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input")
        class1_points.filter.list = ["Point", "Multipoint"]

        class2_points = arcpy.Parameter(
            displayName="Class -1 Points (Non Deposit)",
            name="class2_points",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input")
        class2_points.filter.list = ["Point", "Multipoint"]

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

        field_name = arcpy.Parameter(
            displayName="Class field name",
            name="field_name",
            datatype="GPString",
            parameterType="Required",
            direction="Output")
        field_name.value = "Deposit"

        copy_data = arcpy.Parameter(
            displayName="Copy all the information from the points",
            name="copy_data",
            datatype="Boolean",
            parameterType="Optional",
            direction="Input")
        copy_data.value = True;

        params = [class1_points, class2_points, field_name, copy_data, information_rasters, missing_mask, output ]
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

        classifier_name = arcpy.Parameter(
            displayName="classifier name",
            name="classifier_name",
            datatype="GPString",
            parameterType="Derived",
            direction="Output")
        classifier_name.value = "Adaboost"


        params = [train_points, train_regressors, train_response, num_estimators, learning_rate, output_model,
                  leave_one_out, classifier_name]
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

        return


    def execute(self, parameters, messages):
        """The source code of the tool."""
        execute_tool(arcsdm.ModelTrain.execute, self, parameters, messages)
        return


class ModelValidation(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Model Validation"
        self.description = 'Validates a classification model with an independent test set'
        self.canRunInBackground = False
        self.category = "Adaboost"

    def getParameterInfo(self):

        classification_model = arcpy.Parameter(
            displayName="Classification Model",
            name="classification_model",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input")

        test_points = arcpy.Parameter(
            displayName="Test Points",
            name="test_points",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")
        test_points.filter.list = ["Point", "Multipoint"]

        test_response = arcpy.Parameter(
            displayName="Test Response",
            name="test_response_name",
            datatype="Field",
            parameterType="Required",
            direction="Input")
        test_response.parameterDependencies = [test_points.name]
        test_response.filter.list = ['Short', 'Long', 'Double', 'Float', 'Single']

        plot_file = arcpy.Parameter(
            displayName="Plot Results",
            name="plot_file",
            datatype="DEFile",
            parameterType="Optional",
            direction="Output")

        threshold = arcpy.Parameter(
            displayName="Threshold",
            name="threshold",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input")
        threshold.value = 0.5

        params = [classification_model, test_points, test_response, threshold, plot_file]
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
        execute_tool(arcsdm.ModelValidation.execute, self, parameters, messages)
        return


class ApplyModel(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Apply model"
        self.description = 'Applies a model to a series of data rasters obtain a response raster'
        self.canRunInBackground = False
        self.category = "Adaboost"

    def getParameterInfo(self):

        input_model = arcpy.Parameter(
            displayName="Input Model",
            name="input_model",
            datatype="DEFile",
            parameterType="Required",
            direction="Input")
        input_model.filter.list = ['pkl']

        output_map = arcpy.Parameter(
            displayName="Output Map",
            name="output_map",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output")

        information_rasters = arcpy.Parameter(
            displayName="Information Rasters",
            name="info_rasters",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input",
            multiValue=True)

        params = [input_model, information_rasters, output_map]
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
        execute_tool(arcsdm.ApplyModel.execute, self, parameters, messages)

        return

class MulticlassSplit(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Multiclass to Binary"
        self.description = 'Form a multi-class polygon creates one raster per class'
        self.canRunInBackground = False
        self.category = "Utilities"

    def getParameterInfo(self):

        input_feature = arcpy.Parameter(
            displayName="Input Feature",
            name="input_feature",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")
        input_feature.filter.list = ["Polygon"]

        class_field = arcpy.Parameter(
            displayName="Class Field",
            name="class_field",
            datatype="Field",
            parameterType="Required",
            direction="Input")
        class_field.parameterDependencies = [input_feature.name]

        output_prefix = arcpy.Parameter(
            displayName="Output Prefix",
            name="output_prefix",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output")

        transformation = arcpy.Parameter(
            displayName="Transformation",
            name="transformation",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        transformation.value = "Distance"
        transformation.filter.list = ["Distance", "Inverse Distance", "Inverse Linear Distance", "Binary"]


        params = [input_feature, class_field, output_prefix, transformation]
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
        execute_tool(arcsdm.MulticlassSplit.execute, self, parameters, messages)

        return


class ApplyFilter(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Apply Filter"
        self.description = 'Applies a smoothing filter to a raster'
        self.canRunInBackground = False
        self.category = "Utilities"

    def getParameterInfo(self):

        input_raster = arcpy.Parameter(
            displayName="Input Raster",
            name="input_raster",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input")

        output_raster = arcpy.Parameter(
            displayName="Output Raster",
            name="output_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output")

        filter_type = arcpy.Parameter(
            displayName="Filter Type",
            name="filter_type",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        filter_type.filter.list = ["Gaussian", "Mean", "Median"]

        filter_size = arcpy.Parameter(
            displayName="Filter Size",
            name="filter_size",
            datatype="GPLong",
            parameterType="Required",
            direction="Input")
        filter_size.value = 5
        filter_size.filter.type = "Range"
        filter_size.filter.list = [1, 100]

        params = [input_raster, output_raster, filter_type, filter_size]
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
        execute_tool(arcsdm.ApplyFilter.execute, self, parameters, messages)

        return


class LogisticRegressionTrain(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Logistic Regression Train"
        self.description = 'Trains a classificator using Logistic regression'
        self.canRunInBackground = False

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

        classifier_name = arcpy.Parameter(
            displayName="classifier name",
            name="classifier_name",
            datatype="GPString",
            parameterType="Derived",
            direction="Output")
        classifier_name.value = "Logistic Regression"

        deposit_weight = arcpy.Parameter(
            displayName="Weight of Deposit Class",
            name="deposit_weight",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input")
        deposit_weight.filter.type = "Range"
        deposit_weight.filter.list = [0.1, 99.9]

        random_state = arcpy.Parameter(
            displayName="Random State",
            name="random_state",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input")

        penalty = arcpy.Parameter(
            displayName="Penalty norm",
            name="penalty",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        penalty.value = "l2"
        penalty.filter.list = ["l1", "l2"]

        params = [train_points, train_regressors, train_response, deposit_weight, penalty, output_model, leave_one_out,
                  classifier_name, random_state]
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        parameter_dic = {par.name: par for par in parameters}
        output_model = parameter_dic["output_model"]

        if output_model.altered :
            if not output_model.valueAsText.endswith(".pkl"):
                output_model.value = output_model.valueAsText + ".pkl"

        return


    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        parameter_dic = {par.name: par for par in parameters}
        train_regressors = parameter_dic["train_regressors"]
        train_response = parameter_dic["train_response"]

        if (train_response.altered or train_regressors.altered) and train_regressors.valueAsText is not None :
            for field in train_regressors.valueAsText.split(";"):
                if field == train_response.valueAsText:
                    train_response.setErrorMessage("{} can not be included in {}".format(train_response.displayName,
                                                                                         train_regressors.displayName))

        return


    def execute(self, parameters, messages):
        """The source code of the tool."""
        execute_tool(arcsdm.ModelTrain.execute, self, parameters, messages)
        return



class BrownBoostTrain(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "BrownBoost Train"
        self.description = 'Trains a classificator using BrownBoost'
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

        countdown = arcpy.Parameter(
            displayName="Countdown",
            name="countdown",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input")
        countdown.value = 10
        countdown.filter.type = "Range"
        countdown.filter.list = [0.0000000000001 , 10]

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

        classifier_name = arcpy.Parameter(
            displayName="classifier name",
            name="classifier_name",
            datatype="GPString",
            parameterType="Derived",
            direction="Output")
        classifier_name.value = "Brownboost"


        params = [train_points, train_regressors, train_response, countdown, output_model, leave_one_out,
                  classifier_name]
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        parameter_dic = {par.name: par for par in parameters}
        output_model = parameter_dic["output_model"]

        if output_model.altered:
            if not output_model.valueAsText.endswith(".pkl"):
                output_model.value = output_model.valueAsText + ".pkl"

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        parameter_dic = {par.name: par for par in parameters}
        train_regressors = parameter_dic["train_regressors"]
        train_response = parameter_dic["train_response"]

        if (train_response.altered or train_regressors.altered) and train_regressors.valueAsText is not None:
            for field in train_regressors.valueAsText.split(";"):
                if field == train_response.valueAsText:
                    train_response.setErrorMessage("{} can not be included in {}".format(train_response.displayName,
                                                                                         train_regressors.displayName))

    def execute(self, parameters, messages):
        """The source code of the tool."""
        general.execute_tool(arcsdm.ModelTrain.execute, self, parameters, messages)
        return



class SVMTrain(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Support Vector Machine Train"
        self.description = 'Trains a classificator using SVM'
        self.canRunInBackground = False

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

        classifier_name = arcpy.Parameter(
            displayName="classifier name",
            name="classifier_name",
            datatype="GPString",
            parameterType="Derived",
            direction="Output")
        classifier_name.value = "SVM"

        deposit_weight = arcpy.Parameter(
            displayName="Weight of Deposit Class",
            name="deposit_weight",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input")
        deposit_weight.filter.type = "Range"
        deposit_weight.filter.list = [0.1, 99.9]

        random_state = arcpy.Parameter(
            displayName="Random State",
            name="random_state",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input")

        penalty = arcpy.Parameter(
            displayName="Penalty parameter",
            name="penalty",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input")
        penalty.value = 1.0

        kernel = arcpy.Parameter(
            displayName="Kernel",
            name="kernel",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        kernel.value = "rbf"
        kernel.filter.list = ['linear', 'poly', 'rbf', 'sigmoid']

        normalize = arcpy.Parameter(
            displayName="Normalize data",
            name="normalize",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Output")
        normalize.value = True

        params = [train_points, train_regressors, train_response, kernel, deposit_weight, penalty, output_model,
                  leave_one_out, classifier_name, normalize, random_state]
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        parameter_dic = {par.name: par for par in parameters}
        output_model = parameter_dic["output_model"]

        if output_model.altered :
            if not output_model.valueAsText.endswith(".pkl"):
                output_model.value = output_model.valueAsText + ".pkl"

        return


    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        parameter_dic = {par.name: par for par in parameters}
        train_regressors = parameter_dic["train_regressors"]
        train_response = parameter_dic["train_response"]

        if (train_response.altered or train_regressors.altered) and train_regressors.valueAsText is not None :
            for field in train_regressors.valueAsText.split(";"):
                if field == train_response.valueAsText:
                    train_response.setErrorMessage("{} can not be included in {}".format(train_response.displayName,
                                                                                         train_regressors.displayName))

        return


    def execute(self, parameters, messages):
        """The source code of the tool."""

        general.execute_tool(arcsdm.ModelTrain.execute, self, parameters, messages)
        return

