import sys
import arcpy

import arcsdm.somtool
import arcsdm.rescale_raster
import arcsdm.CreateRandomPoints
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
        self.tools = [rastersom, rescaleraster, CreateRandomPoints, EnrichPoints, AdaboostTrain,
                      ModelValidation, MulticlassSplit, ApplyModel, LogisticRegressionTrain, SVMTrain,
                      BrownBoostTrain, RFTrain, SelectRandomPoints]

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
            if parameters[0].altered and not parameters[4].altered:
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


class CreateRandomPoints(object):
    """ 
    Create Random Points tool
        Creates a set of random points in a given area. The area is calculated restricting to zones with full 
            information and inside/outside buffer areas 
        
        Parameters:
            output_points:(Point) Name of the output file with the created points
            number_points:(Long) Number of random points to be created
            constraining_area:(Polygon) Mayor constraining area 
            constraining_rasters:(Multiband Raster) Information rasters, the selection area will be constrained to 
                zones with finite values. Empty for no restriction.
            buffer_points:(Point) The selection of points will maintain a minimum/maximum distance to these points.
                Empty for no restriction.
            buffer_distance:(Linear Unit) Size of the buffer area around buffer points. Empty for no restriction.
            select_inside:(Boolean) Choose If the created points will be inside (TRUE) or outside (False) the buffer 
                area. Ignored if no points are given.
            minimum_distance:(Linear Unit) Minimum distance between points. Distance of 0 if empty
    """
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Create Random Points"
        self.description = "Creates random points far from selected points inside areas with information"
        self.canRunInBackground = False
        self.category = "Preprocessing"

    def getParameterInfo(self):

        output_points = arcpy.Parameter(
            displayName="Output Points Feature Class",
            name="output_points",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Output")
        output_points.filter.list = ["Point", "Multipoint"]

        number_points = arcpy.Parameter(
            displayName="Number of Random Points",
            name="number_points",
            datatype="GPLong",
            parameterType="Required",
            direction="Input")

        constraining_area = arcpy.Parameter(
            displayName="Constraining Area",
            name="constraining_area",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")
        constraining_area.filter.list = ["Polygon"]

        constraining_rasters = arcpy.Parameter(
            displayName="Constraining Rasters",
            name="constraining_rasters",
            datatype="GPRasterLayer",
            parameterType="Optional",
            direction="Input")


        buffer_points = arcpy.Parameter(
            displayName="Buffer Points",
            name="buffer_points",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input")
        buffer_points.filter.list = ["Point", "Multipoint"]

        buffer_distance = arcpy.Parameter(
            displayName="Buffer Distance",
            name="buffer_distance",
            datatype="GPLinearUnit",
            parameterType="Optional",
            direction="Input",
            enabled=False)

        minimum_distance = arcpy.Parameter(
            displayName="Minimum Distance Between Points",
            name="minimum_distance",
            datatype="GPLinearUnit",
            parameterType="Optional",
            direction="Input")

        select_inside = arcpy.Parameter(
            displayName="Select Inside Buffer",
            name="select_inside",
            datatype="Boolean",
            parameterType="Optional",
            direction="Input",
            enabled=False)
        select_inside.value = False

        params = [output_points, number_points, constraining_area, constraining_rasters, buffer_points,
                  buffer_distance, select_inside, minimum_distance]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        parameter_dic = {par.name: par for par in parameters}
        # Activate buffer_distance and select_inside just if buffer_points has value
        parameter_dic["buffer_distance"].enabled = (parameter_dic["buffer_points"].value is not None)
        parameter_dic["select_inside"].enabled = (parameter_dic["buffer_points"].value is not None)

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        return general.execute_tool(arcsdm.CreateRandomPoints.execute, self, parameters, messages)


class EnrichPoints(object):
    """ 
        Enrich Points Points tool
            Labels two sets of points with values 1, -1, merges them in a single feature, assigns them the value of 
                the given rasters and delete/impute missing data
            Parameters:
                class1_points: (Points) Points of labelled class 1 (Usually deposits) (Empty for no points of this class)
                class2_points: (Points) Points of labelled class -1 (Usually non-deposits) (Empty for no points of this 
                    class)
                field_name: (String) Name of the class field to be assigned 
                copy_data: (Boolean) Select if the previously existent information of the points is kept (True) or 
                    omitted (False) 
                information_rasters: (Multiband Raster) Raster with information to be assigned to the points
                missing_mask: Number to be imputed to missing values (Empty to delete missing values) 
                output: (Points) Name of the output file 
        """
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Enrich Points"
        self.description = 'Adds data to the attribute table of the underlying rasters as well as mark them as ' \
                           'Prospective or not and replaces/deletes missing data'
        self.canRunInBackground = False
        self.category = "Preprocessing"

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
            datatype="GPRasterLayer",
            parameterType="Optional",
            direction="Input")

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
            displayName="Class Field Name",
            name="field_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        field_name.value = "Deposit"

        copy_data = arcpy.Parameter(
            displayName="Copy Fields From Points",
            name="copy_data",
            datatype="Boolean",
            parameterType="Optional",
            direction="Input")
        copy_data.value = True

        params = [class1_points, class2_points, field_name, copy_data, information_rasters, missing_mask, output]
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
        return general.execute_tool(arcsdm.EnrichPoints.execute, self, parameters, messages)


class AdaboostBestParameters(object):
    """     WARNING: Not maintained tool, unexpected errors are likely  :WARNING
        Adaboost Best Parameters tool
            Makes a grid search with the best parameters for the Adaboost model, outputs a table with the accuracy 
                scores for each test and the respective plot
                 
            Parameters:
                train_points: (Points) Points to be used as training points 
                train_regressors: (Field) Fields with the names of fields to be used as regressors 
                train_response: (Field) Field with the name of the response or class  
                num_estimators_min: (Integer) Minimum number of estimators 
                num_estimators_max: (Integer) Maximum number of estimators  
                num_estimators_increment: (Integer) Step-size of the number of estimators 
                learning_rate_min: (Float) Minimum learning rate 
                learning_rate_max: (Float) Maximum learning rate 
                learning_rate_increment: (Float) increment  
                output_table: Name of the table to be output  
                plot_file: Name of the file with the plots
      
    """
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
            displayName="Predictor Fields",
            name="train_regressors",
            datatype="Field",
            parameterType="Required",
            direction="Input",
            multiValue=True)
        train_regressors.parameterDependencies = [train_points.name]
        train_regressors.filter.list = ['Short', 'Long', 'Double', 'Float', 'Single']

        train_response = arcpy.Parameter(
            displayName="Class Field",
            name="train_response",
            datatype="Field",
            parameterType="Required",
            direction="Input")
        train_response.parameterDependencies = [train_points.name]
        train_response.filter.list = ['Short', 'Long', 'Double', 'Float', 'Single']

        num_estimators_min = arcpy.Parameter(
            displayName="Minimum Number of Estimators",
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
            displayName="Increment of Number of Estimators",
            name="num_estimators_increment",
            datatype="GPLong",
            parameterType="Required",
            direction="Input",
            category="Number of estimators")
        num_estimators_increment.value = 2

        learning_rate_min = arcpy.Parameter(
            displayName="Minimum Learning Rate",
            name="learning_rate_min",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="Learning Rate")
        learning_rate_min.value = 0.5

        learning_rate_max = arcpy.Parameter(
            displayName="Maximum Learning Rate",
            name="learning_rate_max",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
            category="Learning Rate")
        learning_rate_max.value = 1.5

        learning_rate_increment = arcpy.Parameter(
            displayName="Increment of Learning Rate",
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
        # Force the output files to the adequate extension
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
            elif parameters[5].value <= 0 and (parameters[4].value > parameters[3].value):
                parameters[5].setErrorMessage("Non positive values forbidden")

        if any([parameters[x].altered for x in xrange(6, 9)]):
            if parameters[6].value > parameters[7].value:
                parameters[6].setErrorMessage("Minimum value greater than maximum")
            elif (parameters[7].value - parameters[6].value) < parameters[8].value:
                parameters[8].setWarningMessage("Increment greater than the interval")
            elif parameters[6].value <= 0:
                parameters[6].setErrorMessage("Non positive values forbidden")
            elif parameters[8].value <= 0 and (parameters[7].value > parameters[6].value):
                parameters[8].setErrorMessage("Non positive values forbidden")
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        return general.execute_tool(arcsdm.AdaboostBestParameters.execute, self, parameters, messages)


class AdaboostTrain(object):
    """
        AdaBoost Train Tool
            Creates and trains a model using adaboost
            Parameters:
                train_points: (Points) Points that will be used for the training 
                train_regressors: (Field) Name of the regressors fields that will be used for the training 
                train_response: (Field) Name of the response/class field that will be used for the training 
                num_estimators: (Integer) Number of estimators to be used 
                learning_rate: (Float) Learning rate of the model 
                output_model: (File path) Name of the file where the model will be stored
                leave_one_out: (Boolean) Choose between test with leave-one-out (true) or 3-fold cross-validation (false)  
                classifier_name: (String) Adaptor parameter for further calculations, value always has to be "AdaBoost"
                          
        For more information about the model visit 
        http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
    """
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Adaboost Train"
        self.description = 'Trains a classifier using Adaboost'
        self.canRunInBackground = False
        self.category = "Modelling"

    def getParameterInfo(self):

        # TODO: Move repeated parameters to an general function (train_points, train_regressors, train_response,
        #       output_model, leave_one_out, classifier_name)
        train_points = arcpy.Parameter(
            displayName="Train Points",
            name="train_points",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")
        train_points.filter.list = ["Point", "Multipoint"]

        train_regressors = arcpy.Parameter(
            displayName="Predictor Fields",
            name="train_regressors",
            datatype="Field",
            parameterType="Required",
            direction="Input",
            multiValue=True)
        train_regressors.parameterDependencies = [train_points.name]
        train_regressors.filter.list = ['Short', 'Long', 'Double', 'Float', 'Single']

        train_response = arcpy.Parameter(
            displayName="Class Field",
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
        learning_rate.filter.list = [0.0000000000001, 10]

        output_model = arcpy.Parameter(
            displayName="Output Model",
            name="output_model",
            datatype="DEFile",
            parameterType="Optional",
            direction="Output")

        leave_one_out = arcpy.Parameter(
            displayName="Leave-one-out Cross Validation",
            name="leave_one_out",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input")
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

        # TODO: Move all this to a general function
        # Enforce file extension
        output_model = parameters[5]
        if output_model.altered:
            if not output_model.valueAsText.endswith(".pkl"):
                output_model.value = output_model.valueAsText + ".pkl"

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        # Check if the response field is not included in the regressors
        # TODO: Move all this to a general function
        train_regressors = parameters[1]
        train_response = parameters[2]
        if (train_response.altered or train_regressors.altered) and train_regressors.valueAsText is not None:
            for field in train_regressors.valueAsText.split(";"):
                if field == train_response.valueAsText:
                    train_response.setErrorMessage("{} can not be included in {}".format(train_response.displayName,
                                                                                         train_regressors.displayName))

        return

    def execute(self, parameters, messages):
        """The source code of the tool."""

        return general.execute_tool(arcsdm.ModelTrain.execute, self, parameters, messages)


class ModelValidation(object):
    """
        Model validation Tool
            Performs evaluation test of the response map and outputs the results
            Parameters:
                classification_model: (Raster) Response function made by a model 
                test_points: (Points) Points to be used for the evaluation 
                test_response: (Field) Name of the field that contains real classification 
                threshold: Threshold point to differentiate prospective from non-prospective. Under this amount is 
                    considered non-prospective and over this is considered prospective 
                plot_file: (Path) Name of the file where the plots will be created. (Empty for no output) 
    """
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Model Validation"
        self.description = 'Validates a classification model with an independent test set'
        self.canRunInBackground = False

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
            displayName="Class Fields",
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

        return general.execute_tool(arcsdm.ModelValidation.execute, self, parameters, messages)


class ApplyModel(object):
    """
        Apply Model tool
            Generates the response map from a previously trained model
            Parameters:
                input_model:(.pkl file) Pickled file created by one of the train tools 
                information_rasters: (Multiband rasters) Rasters  to be used to generate the response, they must be the 
                    same as the ones for the training 
                output_map: (Raster) Name of the output response raster 
    """
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Apply model"
        self.description = 'Applies a model to a series of data rasters obtain a response raster'
        self.canRunInBackground = False
        self.category = "Modelling"

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
            direction="Input")

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

        return general.execute_tool(arcsdm.ApplyModel.execute, self, parameters, messages)


class MulticlassSplit(object):
    """
        Multi-Class Split tool
            Generate distance maps with the distance to each set of polygons grouped by a certain field. Additionally 
            apply transformations to the values
            Parameters: 
                input_feature: (Polygon) Feature class with the polygons to be considered 
                class_field: (Field) Field that classifies the polygons into groups, the minimum distance to the 
                    polygon of that class will be calculated 
                output_prefix: (path) Prefix to be given to each of the output rasters. The postfix will be given by 
                    the value in the class_field
                transformation: (string) Transformation to be applied to the resultant distance.
                max_distance: (Float) All the values greater the max_distance will be assigned to max_distance
    """
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Multiclass to Binary"
        self.description = 'Form a multi-class polygon creates one raster per class'
        self.canRunInBackground = False
        self.category = "Preprocessing"

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
        transformation.filter.list = ["Distance", "Inverse Distance", "Inverse Linear Distance", "Binary", "Logarithmic"]

        max_distance = arcpy.Parameter(
            displayName="Maximum distance",
            name="max_distance",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input")

        params = [input_feature, class_field, output_prefix, transformation, max_distance]
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
        return general.execute_tool(arcsdm.MulticlassSplit.execute, self, parameters, messages)


class ApplyFilter(object):
    """ Warning: Tool not maintained
        Apply filter tool
            Applies a filter to any raster
            Parameters:
                input_raster: (Raster) Raster to be filtered 
                output_raster: (Raster) Name of the filter to be output after filtering 
                filter_type: (String) Type of filter to be applied 
                filter_size: (Integer) Number of cells for the kernel of the filter
    """
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
        return general.execute_tool(arcsdm.ApplyFilter.execute, self, parameters, messages)


class LogisticRegressionTrain(object):
    """
        Logistic Regression Train Tool
            Creates and trains a model using Logistic regression
            Parameters:
                train_points: (Points) Points that will be used for the training 
                train_regressors: (Field) Name of the regressors fields that will be used for the training 
                train_response: (Field) Name of the response/class field that will be used for the training 
                output_model: (File path) Name of the file where the model will be stored
                leave_one_out: (Boolean) Choose between test with leave-one-out (true) or 3-fold cross-validation (false)  
                classifier_name: (String) Adaptor parameter for further calculations, value always has to be 
                    "Logistic Regression"
                deposit_weight: (Integer) weight to be given to the deposits to deal with unbalanced data 
                penalty: (string) type of norm for the penalty 
                random_state: (Integer) seed for random generator, useful to obtain reproducible results 
                
        For more information about the model visit 
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Logistic Regression Train"
        self.description = 'Trains a classifier using Logistic regression'
        self.canRunInBackground = False
        self.category = "Modelling"

    def getParameterInfo(self):

        # TODO: Move repeated parameters to an general function (train_points, train_regressors, train_response,
        #       output_model, leave_one_out, classifier_name)
        train_points = arcpy.Parameter(
            displayName="Train Points",
            name="train_points",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")
        train_points.filter.list = ["Point", "Multipoint"]

        train_regressors = arcpy.Parameter(
            displayName="Predictor Fields",
            name="train_regressors",
            datatype="Field",
            parameterType="Required",
            direction="Input",
            multiValue=True)
        train_regressors.parameterDependencies = [train_points.name]
        train_regressors.filter.list = ['Short', 'Long', 'Double', 'Float', 'Single']

        train_response = arcpy.Parameter(
            displayName="Class Field",
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
            displayName="Leave-one-out Cross Validation",
            name="leave_one_out",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input")
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
            displayName="Penalty Norm",
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
        # TODO: Move all this to a general function
        parameter_dic = {par.name: par for par in parameters}
        output_model = parameter_dic["output_model"]

        # Enforce file extension
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

        # Check if the response field is not include in the regressors
        if (train_response.altered or train_regressors.altered) and train_regressors.valueAsText is not None:
            for field in train_regressors.valueAsText.split(";"):
                if field == train_response.valueAsText:
                    train_response.setErrorMessage("{} can not be included in {}".format(train_response.displayName,
                                                                                         train_regressors.displayName))

        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        return general.execute_tool(arcsdm.ModelTrain.execute, self, parameters, messages)


class BrownBoostTrain(object):
    """
         BrownBoost Train Tool
            Creates and trains a model using BrownBoost
            Parameters:
                train_points: (Points) Points that will be used for the training 
                train_regressors: (Field) Name of the regressors fields that will be used for the training 
                train_response: (Field) Name of the response/class field that will be used for the training 
                output_model: (File path) Name of the file where the model will be stored
                leave_one_out: (Boolean) Choose between test with leave-one-out (true) or 3-fold cross-validation (false)  
                classifier_name: (String) Adaptor parameter for further calculations, value always has to be "Brownboost"
                
                countdown: (Float) Initial value of the countdown timer
    """
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "BrownBoost Train"
        self.description = 'Trains a classifier using BrownBoost'
        self.canRunInBackground = False
        self.category = "Modelling"

    def getParameterInfo(self):

        # TODO: Move repeated parameters to an general function (train_points, train_regressors, train_response,
        #       output_model, leave_one_out, classifier_name)
        train_points = arcpy.Parameter(
            displayName="Train Points",
            name="train_points",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")
        train_points.filter.list = ["Point", "Multipoint"]

        train_regressors = arcpy.Parameter(
            displayName="Predictor Fields",
            name="train_regressors",
            datatype="Field",
            parameterType="Required",
            direction="Input",
            multiValue=True)
        train_regressors.parameterDependencies = [train_points.name]
        train_regressors.filter.list = ['Short', 'Long', 'Double', 'Float', 'Single']

        train_response = arcpy.Parameter(
            displayName="Class Field",
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
        countdown.filter.list = [0.0000000000001, 10]

        output_model = arcpy.Parameter(
            displayName="Output Model",
            name="output_model",
            datatype="DEFile",
            parameterType="Optional",
            direction="Output")

        leave_one_out = arcpy.Parameter(
            displayName="Leave-one-out Cross Validation",
            name="leave_one_out",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input")
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
        # TODO: Move all this to a general function
        parameter_dic = {par.name: par for par in parameters}
        output_model = parameter_dic["output_model"]

        if output_model.altered:
            if not output_model.valueAsText.endswith(".pkl"):
                output_model.value = output_model.valueAsText + ".pkl"

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        # TODO: Move all this to a general function
        parameter_dic = {par.name: par for par in parameters}
        train_regressors = parameter_dic["train_regressors"]
        train_response = parameter_dic["train_response"]

        # Check if the response field is not include in the regressors
        if (train_response.altered or train_regressors.altered) and train_regressors.valueAsText is not None:
            for field in train_regressors.valueAsText.split(";"):
                if field == train_response.valueAsText:
                    train_response.setErrorMessage("{} can not be included in {}".format(train_response.displayName,
                                                                                         train_regressors.displayName))

    def execute(self, parameters, messages):
        """The source code of the tool."""
        return general.execute_tool(arcsdm.ModelTrain.execute, self, parameters, messages)


class SVMTrain(object):
    """
        Support Vector Machine Train Tool
            Creates and trains a model using Support Vector Machine
            Parameters:
                train_points: (Points) Points that will be used for the training 
                train_regressors: (Field) Name of the regressors fields that will be used for the training 
                train_response: (Field) Name of the response/class field that will be used for the training 
                output_model: (File path) Name of the file where the model will be stored
                leave_one_out: (Boolean) Choose between test with leave-one-out (true) or 3-fold cross-validation (false)  
                classifier_name: (String) Adaptor parameter for further calculations, value always has to be "SVM"
                
                kernel: (String) Kernel to be used  
                deposit_weight: (Integer) weight to be given to the deposits to deal with unbalanced data 
                penalty: (string) type of norm for the penalty 
                random_state:(Integer) seed for random generator, useful to obtain reproducible results 
                normalize: (Boolean) Indicates if the data needs to be normalized (True) or not (False). Notice that 
                    SVM is sensitive linear transformations  
                    
        For more information about the model visit 
        http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Support Vector Machine Train"
        self.description = 'Trains a classifier using SVM'
        self.canRunInBackground = False
        self.category = "Modelling"

    def getParameterInfo(self):

        # TODO: Move repeated parameters to an general function (train_points, train_regressors, train_response,
        #       output_model, leave_one_out, classifier_name)
        train_points = arcpy.Parameter(
            displayName="Train Points",
            name="train_points",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")
        train_points.filter.list = ["Point", "Multipoint"]

        train_regressors = arcpy.Parameter(
            displayName="Predictor Fields",
            name="train_regressors",
            datatype="Field",
            parameterType="Required",
            direction="Input",
            multiValue=True)
        train_regressors.parameterDependencies = [train_points.name]
        train_regressors.filter.list = ['Short', 'Long', 'Double', 'Float', 'Single']

        train_response = arcpy.Parameter(
            displayName="Class Field",
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
            displayName="Leave-one-out Cross Validation",
            name="leave_one_out",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input")
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
            displayName="Penalty Parameter",
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
            displayName="Normalize Data",
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
        # TODO: Move all this to a general function
        parameter_dic = {par.name: par for par in parameters}
        output_model = parameter_dic["output_model"]

        # Enforce file extension
        if output_model.altered:
            if not output_model.valueAsText.endswith(".pkl"):
                output_model.value = output_model.valueAsText + ".pkl"

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        # TODO: Move all this to a general function
        parameter_dic = {par.name: par for par in parameters}
        train_regressors = parameter_dic["train_regressors"]
        train_response = parameter_dic["train_response"]

        # Check if the response field is not include in the regressors
        if (train_response.altered or train_regressors.altered) and train_regressors.valueAsText is not None:
            for field in train_regressors.valueAsText.split(";"):
                if field == train_response.valueAsText:
                    train_response.setErrorMessage("{} can not be included in {}".format(train_response.displayName,
                                                                                         train_regressors.displayName))

        return

    def execute(self, parameters, messages):
        """The source code of the tool."""

        return general.execute_tool(arcsdm.ModelTrain.execute, self, parameters, messages)


class RFTrain(object):
    """
        Random Forest Train Tool
            Creates and trains a model using Random Forest
            Parameters:
                train_points: (Points) Points that will be used for the training 
                train_regressors: (Field) Name of the regressors fields that will be used for the training 
                train_response: (Field) Name of the response/class field that will be used for the training 
                output_model: (File path) Name of the file where the model will be stored
                leave_one_out: (Boolean) Choose between test with leave-one-out (true) or 3-fold cross-validation (false)  
                classifier_name: (String) Adaptor parameter for further calculations, value always has to be 
                    "Random Forest"
                
                num_estimators: (Integer) Number of trees to be trained 
                max_depth: (Integer) max depth of the trained trees 
                deposit_weight: (Integer) weight to be given to the deposits to deal with unbalanced data 
                random_state:(Integer) seed for random generator, useful to obtain reproducible results 

        For more information about the model visit 
        http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Random Forest Train"
        self.description = 'Trains a classifier using Random Forest'
        self.canRunInBackground = False
        self.category = "Modelling"

    def getParameterInfo(self):

        # TODO: Move repeated parameters to an general function (train_points, train_regressors, train_response,
        #       output_model, leave_one_out, classifier_name)
        train_points = arcpy.Parameter(
            displayName="Train Points",
            name="train_points",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")
        train_points.filter.list = ["Point", "Multipoint"]

        train_regressors = arcpy.Parameter(
            displayName="Predictor Fields",
            name="train_regressors",
            datatype="Field",
            parameterType="Required",
            direction="Input",
            multiValue=True)
        train_regressors.parameterDependencies = [train_points.name]
        train_regressors.filter.list = ['Short', 'Long', 'Double', 'Float', 'Single']

        train_response = arcpy.Parameter(
            displayName="Class Field",
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
            displayName="Leave-one-out Cross Validation",
            name="leave_one_out",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input")
        leave_one_out.value = True

        classifier_name = arcpy.Parameter(
            displayName="classifier name",
            name="classifier_name",
            datatype="GPString",
            parameterType="Derived",
            direction="Output")
        classifier_name.value = "Random Forest"

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

        num_estimators = arcpy.Parameter(
            displayName="Number of Estimators",
            name="num_estimators",
            datatype="GPLong",
            parameterType="Required",
            direction="Input")
        num_estimators.value = 10
        num_estimators.filter.type = "Range"
        num_estimators.filter.list = [1, 1000]

        max_depth = arcpy.Parameter(
            displayName="Max Depth",
            name="max_depth",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input")

        params = [train_points, train_regressors, train_response, num_estimators, max_depth, deposit_weight,
                  output_model, leave_one_out, classifier_name, random_state]
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        # TODO: Move all this to a general function
        parameter_dic = {par.name: par for par in parameters}
        output_model = parameter_dic["output_model"]

        # Enforce file extension
        if output_model.altered:
            if not output_model.valueAsText.endswith(".pkl"):
                output_model.value = output_model.valueAsText + ".pkl"

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        # TODO: Move all this to a general function
        parameter_dic = {par.name: par for par in parameters}
        train_regressors = parameter_dic["train_regressors"]
        train_response = parameter_dic["train_response"]

        # Check if the response field is not include in the regressors
        if (train_response.altered or train_regressors.altered) and train_regressors.valueAsText is not None:
            for field in train_regressors.valueAsText.split(";"):
                if field == train_response.valueAsText:
                    train_response.setErrorMessage("{} can not be included in {}".format(train_response.displayName,
                                                                                         train_regressors.displayName))

        return

    def execute(self, parameters, messages):
        """The source code of the tool."""

        return general.execute_tool(arcsdm.ModelTrain.execute, self, parameters, messages)

class SelectRandomPoints(object):
    """
         Select Random Points tool
            Makes an aleatory selection of points from a feature and saves them, as well the non-selected points can be 
                saved
            Parameters: 
                points: Set of points to make the selection
                selection_percentage: percentage of points to be in the selection,
                selected_points: (path) Name of the feature that will contain the selected points
                non_selected_points: (path) Name of the feature that will contain the non-selected points. Empty for 
                    not output
    """
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Select Random Points"
        self.description = "Selects given proportion of points of a given data layer"
        self.canRunInBackground = False
        self.category = "Preprocessing"

    def getParameterInfo(self):
        """Define parameter definitions"""
        points = arcpy.Parameter(
            displayName="Points",
            name="points",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")
        points.filter.list = ["Point", "Multipoint"]

        selection_percentage = arcpy.Parameter(
            displayName="Selection Percentage",
            name="selection_percentage",
            datatype="GPLong",
            parameterType="Required",
            direction="Input")
        selection_percentage.value = 10
        selection_percentage.filter.type = "Range"
        selection_percentage.filter.list = [1, 99]

        selected_points = arcpy.Parameter(
            displayName="Selected Points",
            name="selected_points",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Output")
        selected_points.filter.list = ["Point", "Multipoint"]

        non_selected_points = arcpy.Parameter(
            displayName="Non Selected Points",
            name="non_selected_points",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Output")
        non_selected_points.filter.list = ["Point", "Multipoint"]

        params = [points,selection_percentage,selected_points,non_selected_points]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        parameter_dic = {par.name: par for par in parameters}

        selected_points = parameter_dic["selected_points"]
        non_selected_points = parameter_dic["non_selected_points"]

        # Avoid Both outputs with the same name
        if selected_points is not  None and non_selected_points is not None and selected_points==non_selected_points:
            selected_points.setErrorMessage("Both outputs have the same name")
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
            Select random Points tool
            Selects randomly a subset of points and saves them in a feature class, additionally,
             it can save the non-selected points 
            :param parameters: parameters object with all the parameters from the python-tool. It contains:
                points: universe of points to make the selection
                selection_percentage: Percentage of the universe to be selected 
                selected_points: Name of the feature with the selected points
                non_selected_points: Name of the feature with the non-selected points
            :param messages: Messages object given by the tool 
        :return: 
        """

        parameter_dic = {par.name: par for par in parameters}

        points = parameter_dic["points"].valueAsText
        selection_percentage = int(parameter_dic["selection_percentage"].value)
        selected_points = parameter_dic["selected_points"].valueAsText
        non_selected_points = parameter_dic["non_selected_points"].valueAsText


        if selection_percentage > 100:
            print ("percent is greater than 100")
            return
        if selection_percentage < 0:
            print ("percent is less than zero")
            return

        import random
        fc = arcpy.Describe(points).catalogPath
        featureCount = float(arcpy.GetCount_management(fc).getOutput(0))
        count = int(featureCount * float(selection_percentage) / float(100))
        if not count:
            arcpy.SelectLayerByAttribute_management(points, "CLEAR_SELECTION")
            return
        oids = [oid for oid, in arcpy.da.SearchCursor(fc, "OID@")]
        oidFldName = arcpy.Describe(points).OIDFieldName
        delimOidFld = arcpy.AddFieldDelimiters(points, oidFldName)
        randOids = random.sample(oids, count)
        oidsStr = ", ".join(map(str, randOids))
        sql = "{0} IN ({1})".format(delimOidFld, oidsStr)

        arcpy.MakeFeatureLayer_management(fc, "selection_lyr")
        arcpy.SelectLayerByAttribute_management("selection_lyr", "NEW_SELECTION", sql)
        arcpy.CopyFeatures_management("selection_lyr", selected_points)
        if non_selected_points is not None:
            arcpy.SelectLayerByAttribute_management("selection_lyr", "SWITCH_SELECTION", )
            arcpy.CopyFeatures_management("selection_lyr", non_selected_points)

        return
