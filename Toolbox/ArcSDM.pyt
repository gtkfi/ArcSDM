import arcpy
import json
import os

import arcsdm.agterbergchengci
import arcsdm.areafrequency
import arcsdm.calculateresponse_arcpy_wip
import arcsdm.calculateresponse
import arcsdm.calculateweights
import arcsdm.categoricalreclass
from arcsdm.mlp import mlp_classification
from arcsdm.mlp import mlp_regression
import arcsdm.pca
import arcsdm.roctool
import arcsdm.splitting
import arcsdm.thinning
import arcsdm.wofe_common

from arcsdm.common import execute_tool

from arcsdm.machine_learning.mlp_common import (
    ACTIVATION_LINEAR,
    ACTIVATION_RELU,
    ACTIVATION_SIGMOID,
    ACTIVATION_SOFTMAX,
    ACTIVATION_TANH,
    LOSS_HUBER,
    LOSS_L1,
    LOSS_MSE,
    OPTIMIZER_ADAGRAD,
    OPTIMIZER_ADAM,
    OPTIMIZER_RMSPROP,
    OPTIMIZER_SGD,
    VALIDATION_ACCURACY,
    VALIDATION_F1,
    VALIDATION_L1,
    VALIDATION_MSE,
    VALIDATION_PRECISION,
    VALIDATION_R2,
    VALIDATION_RECALL,
    VALIDATION_RMSE
)


# Toolsets and sub-toolsets within ArcSDM toolbox
TS_EXPLORATORY_DATA_ANALYSIS = "Exploratory Data Analysis"
TS_PREPROCESSING = "Preprocessing"
TS_PREDICTIVE_MODELING = "Predictive Modeling"
TS_MLP = "Multilayer Perceptron"
TS_CLASSIFIER_TESTING = "Classifier Testing"
TS_RASTER_PROCESSING = "Raster Processing"
TS_REGRESSOR_TESTING = "Regressor Testing"
TS_CLASSIFIER_APPLICATION = "Classifier Application"
TS_REGRESSOR_APPLICATION = "Regressor Application"
TS_WOFE = "Weights of Evidence"
TS_VALIDATION = "Validation"
TS_VECTOR_PROCESSING = "Vector Processing"


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the .pyt file)."""
        self.label = "ArcSDM Tools"
        self.alias = "ArcSDM"
        self.tools = [
            AgterbergChengCITest,
            AreaFrequencyTable,
            CalculateResponse,
            CalculateWeights,
            ReclassAndFuzzify,
            GetSDMValues,
            PCARaster,
            PCAVector,
            PredictMLPClassifier,
            PredictMLPRegressor,
            ROCTool,
            SplitPoints,
            ThinPoints,
            TrainMLPClassifier,
            TrainMLPRegressor,
            ValidateMLPClassifier,
            ValidateMLPRegressor,
        ]


class GetSDMValues(object):
    def __init__(self):
        self.label = "Log WofE Details"
        self.description = "This tool is used to view details related to the the training site and study area for Weights of Evidence."
        self.canRunInBackground = True
        self.category = f"{TS_PREDICTIVE_MODELING}\\{TS_WOFE}"

    def getParameterInfo(self):
        param_training_sites_feature = arcpy.Parameter(
            displayName="Training sites",
            name="training_sites",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )

        param_unit_cell_area = arcpy.Parameter(
            displayName="Unit area (km2)",
            name="Unit_Area__sq_km_",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        param_unit_cell_area.value = "1"

        param_output_txt_file = arcpy.Parameter(
            displayName="Log results to a file",
            name="file_log",
            datatype="File",
            parameterType="Optional",
            direction="Output"
        )

        params = [param_training_sites_feature, param_unit_cell_area, param_output_txt_file]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed. This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        execute_tool(arcsdm.wofe_common.execute, self, parameters, messages)
        return


class AreaFrequencyTable(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Area Frequency Table"
        self.description = "Create a table for charting area of evidence classes vs number of training sites."
        self.canRunInBackground = False
        self.category = TS_EXPLORATORY_DATA_ANALYSIS

    def getParameterInfo(self):
        """Define parameter definitions"""
        param_training_sites = arcpy.Parameter(
            displayName="Training sites",
            name="training_sites",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )

        param_evidence_raster = arcpy.Parameter(
            displayName="Input Raster Layer",
            name="input_raster_layer",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input"
        )

        param_value_field = arcpy.Parameter(
            displayName="Value field",
            name="valuefield_name",
            datatype="Field",
            parameterType="Optional",
            direction="Input"
        )
        param_value_field.value = "VALUE"

        param_unit_cell_area = arcpy.Parameter(
            displayName="Unit area (km2)",
            name="Unit_Area__sq_km_",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )

        param_output_table = arcpy.Parameter(
            displayName="Output table",
            name="Output_Table",
            datatype="DEDbaseTable",
            parameterType="Required",
            direction="Output"
        )
        param_output_table.value = "%Workspace%\AreaFrequencyTable"

        params = [param_training_sites, param_evidence_raster, param_value_field, param_unit_cell_area, param_output_table]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed. This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        execute_tool(arcsdm.areafrequency.Execute, self, parameters, messages)
        return


class ROCTool(object):
    def __init__(self):
        self.label = "Calculate ROC Curves and AUC Values"
        self.description = "Calculates Receiver Operator Characteristic curves and Areas Under the Curves"
        self.category = TS_VALIDATION
        self.canRunInBackground = False

    def getParameterInfo(self):
        positives_param = arcpy.Parameter(
            displayName="Presence locations",
            name="positive_points",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")
        positives_param.filter.list = ["Point", "Multipoint"]

        negatives_param = arcpy.Parameter(
            displayName="Absence locations",
            name="negative_points",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input")
        negatives_param.filter.list = ["Point", "Multipoint"]

        models_param = arcpy.Parameter(
            displayName="Prediction raster(s)",
            name="model_rasters",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input",
            multiValue=True)

        folder_param = arcpy.Parameter(
            displayName="Destination Folder",
            name="dest_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input")
        folder_param.filter.list = ["File System"]

        return [positives_param, negatives_param, models_param, folder_param]

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False
        return True

    def execute(self, parameters, messages):
        execute_tool(arcsdm.roctool.execute, self, parameters, messages)
        return


class CalculateResponseNew(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Calculate Response (Experimental)"
        self.description = "Use this tool to combine the evidence weighted by their associated generalization in the weights-of-evidence table. This tool calculates the posterior probability, standard deviation (uncertainty) due to weights, variance (uncertainty) due to missing data, and the total standard deviation (uncertainty) based on the evidence and how the evidence is generalized in the associated weights-of-evidence tables.The calculations use the Weight and W_Std in the weights table from Calculate Weights."
        self.canRunInBackground = False
        self.category = f"{TS_PREDICTIVE_MODELING}\\{TS_WOFE}"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param_evidence_rasters = arcpy.Parameter(
            displayName="Input Raster Layer(s)",
            name="Input_evidence_raster_layers",
            datatype="GPValueTable",
            parameterType="Required",
            direction="Input"
        )
        param_evidence_rasters.columns = [['GPRasterLayer', 'Evidence raster']]

        param_weights_tables = arcpy.Parameter(
            displayName="Input weights tables",
            name="input_weights_tables",
            datatype="GPValueTable",
            parameterType="Required",
            direction="Input"
        )
        param_weights_tables.columns = [['DETable', 'Weights table']]

        param_training_sites_feature = arcpy.Parameter(
            displayName="Training sites",
            name="training_sites",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )

        param_ignore_missing_data = arcpy.Parameter(
            displayName="Ignore missing data",
            name="Ignore missing data",
            datatype="Boolean",
            parameterType="Optional",
            direction="Input"
        )

        param_nodata_value = arcpy.Parameter(
            displayName="Missing data value",
            name="Missing_Data_Value",
            datatype="GPLong",
            direction="Input"
        )
        param_nodata_value.value= -99

        param_unit_cell_area = arcpy.Parameter(
            displayName="Unit area (km^2)",
            name="Unit_Area_sq_km",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        param_unit_cell_area.value = "1"

        param_pprb_output = arcpy.Parameter(
            displayName="Output post probablity raster",
            name="Output_Post_Probability_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"
        )
        param_pprb_output.value = "%Workspace%\W_pprb"

        param_std_output = arcpy.Parameter(
            displayName="Output standard deviation raster",
            name="Output_Standard_Deviation_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"
        )
        param_std_output.value = "%Workspace%\W_std"

        param_md_variance_output = arcpy.Parameter(
            displayName="Output MD variance raster",
            name="output_md_variance_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"
        )
        param_md_variance_output.value = "%Workspace%\W_MDvar"

        param_total_stddev_output = arcpy.Parameter(
            displayName="Output Total Std Deviation Raster",
            name="output_total_std_dev_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"
        )
        param_total_stddev_output.value = "%Workspace%\W_Tstd"

        param_confidence_output = arcpy.Parameter(
            displayName="Output confidence raster",
            name="Output_Confidence_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"
        )
        param_confidence_output.value = "%Workspace%\W_conf"

        params = [
            param_evidence_rasters, # 0
            param_weights_tables, # 1
            param_training_sites_feature, # 2
            param_ignore_missing_data, # 3
            param_nodata_value, # 4
            param_unit_cell_area, # 5
            param_pprb_output, # 6
            param_std_output, # 7
            param_md_variance_output, # 8
            param_total_stddev_output, # 9
            param_confidence_output] # 10
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed. This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        execute_tool(arcsdm.calculateresponse_arcpy_wip.Execute, self, parameters, messages)
        return


class CalculateResponse(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Calculate Response"
        self.description = "Use this tool to combine the evidence weighted by their associated generalization in the weights-of-evidence table. This tool calculates the posterior probability, standard deviation (uncertainty) due to weights, variance (uncertainty) due to missing data, and the total standard deviation (uncertainty) based on the evidence and how the evidence is generalized in the associated weights-of-evidence tables.The calculations use the Weight and W_Std in the weights table from Calculate Weights."
        self.canRunInBackground = False
        self.category = f"{TS_PREDICTIVE_MODELING}\\{TS_WOFE}"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param_evidence_rasters = arcpy.Parameter(
            displayName="Input Raster Layer(s)",
            name="Input_evidence_raster_layers",
            datatype="GPValueTable",
            parameterType="Required",
            direction="Input"
        )
        param_evidence_rasters.columns = [['GPRasterLayer', 'Evidence raster']]

        param_weights_tables = arcpy.Parameter(
            displayName="Input weights tables",
            name="input_weights_tables",
            datatype="GPValueTable",
            parameterType="Required",
            direction="Input"
        )
        param_weights_tables.columns = [['DETable', 'Weights table']]

        param_training_sites_feature = arcpy.Parameter(
            displayName="Training sites",
            name="training_sites",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )

        param_ignore_missing_data = arcpy.Parameter(
            displayName="Ignore missing data",
            name="Ignore missing data",
            datatype="Boolean",
            parameterType="Optional",
            direction="Input"
        )

        param_nodata_value = arcpy.Parameter(
            displayName="Missing data value",
            name="Missing_Data_Value",
            datatype="GPLong",
            direction="Input"
        )
        param_nodata_value.value= -99

        param_unit_cell_area = arcpy.Parameter(
            displayName="Unit area (km^2)",
            name="Unit_Area_sq_km",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        param_unit_cell_area.value = "1"

        param_pprb_output = arcpy.Parameter(
            displayName="Output post probablity raster",
            name="Output_Post_Probability_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"
        )
        param_pprb_output.value = "%Workspace%\W_pprb"

        param_std_output = arcpy.Parameter(
            displayName="Output standard deviation raster",
            name="Output_Standard_Deviation_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"
        )
        param_std_output.value = "%Workspace%\W_std"

        param_md_variance_output = arcpy.Parameter(
            displayName="Output MD variance raster",
            name="output_md_variance_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"
        )
        param_md_variance_output.value = "%Workspace%\W_MDvar"

        param_total_stddev_output = arcpy.Parameter(
            displayName="Output Total Std Deviation Raster",
            name="output_total_std_dev_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"
        )
        param_total_stddev_output.value = "%Workspace%\W_Tstd"

        param_confidence_output = arcpy.Parameter(
            displayName="Output confidence raster",
            name="Output_Confidence_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"
        )
        param_confidence_output.value = "%Workspace%\W_conf"

        params = [
            param_evidence_rasters, # 0
            param_weights_tables, # 1
            param_training_sites_feature, # 2
            param_ignore_missing_data, # 3
            param_nodata_value, # 4
            param_unit_cell_area, # 5
            param_pprb_output, # 6
            param_std_output, # 7
            param_md_variance_output, # 8
            param_total_stddev_output, # 9
            param_confidence_output] # 10
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed. This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        execute_tool(arcsdm.calculateresponse.Execute, self, parameters, messages)
        return


class CalculateWeights(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Calculate Weights"
        self.description = "Calculate weight rasters from the inputs"
        self.canRunInBackground = True
        self.category = f"{TS_PREDICTIVE_MODELING}\\{TS_WOFE}"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param_evidence_raster = arcpy.Parameter(
            displayName="Evidence raster layer",
            name="evidence_raster_layer",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input"
        )

        param_codefield = arcpy.Parameter(
            displayName="Evidence raster codefield",
            name="Evidence_Raster_Code_Field",
            datatype="Field",
            parameterType="Optional",
            direction="Input"
        )
        param_codefield.filter.list = ["Text"]
        param_codefield.parameterDependencies = [param_evidence_raster.name]

        param_training_sites_feature = arcpy.Parameter(
            displayName="Training points feature",
            name="Training_points",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )

        param_weight_type = arcpy.Parameter(
            displayName="Type",
            name="Type",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        param_weight_type.filter.type = "ValueList"
        param_weight_type.filter.list = ["Descending", "Ascending", "Categorical"]
        param_weight_type.value = ""

        param_output_table = arcpy.Parameter(
            displayName="Output weights table",
            name="output_weights_table",
            datatype="DETable",
            parameterType="Required",
            direction="Output"
        )

        param_studentized_contrast_threshold = arcpy.Parameter(
            displayName="Confidence Level of Studentized Contrast",
            name="Confidence_Level_of_Studentized_Contrast",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        param_studentized_contrast_threshold.value = "2"

        param_unit_cell_area = arcpy.Parameter(
            displayName="Unit area (km2)",
            name="Unit_Area__sq_km_",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        param_unit_cell_area.value = "1"

        param_nodata_value = arcpy.Parameter(
            displayName="Missing data value",
            name="Missing_Data_Value",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        param_nodata_value.value = "-99"

        params = [
            param_evidence_raster, # 0
            param_codefield, # 1
            param_training_sites_feature, # 2
            param_weight_type, # 3
            param_output_table, # 4
            param_studentized_contrast_threshold, # 5
            param_unit_cell_area, # 6
            param_nodata_value # 7
        ]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed. This method is called whenever a parameter
        has been changed."""
        param_evidence_raster = parameters[0]
        param_weight_type = parameters[3]
        param_output_table = parameters[4]

        if param_evidence_raster.value and param_weight_type.value:
            if (param_evidence_raster.altered or param_weight_type.altered) and not param_output_table.altered:
                # Name the output table based on input layer and selected weight type
                layer = param_evidence_raster.valueAsText
                desc = arcpy.Describe(layer)
                name = desc.file
                weight_type = param_weight_type.valueAsText
                char = weight_type[:1]
                if (char != 'C'):
                    # Ascending or descending:  _C + first letter of type
                    char = 'C' + char
                else:
                        # Categorical
                        char = 'CT'
                # Update name accordingly
                default_output_name = "%WORKSPACE%\\" + name + "_" + char
                default_output_name = default_output_name.replace(".", "")
                # Add .dbf to Weights Table Name if Workspace is not File Geodatabase
                # If using GDB database, remove numbers and underscore from the beginning of the name (else block)
                if not ".gdb" in arcpy.env.workspace:
                    default_output_name = default_output_name + ".dbf"
                else:
                    wtsbase = os.path.basename(default_output_name)
                    while len(wtsbase) > 0 and (wtsbase[:1] <= "9" or wtsbase[:1] == "_"):
                        wtsbase = wtsbase[1:]
                    default_output_name = os.path.dirname(default_output_name) + "\\" + wtsbase
                param_output_table.value = default_output_name
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""

        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        execute_tool(arcsdm.calculateweights.Calculate, self, parameters, messages)
        return


class SplitPoints(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Split Points"
        self.description = "Split training sites into training and testing datasets based on a random percentage."
        self.canRunInBackground = False
        self.category = f"{TS_PREPROCESSING}\\{TS_VECTOR_PROCESSING}"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param_input_layer = arcpy.Parameter(
            displayName="Input points",
            name="training_sites_layer",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )

        param_random_percentage = arcpy.Parameter(
            displayName="Training fraction",
            name="random_percentage",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        param_random_percentage.filter.type = "Range"
        param_random_percentage.filter.list = [1, 99]

        param_output_layer = arcpy.Parameter(
            displayName="Output training layer",
            name="output_training_layer",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Output"
        )
        param_output_layer.value = "training_points"

        param_inverse_output_layer = arcpy.Parameter(
            displayName="Output testing layer (optional)",
            name="output_testing_layer",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Output"
        )
        param_inverse_output_layer.value = "testing_points"

        params = [param_input_layer, param_random_percentage, param_output_layer, param_inverse_output_layer]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool parameter."""
        if parameters[1].value and not (0 < parameters[1].value <= 100):
            parameters[1].setErrorMessage("Random percentage must be between 0 and 100.")
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        execute_tool(arcsdm.splitting.SplitSites, self, parameters, messages)
        return


class ThinPoints(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Thin Points"
        self.description = "Selects subset of the training points based on a thinning value and minimum distance."
        self.canRunInBackground = False
        self.category = f"{TS_PREPROCESSING}\\{TS_VECTOR_PROCESSING}"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param_input_layer = arcpy.Parameter(
            displayName="Input points",
            name="input_point_layer",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")

        param_unit_area = arcpy.Parameter(
            displayName="Unit area",
            name="unit_area_size",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input")
        param_unit_area.value = 500

        param_area_unit = arcpy.Parameter(
            displayName="Area Unit",
            name="area_unit",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        param_area_unit.filter.type = "ValueList"
        param_area_unit.filter.list = [
            "Square Kilometers",
            "Square Meters",
            "Square Miles",
            "Square Yards",
            "Square Feet",
            "Acres",
            "Hectares",
        ]
        param_area_unit.value = "Square Kilometers"

        param_min_distance = arcpy.Parameter(
            displayName="Minimum Distance (Meters)",
            name="min_distance_meters",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input")

        param_output = arcpy.Parameter(
            displayName="Output layer",
            name="output_layer",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Output")
        param_output.value = "thinned_points"

        params = [param_input_layer, param_unit_area, param_area_unit, param_min_distance, param_output]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool parameter."""
        if parameters[1].value is not None and parameters[1].value <= 0:
            parameters[1].setErrorMessage("Unit area must be greater than 0.")
        if parameters[3].value is not None and parameters[3].value <= 0:
            parameters[3].setErrorMessage("Minimum distance must be greater than 0.")
        return

    def execute(self, parameters, messages):
        """Execute the thinning tool."""
        execute_tool(arcsdm.thinning.ThinSites, self, parameters, messages)
        return


class ReclassAndFuzzify(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Reclass & Fuzzify"
        self.description = "Create fuzzy memberships for categorical data by first reclassification to integers and then division by an appropriate value."
        self.canRunInBackground = False
        self.category = f"{TS_PREPROCESSING}\\{TS_RASTER_PROCESSING}"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
        displayName="Categorical evidence raster",
        name="categorical_evidence",
        datatype="GPRasterLayer",
        parameterType="Required",
        direction="Input")

        param1 = arcpy.Parameter(
        displayName="Reclass field",
        name="reclass_field",
        datatype="Field",
        parameterType="Required",
        direction="Input")

        param2 = arcpy.Parameter(
        displayName="Reclassification",
        name="reclassification",
        datatype="remap",
        parameterType="Required",
        direction="Input")

        param3 = arcpy.Parameter(
        displayName="FM Categorical",
        name="fmcat",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output")

        param4 = arcpy.Parameter(
        displayName="Divisor",
        name="divisor",
        datatype="GPLong",
        parameterType="Required",
        direction="Input")

        param1.value = "VALUE"
        param1.enabled = False
        param2.enabled = False
        param1.parameterDependencies = [param0.name]
        param2.parameterDependencies = [param0.name,param1.name]

        params = [param0,param1,param2,param3,param4]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed. This method is called whenever a parameter
        has been changed."""
        if parameters[0].value:
            parameters[1].enabled = True
            parameters[2].enabled = True
        else:
            parameters[1].enabled = False
            parameters[2].enabled = False
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        execute_tool(arcsdm.categoricalreclass.Calculate, self, parameters, messages)
        return


class AgterbergChengCITest(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Agterberg-Cheng CI Test"
        self.description = "Perform the Agterberg-Cheng Conditional Independence test (Agterberg & Cheng 2002) on a mineral prospectivity map and save the results to a file."
        self.canRunInBackground = False
        self.category = f"{TS_PREDICTIVE_MODELING}\\{TS_WOFE}"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param_pprb_raster = arcpy.Parameter(
            displayName="Post Probability raster",
            name="pp_raster",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input"
        )

        param_pprb_std_raster = arcpy.Parameter(
            displayName="Probability Std raster",
            name="ps_raster",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input"
        )

        param_training_sites_feature = arcpy.Parameter(
            displayName="Training sites",
            name="training_sites",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )

        param_unit_cell_area = arcpy.Parameter(
        displayName="Unit area (km2)",
        name="Unit_Area__sq_km",
        datatype="GPDouble",
        parameterType="Required",
        direction="Input")

        param_output_ci_test_file = arcpy.Parameter(
        displayName="Output CI Test File",
        name="ci_test_file",
        datatype="DEFile",
        parameterType="Optional",
        direction="Output")

        params = [
            param_pprb_raster, # 0
            param_pprb_std_raster, # 1
            param_training_sites_feature, # 2
            param_unit_cell_area, # 3
            param_output_ci_test_file # 5
        ]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed. This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""

        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        execute_tool(arcsdm.agterbergchengci.Calculate, self, parameters, messages)
        return


class PCARaster(object):
    def __init__(self):
        """Principal Component Analysis (Raster)"""
        self.label = "Principal Component Analysis (Raster)"
        self.description = "Perform Principal Component Analysis on input rasters"
        self.canRunInBackground = False
        self.category = TS_EXPLORATORY_DATA_ANALYSIS

    def getParameterInfo(self):
        """Define parameter definitions"""

        # Input data parameter
        param_input_rasters = arcpy.Parameter(
            displayName="Input Raster Layer(s) (min. 2 bands)",
            name="input_rasters",
            datatype=["GPRasterLayer", "GPRasterDataLayer"],
            parameterType="Required",
            direction="Input",
            multiValue=True
        )

        param_num_components = arcpy.Parameter(
            displayName="Number of Components",
            name="num_components",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )

        param_scaler_type = arcpy.Parameter(
            displayName="Scaler Type",
            name="scaler_type",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        param_scaler_type.filter.list = ["standard", "min_max", "robust"]
        param_scaler_type.value = "standard"

        param_nodata_handling = arcpy.Parameter(
            displayName="Nodata Handling",
            name="nodata_handling",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        param_nodata_handling.filter.list = ["remove", "replace"]
        param_nodata_handling.value = "remove"

        param_transformed_data = arcpy.Parameter(
            displayName="Transformed Data",
            name="transformed_data",
            datatype="DETable",
            parameterType="Required",
            direction="Output"
        )
        param_transformed_data.value = 'PCA_scores_raster'

        params = [param_input_rasters,
                #   param_nodata_value,
                  param_num_components,
                  param_scaler_type,
                  param_nodata_handling,
                  param_transformed_data,
                ]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed. This method is called whenever a parameter
        has been changed."""
        if not parameters[4].altered:
            parameters[4].value = "PCA_scores_raster"

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        execute_tool(arcsdm.pca.Execute, self, parameters, messages)
        return


class PCAVector(object):
    def __init__(self):
        """Principal Component Analysis (Vector)"""
        self.label = "Principal Component Analysis (Vector)"
        self.description = "Perform Principal Component Analysis on input vectors"
        self.canRunInBackground = False
        self.category = TS_EXPLORATORY_DATA_ANALYSIS

    def getParameterInfo(self):
        """Define parameter definitions"""

        # Input data parameter
        param_input_vectors = arcpy.Parameter(
            displayName="Input Vector",
            name="input_vectors",
            datatype=["GPFeatureLayer"],
            parameterType="Required",
            direction="Input",
        )

        param_input_fields = arcpy.Parameter(
            displayName="Select Fields (min. 2)",
            name="input_fields",
            datatype="Field",
            parameterType="Required",
            direction="Input",
            multiValue=True
        )
        param_input_fields.parameterDependencies = [param_input_vectors.name]

        param_nodata_value = arcpy.Parameter(
            displayName="NoData Value",
            name="nodata_value",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )
        param_nodata_value.value = -99

        param_num_components = arcpy.Parameter(
            displayName="Number of Components",
            name="num_components",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )

        param_scaler_type = arcpy.Parameter(
            displayName="Scaler Type",
            name="scaler_type",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        param_scaler_type.filter.list = ["standard", "min_max", "robust"]
        param_scaler_type.value = "standard"

        param_nodata_handling = arcpy.Parameter(
            displayName="Nodata Handling",
            name="nodata_handling",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        param_nodata_handling.filter.list = ["remove", "replace"]
        param_nodata_handling.value = "remove"

        param_transformed_data = arcpy.Parameter(
            displayName="Transformed Data",
            name="transformed_data",
            datatype="DETable",
            parameterType="Required",
            direction="Output"
        )
        param_transformed_data.value = 'PCA_scores_vector'

        params = [param_input_vectors,
                  param_input_fields,
                  param_nodata_value,
                  param_num_components,
                  param_scaler_type,
                  param_nodata_handling,
                  param_transformed_data,
                ]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed. This method is called whenever a parameter
        has been changed."""
        if not parameters[6].altered:
            parameters[6].value = "PCA_scores_vector"

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""

        if parameters[1].value and parameters[1].value.rowCount < 2:
            parameters[1].setErrorMessage("Select Fields requires at least two fields.")
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        execute_tool(arcsdm.pca.Execute, self, parameters, messages)
        return



class TrainMLPClassifier:
    def __init__(self):
        self.label = "Train MLP Classifier"
        self.description = ""
        self.canRunInBackground = False
        self.category = f"{TS_PREDICTIVE_MODELING}\\{TS_MLP}"
        # Store parameter indices in order to have just one place to edit them if they ever change
        self.idx_X = 0
        self.idx_X_nodata_value = 1
        self.idx_X_standardize = 2
        self.idx_y = 3
        self.idx_y_attribute = 4
        self.idx_y_nodata_value = 5
        self.idx_hidden_layers = 6
        self.idx_hidden_activation = 7
        self.idx_last_activation = 8
        self.idx_epochs = 9
        self.idx_batch_size = 10
        self.idx_optimizer = 11
        self.idx_learning_rate = 12
        self.idx_is_early_stopping = 13
        self.idx_early_stopping_patience = 14
        self.idx_validation_split = 15
        self.idx_validation_data = 16
        self.idx_metrics = 17
        self.idx_random_state = 18
        self.idx_apply_smote = 19
        self.idx_n_synthetic_samples = 20
        self.idx_minority_class_label = 21
        self.idx_k_neighbors = 22
        self.idxs_hidden_smote_params = [
            self.idx_n_synthetic_samples,
            self.idx_minority_class_label,
            self.idx_k_neighbors
        ]
        self.idx_output_file = 23

    def getParameterInfo(self):
        (param_X,
            param_X_nodata_value,
            param_X_standardize,
            param_y,
            param_y_attribute,
            param_y_nodata_value
        ) = make_mlp_X_y_params()

        param_hidden_layers = make_mlp_hidden_layers_params()
        param_hidden_layer_activation = make_mlp_hidden_layer_activation_param()

        param_last_layer_activation = arcpy.Parameter(
            displayName="Last Layer Activation Function",
            name="last_layer_activation",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        param_last_layer_activation.filter.type = "ValueList"
        param_last_layer_activation.filter.list = [ACTIVATION_LINEAR, ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX]
        param_last_layer_activation.value = ACTIVATION_SIGMOID

        (param_validation_split,
            param_validation_data,
            param_epochs,
            param_batch_size,
            param_optimizer,
            param_learning_rate,
            param_is_early_stopping,
            param_early_stopping_patience,
            param_random_state,
            param_apply_smote,
            param_n_synthetic_samples,
            param_minority_class_label,
            param_k_neighbors
        ) = make_mlp_training_params()

        param_metrics = arcpy.Parameter(
            displayName="Validation Metrics",
            name="validation_metrics",
            datatype="GPString",
            parameterType="Optional",
            direction="Input")
        param_metrics.filter.type = "ValueList"
        param_metrics.filter.list = [VALIDATION_ACCURACY, VALIDATION_PRECISION, VALIDATION_RECALL, VALIDATION_F1]
        param_metrics.value = VALIDATION_ACCURACY

        param_output_model_filepath = arcpy.Parameter(
            displayName="Output Model File",
            name="output_file",
            datatype="DEFile",
            parameterType="Required",
            direction="Output")
        param_output_model_filepath.value = "classifier_model.pth"

        params = [
            param_X,  # 0
            param_X_nodata_value,  # 1
            param_X_standardize,  # 2
            param_y,  # 3
            param_y_attribute,  # 4
            param_y_nodata_value,  # 5
            param_hidden_layers,  # 6
            param_hidden_layer_activation,  # 7
            param_last_layer_activation,  # 8
            param_epochs,  # 9
            param_batch_size,  # 10
            param_optimizer,  # 11
            param_learning_rate,  # 12
            param_is_early_stopping,  # 13
            param_early_stopping_patience,  # 14
            param_validation_split,  # 15
            param_validation_data,  # 16
            param_metrics,  # 17
            param_random_state,  # 18
            param_apply_smote,  # 19
            param_n_synthetic_samples,  # 20
            param_minority_class_label,  # 21
            param_k_neighbors,  # 22
            param_output_model_filepath  # 23
        ]
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        # Enable y nodata field if any of the y layers is a raster
        # Enable y attribute field if any of the y layers has an attribute table
        y = parameters[self.idx_y]
        if y.value and not y.hasBeenValidated:
            try:
                contains_raster, has_attribute_table, more_than_one = check_mlp_y_conditionals(y)

                # Classifier allows more than one label file, but in that case they are all expected
                # to be feature layers, so nodata param is not applicable. Each feature will be considered
                # to be one target class, so no need for attribute param either.
                if more_than_one:
                    parameters[self.idx_y_nodata_value].enabled = False
                    parameters[self.idx_y_attribute].enabled = False
                else:
                    parameters[self.idx_y_nodata_value].enabled = contains_raster
                    parameters[self.idx_y_attribute].enabled = has_attribute_table
            except Exception:
                pass
        if not y.value:
            parameters[self.idx_y_nodata_value].enabled = False
            parameters[self.idx_y_attribute].enabled = False

        # Enable early stopping patience field if early stopping is selected
        parameters[self.idx_early_stopping_patience].enabled = parameters[self.idx_is_early_stopping].value

        # Enable SMOTE related params when SMOTE is selected
        apply_smote = parameters[self.idx_apply_smote]
        if apply_smote.value:
            for idx in self.idxs_hidden_smote_params:
                parameters[idx].enabled = True
        else:
            for idx in self.idxs_hidden_smote_params:
                parameters[idx].enabled = False

        # Properly initialize output path (cannot be done in getParameterInfo() because ArcGIS Pro doesn't
        # have access to the current project yet)
        output_file = parameters[self.idx_output_file]
        if output_file.value and not any(x in str(output_file.value) for x in ["\\", "/"]):
            try:
                home_folder = arcpy.mp.ArcGISProject("CURRENT").homeFolder
                filename = str(output_file.value)

                output_file.value = os.path.join(home_folder, filename)

            except Exception:
                pass

        return

    def updateMessages(self, parameters):
        # Validate geometry & amount of rasters of y
        param_y = parameters[self.idx_y]
        param_y.clearMessage()
        if param_y.value and not param_y.hasBeenValidated:
            try:
                y_text = param_y.valueAsText
                y_paths = y_text.split(";")
                y_paths_clean = [path.strip("'") for path in y_paths if path]

                contains_raster = False

                for path in y_paths_clean:
                    desc = arcpy.Describe(path)
                    data_type = desc.dataType
                    if data_type in ["RasterLayer", "RasterDataset", "RasterBand"] and not (contains_raster):
                        contains_raster = True
                    elif data_type in ["FeatureLayer", "FeatureClass", "ShapeFile"]:
                        if desc.shapeType != "Point":
                            param_y.setErrorMessage("At least one of the target label layers has non-point geometry, which is not supported.")
                            break

                if (len(y_paths_clean) > 1) and contains_raster:
                    param_y.setErrorMessage("Only one raster file is supported. If multiple target label layers are provided, they must be feature layers.")
            except Exception:
                pass

        # Validate dropout rate(s)
        param_hidden_layers = parameters[self.idx_hidden_layers]
        param_hidden_layers.clearMessage()

        for layer in param_hidden_layers.value:
            dropout_rate = get_mlp_hidden_layer_dropout_rate(layer)
            if dropout_rate is not None and ((dropout_rate < 0) or (dropout_rate >= 1)):
                param_hidden_layers.setErrorMessage(f"Invalid dropout rate {dropout_rate}. Must be between 0 and 1.")

        # Validate number of epochs
        param_epochs = parameters[self.idx_epochs]
        param_epochs.clearMessage()

        if param_epochs.value < 1:
            param_epochs.setErrorMessage(f"Invalid number of epochs {param_epochs.value}. Must be at least one.")

        # Validate validation split
        # TODO: make this conditional based on whether validation data is provided
        param_validation_split = parameters[self.idx_validation_split]
        param_validation_split.clearMessage()
        validation_split = param_validation_split.value

        if (validation_split <= 0) or (validation_split >= 1):
            param_validation_split.setErrorMessage(f"Invalid validation split {validation_split}. Must be between 0 and 1.")

        # Warn if output file already exists
        param_output_file = parameters[self.idx_output_file]
        param_output_file.clearMessage()
        output_file = param_output_file.valueAsText
        if output_file:
            if os.path.exists(output_file):
                param_output_file.setWarningMessage("Output model file already exists and will be overwritten.")

        return

    def execute(self, parameters, messages):
        input_rasters = parameters[self.idx_X].valueAsText.split(";")
        target_labels = parameters[self.idx_y].valueAsText.split(";")
        hidden_layers = make_mlp_hidden_layer_specs(
            parameters[self.idx_hidden_layers].value,
            parameters[self.idx_hidden_activation].valueAsText
        )
        last_layer = parameters[self.idx_last_activation].valueAsText
        validation_data = parameters[self.idx_validation_data].valueAsText if parameters[self.idx_validation_data].value is not None else None
        apply_smote = parameters[self.idx_apply_smote].value
        smote_params = None
        if apply_smote:
            smote_params = (
                parameters[self.idx_n_synthetic_samples].value,
                parameters[self.idx_minority_class_label].value,
                parameters[self.idx_k_neighbors].value
            )

        mlp_classification.train_MLP_classifier(
            input_rasters=input_rasters,
            X_nodata_value=parameters[self.idx_X_nodata_value].value,
            standardize=parameters[self.idx_X_standardize].value,
            target_labels=target_labels,
            target_labels_attr=get_valueAsText_if_enabled(parameters[self.idx_y_attribute]),
            y_nodata_value=get_value_if_enabled(parameters[self.idx_y_nodata_value]),
            hidden_layers=hidden_layers,
            last_layer=last_layer,
            epochs=parameters[self.idx_epochs].value,
            batch_size=parameters[self.idx_batch_size].value,
            optimizer=parameters[self.idx_optimizer].valueAsText,
            learning_rate=parameters[self.idx_learning_rate].value,
            is_early_stopping=parameters[self.idx_is_early_stopping].value,
            early_stopping_patience=get_value_if_enabled(parameters[self.idx_early_stopping_patience]),
            validation_split=parameters[self.idx_validation_split].value,
            validation_data=validation_data,
            validation_metrics=parameters[self.idx_metrics].valueAsText,
            random_state=parameters[self.idx_random_state].value,
            apply_smote=apply_smote,
            smote_params=smote_params,
            output_model_file=parameters[self.idx_output_file].valueAsText
        )

    def postExecute(self, parameters):
        return


class TrainMLPRegressor:
    def __init__(self):
        self.label = "Train MLP Regressor"
        self.description = "Train a Multi-Layer Perceptron (MLP) regressor with the given parameters."
        self.canRunInBackground = False
        self.category = f"{TS_PREDICTIVE_MODELING}\\{TS_MLP}"
        self.idx_X = 0
        self.idx_X_nodata_value = 1
        self.idx_X_standardize = 2
        self.idx_y = 3
        self.idx_y_attribute = 4
        self.idx_y_nodata_value = 5
        self.idx_hidden_layers = 6
        self.idx_hidden_activation = 7
        self.idx_last_layer_activation = 8
        self.idx_epochs = 9
        self.idx_batch_size = 10
        self.idx_optimizer = 11
        self.idx_learning_rate = 12
        self.idx_loss_function = 13
        self.idx_is_early_stopping = 14
        self.idx_early_stopping_patience = 15
        self.idx_validation_split = 16
        self.idx_validation_data = 17
        self.idx_metrics = 18
        self.idx_random_state = 19
        self.idx_apply_smote = 20
        self.idx_n_synthetic_samples = 21
        self.idx_minority_class_label = 22
        self.idx_k_neighbors = 23
        self.idxs_hidden_smote_params = [
            self.idx_n_synthetic_samples,
            self.idx_minority_class_label,
            self.idx_k_neighbors
        ]
        self.idx_output_file = 24

    def getParameterInfo(self):
        (param_X,
            param_X_nodata_value,
            param_X_standardize,
            param_y,
            param_y_attribute,
            param_y_nodata_value
        ) = make_mlp_X_y_params(multiple_y_supported=False)

        param_hidden_layers = make_mlp_hidden_layers_params()
        param_hidden_layer_activation = make_mlp_hidden_layer_activation_param()

        param_last_layer_activation = arcpy.Parameter(
            displayName="Last Layer Activation Function",
            name="last_layer_activation",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        param_last_layer_activation.filter.type = "ValueList"
        param_last_layer_activation.filter.list = [ACTIVATION_LINEAR, ACTIVATION_SIGMOID]
        param_last_layer_activation.value = ACTIVATION_LINEAR

        (param_validation_split,
            param_validation_data,
            param_epochs,
            param_batch_size,
            param_optimizer,
            param_learning_rate,
            param_is_early_stopping,
            param_early_stopping_patience,
            param_random_state,
            param_apply_smote,
            param_n_synthetic_samples,
            param_minority_class_label,
            param_k_neighbors
        ) = make_mlp_training_params()

        param_loss_function = arcpy.Parameter(
            displayName="Loss Function",
            name="loss_function",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        param_loss_function.filter.type = "ValueList"
        param_loss_function.filter.list = [LOSS_MSE, LOSS_L1, LOSS_HUBER]
        param_loss_function.value = LOSS_MSE

        param_metrics = arcpy.Parameter(
            displayName="Validation Metrics",
            name="validation_metrics",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        param_metrics.filter.type = "ValueList"
        param_metrics.filter.list = [VALIDATION_MSE, VALIDATION_RMSE, VALIDATION_L1, VALIDATION_R2]
        param_metrics.value = VALIDATION_MSE

        param_output_model_filepath = arcpy.Parameter(
            displayName="Output Model File",
            name="output_file",
            datatype="DEFile",
            parameterType="Required",
            direction="Output"
        )
        param_output_model_filepath.value = "regressor_model.pth"

        params = [
            param_X,  # 0
            param_X_nodata_value,  # 1
            param_X_standardize,  # 2
            param_y,  # 3
            param_y_attribute,  # 4
            param_y_nodata_value,  # 5
            param_hidden_layers,  # 6
            param_hidden_layer_activation,  # 7
            param_last_layer_activation,  # 8
            param_epochs,  # 9
            param_batch_size,  # 10
            param_optimizer,  # 11
            param_learning_rate,  # 12
            param_loss_function,  # 13
            param_is_early_stopping,  # 14
            param_early_stopping_patience,  # 15
            param_validation_split,  # 16
            param_validation_data,  # 17
            param_metrics,  # 18
            param_random_state,  # 19
            param_apply_smote,  # 20
            param_n_synthetic_samples,  # 21
            param_minority_class_label,  # 22
            param_k_neighbors,  # 23
            param_output_model_filepath,  # 24
        ]
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        # Enable y nodata field if any of the y layers is a raster
        # Enable y attribute field if any of the y layers has an attribute table
        y = parameters[self.idx_y]
        if y.value and not y.hasBeenValidated:
            try:
                contains_raster, has_attribute_table, _ = check_mlp_y_conditionals(y)

                parameters[self.idx_y_nodata_value].enabled = contains_raster
                parameters[self.idx_y_attribute].enabled = has_attribute_table
            except Exception:
                pass
        if not y.value:
            parameters[self.idx_y_nodata_value].enabled = False
            parameters[self.idx_y_attribute].enabled = False

        # Enable early stopping patience field if early stopping is selected
        parameters[self.idx_early_stopping_patience].enabled = parameters[self.idx_is_early_stopping].value

        # Enable SMOTE related params when SMOTE is selected
        apply_smote = parameters[self.idx_apply_smote]
        if apply_smote.value:
            for idx in self.idxs_hidden_smote_params:
                parameters[idx].enabled = True
        else:
            for idx in self.idxs_hidden_smote_params:
                parameters[idx].enabled = False

        # Properly initialize output path (cannot be done in getParameterInfo() because ArcGIS Pro doesn't
        # have access to the current project yet)
        output_file = parameters[self.idx_output_file]
        if output_file.value and not any(x in str(output_file.value) for x in ["\\", "/"]):
            try:
                home_folder = arcpy.mp.ArcGISProject("CURRENT").homeFolder
                filename = str(output_file.value)

                output_file.value = os.path.join(home_folder, filename)

            except Exception:
                pass

        return

    def updateMessages(self, parameters):
        # Validate geometry of y
        param_y = parameters[self.idx_y]
        param_y.clearMessage()
        if param_y.value and not param_y.hasBeenValidated:
            try:
                y_text = param_y.valueAsText
                y_paths = y_text.split(";")
                y_paths_clean = [path.strip("'") for path in y_paths if path]

                for path in y_paths_clean:
                    desc = arcpy.Describe(path)
                    data_type = desc.dataType
                    if data_type in ["FeatureLayer", "FeatureClass", "ShapeFile"]:
                        if desc.shapeType != "Point":
                            param_y.setErrorMessage("At least one of the target label layers has non-point geometry, which is not supported.")
                            break
            except Exception:
                pass

        # Validate dropout rate(s)
        param_hidden_layers = parameters[self.idx_hidden_layers]
        param_hidden_layers.clearMessage()

        for layer in param_hidden_layers.value:
            dropout_rate = get_mlp_hidden_layer_dropout_rate(layer)
            if dropout_rate is not None and ((dropout_rate < 0) or (dropout_rate >= 1)):
                param_hidden_layers.setErrorMessage(f"Invalid dropout rate {dropout_rate}. Must be between 0 and 1.")

        # Validate number of epochs
        param_epochs = parameters[self.idx_epochs]
        param_epochs.clearMessage()

        if param_epochs.value < 1:
            param_epochs.setErrorMessage(f"Invalid number of epochs {param_epochs.value}. Must be at least one.")

        # Validate validation split
        # TODO: make this conditional based on whether validation data is provided
        param_validation_split = parameters[self.idx_validation_split]
        param_validation_split.clearMessage()
        validation_split = param_validation_split.value

        if (validation_split <= 0) or (validation_split >= 1):
            param_validation_split.setErrorMessage(f"Invalid validation split {validation_split}. Must be between 0 and 1.")

        # Warn if output file already exists
        param_output_file = parameters[self.idx_output_file]
        param_output_file.clearMessage()
        output_file = param_output_file.valueAsText
        if output_file:
            if os.path.exists(output_file):
                param_output_file.setWarningMessage("Output model file already exists and will be overwritten.")

        return

    def execute(self, parameters, messages):
        input_rasters = parameters[self.idx_X].valueAsText.split(";")
        target_labels = parameters[self.idx_y].valueAsText.split(";")
        hidden_layers = make_mlp_hidden_layer_specs(
            parameters[self.idx_hidden_layers].value,
            parameters[self.idx_hidden_activation].valueAsText
        )
        last_layer = parameters[self.idx_last_layer_activation].valueAsText
        validation_data = parameters[self.idx_validation_data].valueAsText if parameters[self.idx_validation_data].value is not None else None
        apply_smote = parameters[self.idx_apply_smote].value
        smote_params = None
        if apply_smote:
            smote_params = (
                parameters[self.idx_n_synthetic_samples].value,
                parameters[self.idx_minority_class_label].value,
                parameters[self.idx_k_neighbors].value
            )

        mlp_regression.train_MLP_regressor(
            input_rasters=input_rasters,
            X_nodata_value=parameters[self.idx_X_nodata_value].value,
            standardize=parameters[self.idx_X_standardize].value,
            target_labels=target_labels,
            target_labels_attr=get_valueAsText_if_enabled(parameters[self.idx_y_attribute]),
            y_nodata_value=get_value_if_enabled(parameters[self.idx_y_nodata_value]),
            hidden_layers=hidden_layers,
            last_layer=last_layer,
            epochs=parameters[self.idx_epochs].value,
            batch_size=parameters[self.idx_batch_size].value,
            optimizer=parameters[self.idx_optimizer].valueAsText,
            learning_rate=parameters[self.idx_learning_rate].value,
            loss_function=parameters[self.idx_loss_function].valueAsText,
            is_early_stopping=parameters[self.idx_is_early_stopping].value,
            early_stopping_patience=get_value_if_enabled(parameters[self.idx_early_stopping_patience]),
            validation_split=parameters[self.idx_validation_split].value,
            validation_data=validation_data,
            validation_metrics=parameters[self.idx_metrics].valueAsText,
            random_state=parameters[self.idx_random_state].value,
            apply_smote=apply_smote,
            smote_params=smote_params,
            output_model_file=parameters[self.idx_output_file].valueAsText
        )

    def postExecute(self, parameters):
        return


class ValidateMLPClassifier:
    def __init__(self):
        self.label = "Validate MLP Classifier"
        self.description = ""
        self.canRunInBackground = False
        self.category = f"{TS_PREDICTIVE_MODELING}\\{TS_MLP}"
        self.idx_X = 0
        self.idx_X_nodata_value = 1
        self.idx_X_standardize = 2
        self.idx_y = 3
        self.idx_y_attribute = 4
        self.idx_y_nodata_value = 5
        self.idx_model_file = 6
        self.idx_threshold = 7
        self.idx_output_prob_raster = 8
        self.idx_output_classified_raster = 9
        self.idx_metrics = 10

    def getParameterInfo(self):
        (param_X,
            param_X_nodata_value,
            param_X_standardize,
            param_y,
            param_y_attribute,
            param_y_nodata_value
        ) = make_mlp_X_y_params()

        param_model_file = make_mlp_input_model_file_param()

        param_test_metrics = arcpy.Parameter(
            displayName="Test metrics",
            name="test_metrics",
            datatype="String",
            parameterType="Required",
            direction="Input",
            multiValue=True
        )
        param_test_metrics.filter.type = "ValueList"
        param_test_metrics.filter.list = [VALIDATION_ACCURACY, VALIDATION_PRECISION, VALIDATION_RECALL, VALIDATION_F1]
        param_test_metrics.value = VALIDATION_ACCURACY

        param_classification_threshold, param_output_prob_raster, param_output_classification_result_raster = make_mlp_classifier_prediction_params()

        params = [
            param_X,  # 0
            param_X_nodata_value,  # 1
            param_X_standardize,  # 2
            param_y,  # 3
            param_y_attribute,  # 4
            param_y_nodata_value,  # 5
            param_model_file,  # 6
            param_classification_threshold,  # 7
            param_output_prob_raster,  # 8
            param_output_classification_result_raster,  # 9
            param_test_metrics  # 10
        ]
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        # Enable y nodata field if any of the y layers is a raster
        # Enable y attribute field if any of the y layers has an attribute table
        y = parameters[self.idx_y]
        if y.value and not y.hasBeenValidated:
            try:
                contains_raster, has_attribute_table, more_than_one = check_mlp_y_conditionals(y)

                # Classifier allows more than one label file, but in that case they are all expected
                # to be feature layers, so nodata param is not applicable. Each feature will be considered
                # to be one target class, so no need for attribute param either.
                if more_than_one:
                    parameters[self.idx_y_nodata_value].enabled = False
                    parameters[self.idx_y_attribute].enabled = False
                else:
                    parameters[self.idx_y_nodata_value].enabled = contains_raster
                    parameters[self.idx_y_attribute].enabled = has_attribute_table
            except Exception:
                pass
        if not y.value:
            parameters[self.idx_y_nodata_value].enabled = False
            parameters[self.idx_y_attribute].enabled = False

        update_mlp_classifier_threshold_parameter(
            model_file_param=parameters[self.idx_model_file],
            threshold_param=parameters[self.idx_threshold]
        )

        return

    def updateMessages(self, parameters):
        # Validate geometry & amount of rasters of y
        param_y = parameters[self.idx_y]
        param_y.clearMessage()
        if param_y.value and not param_y.hasBeenValidated:
            try:
                y_text = param_y.valueAsText
                y_paths = y_text.split(";")
                y_paths_clean = [path.strip("'") for path in y_paths if path]

                contains_raster = False

                for path in y_paths_clean:
                    desc = arcpy.Describe(path)
                    data_type = desc.dataType
                    if data_type in ["RasterLayer", "RasterDataset", "RasterBand"] and not (contains_raster):
                        contains_raster = True
                    elif data_type in ["FeatureLayer", "FeatureClass", "ShapeFile"]:
                        if desc.shapeType != "Point":
                            param_y.setErrorMessage("At least one of the target label layers has non-point geometry, which is not supported.")
                            break

                if (len(y_paths_clean) > 1) and contains_raster:
                    param_y.setErrorMessage("Only one raster file is supported. If multiple target label layers are provided, they must be feature layers.")
            except Exception:
                pass

        return

    def execute(self, parameters, messages):
        input_rasters = parameters[self.idx_X].valueAsText.split(";")
        target_labels = parameters[self.idx_y].valueAsText.split(";")

        mlp_classification.test_MLP_classifier(
            input_rasters=input_rasters,
            X_nodata_value=parameters[self.idx_X_nodata_value].value,
            standardize=parameters[self.idx_X_standardize].value,
            target_labels=target_labels,
            target_labels_attr=get_valueAsText_if_enabled(parameters[self.idx_y_attribute]),
            y_nodata_value=get_value_if_enabled(parameters[self.idx_y_nodata_value]),
            model_file=parameters[self.idx_model_file].valueAsText,
            classification_threshold=parameters[self.idx_threshold].value,
            output_raster_prob=parameters[self.idx_output_prob_raster].valueAsText,
            output_raster_classified=parameters[self.idx_output_classified_raster].valueAsText,
            test_metrics=parameters[self.idx_metrics].valueAsText
        )

    def postExecute(self, parameters):
        return


class ValidateMLPRegressor:
    def __init__(self):
        self.label = "Validate MLP Regressor"
        self.description = ""
        self.canRunInBackground = False
        self.category = f"{TS_PREDICTIVE_MODELING}\\{TS_MLP}"
        self.idx_X = 0
        self.idx_X_nodata_value = 1
        self.idx_X_standardize = 2
        self.idx_y = 3
        self.idx_y_attribute = 4
        self.idx_y_nodata_value = 5
        self.idx_model_file = 6
        self.idx_output_result_raster = 7
        self.idx_metrics = 8

    def getParameterInfo(self):
        (param_X,
            param_X_nodata_value,
            param_X_standardize,
            param_y,
            param_y_attribute,
            param_y_nodata_value
        ) = make_mlp_X_y_params(multiple_y_supported=False)

        param_model_file = make_mlp_input_model_file_param()

        param_output_regression_result_raster = make_mlp_regressor_prediction_output_params(filename="predicted_values_test_result")

        param_test_metrics = arcpy.Parameter(
            displayName="Test metrics",
            name="test_metrics",
            datatype="String",
            parameterType="Required",
            direction="Input",
            multiValue=True
        )
        param_test_metrics.filter.type = "ValueList"
        param_test_metrics.filter.list = [VALIDATION_MSE, VALIDATION_RMSE, VALIDATION_L1, VALIDATION_R2]
        param_test_metrics.value = VALIDATION_MSE

        params = [
            param_X,  # 0
            param_X_nodata_value,  # 1
            param_X_standardize,  # 2
            param_y,  # 3
            param_y_attribute,  # 4
            param_y_nodata_value,  # 5
            param_model_file,  # 6
            param_output_regression_result_raster,  # 7
            param_test_metrics  # 8
        ]
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        # Enable y nodata field if any of the y layers is a raster
        # Enable y attribute field if any of the y layers has an attribute table
        y = parameters[self.idx_y]
        if y.value and not y.hasBeenValidated:
            try:
                contains_raster, has_attribute_table, _ = check_mlp_y_conditionals(y)

                parameters[self.idx_y_nodata_value].enabled = contains_raster
                parameters[self.idx_y_attribute].enabled = has_attribute_table
            except Exception:
                pass
        if not y.value:
            parameters[self.idx_y_nodata_value].enabled = False
            parameters[self.idx_y_attribute].enabled = False

        return

    def updateMessages(self, parameters):
        # Validate geometry of y
        param_y = parameters[self.idx_y]
        param_y.clearMessage()
        if param_y.value and not param_y.hasBeenValidated:
            try:
                y_text = param_y.valueAsText
                y_paths = y_text.split(";")
                y_paths_clean = [path.strip("'") for path in y_paths if path]

                contains_raster = False

                for path in y_paths_clean:
                    desc = arcpy.Describe(path)
                    data_type = desc.dataType
                    if data_type in ["RasterLayer", "RasterDataset", "RasterBand"] and not (contains_raster):
                        contains_raster = True
                    elif data_type in ["FeatureLayer", "FeatureClass", "ShapeFile"]:
                        if desc.shapeType != "Point":
                            param_y.setErrorMessage("At least one of the target label layers has non-point geometry, which is not supported.")
                            break

                if (len(y_paths_clean) > 1) and contains_raster:
                    param_y.setErrorMessage("Only one raster file is supported. If multiple target label layers are provided, they must be feature layers.")
            except Exception:
                pass

        return

    def execute(self, parameters, messages):
        input_rasters = parameters[self.idx_X].valueAsText.split(";")
        target_labels = parameters[self.idx_y].valueAsText.split(";")

        mlp_regression.test_MLP_regressor(
            input_rasters=input_rasters,
            X_nodata_value=parameters[self.idx_X_nodata_value].value,
            standardize=parameters[self.idx_X_standardize].value,
            target_labels=target_labels,
            target_labels_attr=get_valueAsText_if_enabled(parameters[self.idx_y_attribute]),
            y_nodata_value=get_value_if_enabled(parameters[self.idx_y_nodata_value]),
            model_file=parameters[self.idx_model_file].valueAsText,
            output_raster=parameters[self.idx_output_result_raster].valueAsText,
            test_metrics=parameters[self.idx_metrics].valueAsText
        )

    def postExecute(self, parameters):
        return


class PredictMLPClassifier:
    def __init__(self):
        self.label = "Predict with MLP Classifier"
        self.description = "Predict with a trained machine learning classifier model."
        self.canRunInBackground = False
        self.category = f"{TS_PREDICTIVE_MODELING}\\{TS_MLP}"
        self.idx_param_X = 0
        self.idx_param_X_nodata_value = 1
        self.idx_param_X_standardize = 2
        self.idx_param_model_file = 3
        self.idx_param_classification_threshold = 4
        self.idx_param_output_prob_raster = 5
        self.idx_param_output_classification_result_raster = 6

    def getParameterInfo(self):
        (param_X,
            param_X_nodata_value,
            param_X_standardize,
            param_model_file
        ) = make_mlp_prediction_input_params()

        param_classification_threshold, param_output_prob_raster, param_output_classification_result_raster = make_mlp_classifier_prediction_params(
            filename_prod="predicted_probabilities_test_result",
            filename_classified="predicted_class_test_result"
        )

        params = [
            param_X,  # 0
            param_X_nodata_value,  # 1
            param_X_standardize,  # 2
            param_model_file,  # 3
            param_classification_threshold,  # 4
            param_output_prob_raster,  # 5
            param_output_classification_result_raster  # 6
        ]
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        update_mlp_classifier_threshold_parameter(
            model_file_param=parameters[self.idx_param_model_file],
            threshold_param=parameters[self.idx_param_classification_threshold]
        )
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, messages):
        input_rasters = parameters[self.idx_param_X].valueAsText.split(";")

        mlp_classification.predict_with_MLP_classifier(
            input_rasters=input_rasters,
            X_nodata_value=parameters[self.idx_param_X_nodata_value].value,
            standardize=parameters[self.idx_param_X_standardize].value,
            model_file=parameters[self.idx_param_model_file].valueAsText,
            classification_threshold=parameters[self.idx_param_classification_threshold].value,
            output_raster_prob=parameters[self.idx_param_output_prob_raster].valueAsText,
            output_raster_classified=parameters[self.idx_param_output_classification_result_raster].valueAsText
        )

    def postExecute(self, parameters):
        return


class PredictMLPRegressor:
    def __init__(self):
        self.label = "Predict with MLP Regressor"
        self.description = "Predict with a trained machine learning regressor model."
        self.canRunInBackground = False
        self.category = f"{TS_PREDICTIVE_MODELING}\\{TS_MLP}"
        self.idx_param_X = 0
        self.idx_param_X_nodata_value = 1
        self.idx_param_X_standardize = 2
        self.idx_param_model_file = 3
        self.idx_param_output_regression_result_raster = 4

    def getParameterInfo(self):
        (param_X,
            param_X_nodata_value,
            param_X_standardize,
            param_model_file
        ) = make_mlp_prediction_input_params()

        param_output_regression_result_raster = make_mlp_regressor_prediction_output_params()

        params = [
            param_X,  # 0
            param_X_nodata_value,  # 1
            param_X_standardize,  # 2
            param_model_file,  # 3
            param_output_regression_result_raster  # 4
        ]
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, messages):
        input_rasters = parameters[self.idx_param_X].valueAsText.split(";")

        mlp_regression.predict_with_MLP_regressor(
            input_rasters=input_rasters,
            X_nodata_value=parameters[self.idx_param_X_nodata_value].value,
            standardize=parameters[self.idx_param_X_standardize].value,
            model_file=parameters[self.idx_param_model_file].valueAsText,
            output_raster=parameters[self.idx_param_output_regression_result_raster].valueAsText
        )

    def postExecute(self, parameters):
        return


def make_mlp_X_params():
    param_X = arcpy.Parameter(
        displayName="Input Features",
        name="X",
        datatype=["GPRasterLayer", "DERasterDataset"],
        parameterType="Required",
        multiValue=True,
        direction="Input"
    )

    param_X_nodata_value = arcpy.Parameter(
        displayName="Input Feature NoData Value",
        name="X_nodata_value",
        datatype="GPLong",  # TODO: should this be GPDouble?
        parameterType="Optional",
        direction="Input"
    )
    param_X_nodata_value.value = -99   # TODO: remove default value?

    param_X_standardize = arcpy.Parameter(
        displayName="Standardize Features",
        name="standardize_X",
        datatype="GPBoolean",
        parameterType="Optional",
        direction="Input"
    )
    param_X_standardize.value = False

    return (
        param_X,
        param_X_nodata_value,
        param_X_standardize,
    )


def make_mlp_X_y_params(multiple_y_supported=True):
    """Construct parameter objects for the parameters shared by MLP training tools."""
    param_X, param_X_nodata_value, param_X_standardize = make_mlp_X_params()

    param_y = arcpy.Parameter(
        displayName="Target Labels",
        name="y",
        datatype=["GPRasterLayer", "DERasterDataset", "GPFeatureLayer", "DEFeatureClass"],
        parameterType="Required",
        multiValue=multiple_y_supported,
        direction="Input"
    )

    param_y_attribute = arcpy.Parameter(
        displayName="Target Labels Attribute",
        name="y_attribute",
        datatype="Field",
        parameterType="Optional",
        multiValue=False,
        direction="Input"
    )
    param_y_attribute.parameterDependencies = [param_y.name]
    param_y_attribute.enabled = False

    param_y_nodata_value = arcpy.Parameter(
        displayName="Label NoData Value",
        name="y_nodata_value",
        datatype="GPLong",  # TODO: should this be GPDouble?
        parameterType="Optional",
        direction="Input"
    )
    param_y_nodata_value.enabled = False

    return (
        param_X,
        param_X_nodata_value,
        param_X_standardize,
        param_y,
        param_y_attribute,
        param_y_nodata_value,
    )


def make_mlp_hidden_layers_params():
    param_hidden_layers = arcpy.Parameter(
        displayName="Hidden Layers",
        name="hidden_layers",
        datatype="GPValueTable",
        parameterType="Required",
        direction="Input"
    )

    param_hidden_layers.columns = [
        ["GPLong", "Neurons in Layer"],
        ["GPDouble", "Dropout Rate"]
    ]
    # Note: Because of how GPValueTable gets handled by ArcGIS Pro,
    # if "Dropout Rate" is None, it will get evaluated to 0.0.
    # Something to be aware of
    param_hidden_layers.values = [[5, None]]
    # For horizontal display of 2 columns:
    param_hidden_layers.controlCLSID = "{1AA9A769-D3F3-4EB0-85CB-CC07C79313C8}"

    return param_hidden_layers


def make_mlp_hidden_layer_activation_param():
    param_hidden_layer_activation = arcpy.Parameter(
        displayName="Hidden Layer Activation Function",
        name="hidden_layer_activation",
        datatype="GPString",
        parameterType="Required",
        direction="Input"
    )
    param_hidden_layer_activation.filter.type = "ValueList"
    param_hidden_layer_activation.filter.list = [ACTIVATION_RELU, ACTIVATION_SIGMOID, ACTIVATION_TANH]
    param_hidden_layer_activation.value = ACTIVATION_RELU

    return param_hidden_layer_activation


def get_mlp_hidden_layer_dropout_rate(layer):
    if len(layer) > 2:
        return layer[2]
    if len(layer) > 1:
        return layer[1]
    return None


def make_mlp_hidden_layer_specs(hidden_layers, hidden_layer_activation):
    return [
        (layer[0], hidden_layer_activation, get_mlp_hidden_layer_dropout_rate(layer))
        for layer in hidden_layers
    ]


def make_mlp_training_params():
    param_validation_split = arcpy.Parameter(
        displayName="Validation Split",
        name="validation_split",
        datatype="GPDouble",
        parameterType="Optional",
        direction="Input"
    )
    param_validation_split.value = 0.2

    # TODO: consider leaving out
    param_validation_data = arcpy.Parameter(
        displayName="Validation Data",
        name="validation_data",
        datatype="GPTableView",
        parameterType="Optional",
        direction="Input"
    )

    param_epochs = arcpy.Parameter(
        displayName="Epochs",
        name="epochs",
        datatype="GPLong",
        parameterType="Required",
        direction="Input"
    )
    param_epochs.value = 50

    param_batch_size = arcpy.Parameter(
        displayName="Batch Size",
        name="batch_size",
        datatype="GPLong",
        parameterType="Required",
        direction="Input"
    )
    param_batch_size.value = 32

    param_optimizer = arcpy.Parameter(
        displayName="Optimizer",
        name="optimizer",
        datatype="GPString",
        parameterType="Required",
        direction="Input"
    )
    param_optimizer.filter.type = "ValueList"
    param_optimizer.filter.list = [OPTIMIZER_ADAM, OPTIMIZER_ADAGRAD, OPTIMIZER_RMSPROP, OPTIMIZER_SGD]
    param_optimizer.value = OPTIMIZER_ADAM

    param_learning_rate = arcpy.Parameter(
        displayName="Learning Rate",
        name="learning_rate",
        datatype="GPDouble",
        parameterType="Required",
        direction="Input"
    )
    param_learning_rate.value = 0.001

    param_is_early_stopping = arcpy.Parameter(
        displayName="Early Stopping",
        name="early_stopping",
        datatype="GPBoolean",
        parameterType="Optional",
        direction="Input"
    )
    param_is_early_stopping.value = True

    param_early_stopping_patience = arcpy.Parameter(
        displayName="Early Stopping Patience",
        name="es_patience",
        datatype="GPLong",
        parameterType="Optional",
        direction="Input"
    )
    param_early_stopping_patience.value = 5

    param_random_state = arcpy.Parameter(
        displayName="Random State",
        name="random_state",
        datatype="GPLong",
        parameterType="Optional",
        direction="Input"
    )

    (param_apply_smote,
        param_n_synthetic_samples,
        param_minority_class_label,
        param_k_neighbors
    ) = make_mlp_smote_params()

    return (
        param_validation_split,
        param_validation_data,
        param_epochs,
        param_batch_size,
        param_optimizer,
        param_learning_rate,
        param_is_early_stopping,
        param_early_stopping_patience,
        param_random_state,
        param_apply_smote,
        param_n_synthetic_samples,
        param_minority_class_label,
        param_k_neighbors
    )


def make_mlp_smote_params():
    param_apply_smote = arcpy.Parameter(
        displayName="Apply SMOTE",
        name="apply_smote",
        datatype="GPBoolean",
        parameterType="Optional",
        direction="Input"
    )
    param_apply_smote.value = False

    param_n_synthetic_samples = arcpy.Parameter(
        displayName="Number of Synthetic Samples",
        name="n_synthetic_samples",
        datatype="GPLong",
        parameterType="Optional",
        direction="Input"
    )

    param_minority_class_label = arcpy.Parameter(
        displayName="Minority Class Label",
        name="minority_class",
        datatype="GPLong",
        parameterType="Optional",
        direction="Input"
    )
    param_minority_class_label.value = 1

    param_k_neighbors = arcpy.Parameter(
        displayName="Number of Nearest Neighbors (k)",
        name="k_neighbors",
        datatype="GPLong",
        parameterType="Optional",
        direction="Input"
    )
    param_k_neighbors.value = 5

    return (
        param_apply_smote,
        param_n_synthetic_samples,
        param_minority_class_label,
        param_k_neighbors
    )


def make_mlp_input_model_file_param():
    return arcpy.Parameter(
        displayName="Input Model File",
        name="model_file",
        datatype="DEFile",
        parameterType="Required",
        direction="input"
    )


def is_mlp_classifier_multiclass_model(model_file):
    if not model_file:
        return False

    metadata_file = f"{os.path.splitext(str(model_file))[0]}.meta.json"
    if not os.path.exists(metadata_file):
        return False

    try:
        with open(metadata_file, "r", encoding="utf-8") as metadata_stream:
            metadata = json.load(metadata_stream)
        return int(metadata.get("target_label_count", 1)) > 1
    except Exception:
        return False


def update_mlp_classifier_threshold_parameter(model_file_param, threshold_param):
    threshold_param.enabled = not is_mlp_classifier_multiclass_model(model_file_param.valueAsText)


def make_mlp_prediction_input_params():
    """Construct parameter objects for the parameters shared by MLP prediction tools."""
    param_X, param_X_nodata_value, param_X_standardize = make_mlp_X_params()

    param_model_file = make_mlp_input_model_file_param()

    return (
        param_X,
        param_X_nodata_value,
        param_X_standardize,
        param_model_file
    )


def make_mlp_classifier_prediction_params(
    filename_prod="predicted_probabilities",
    filename_classified="predicted_class"
):
    param_classification_threshold = arcpy.Parameter(
        displayName="Classification threshold",
        name="classification_threshold",
        datatype="GPDouble",
        parameterType="Optional",
        direction="Input"
    )
    param_classification_threshold.value = 0.5

    param_output_prob_raster = arcpy.Parameter(
        displayName="Output predicted values probability raster",
        name="output_prob_raster",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output"
    )
    param_output_prob_raster.value = f"%workspace%\\{filename_prod}"

    param_output_classification_result_raster = arcpy.Parameter(
        displayName="Output predicted values classified raster",
        name="output_classification_result_raster",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output"
    )
    param_output_classification_result_raster.value = f"%workspace%\\{filename_classified}"

    return (
        param_classification_threshold,
        param_output_prob_raster,
        param_output_classification_result_raster
    )


def make_mlp_regressor_prediction_output_params(filename="predicted_values"):
    param_output_regression_result_raster = arcpy.Parameter(
        displayName="Output predicted values raster",
        name="output_regression_result_raster",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output"
    )
    param_output_regression_result_raster.value = f"%workspace%\\{filename}"

    return param_output_regression_result_raster


def get_value_if_enabled(parameter):
    return parameter.value if parameter.enabled else None


def get_valueAsText_if_enabled(parameter):
    return parameter.valueAsText if parameter.enabled else None


def check_mlp_y_conditionals(y_param):
    y_text = y_param.valueAsText
    y_paths = y_text.split(";")
    y_paths_clean = [path.strip("'") for path in y_paths if path]

    contains_raster = False
    has_attribute_table = False

    for path in y_paths_clean:
        desc = arcpy.Describe(path)
        data_type = desc.dataType
        if data_type in ["RasterLayer", "RasterDataset", "RasterBand"] and not (contains_raster):
            contains_raster = True
            raster = arcpy.Raster(path)
            if raster.hasRAT:
                has_attribute_table = True
        elif data_type in ["FeatureLayer", "FeatureClass", "ShapeFile"]:
            has_attribute_table = True

    return contains_raster, has_attribute_table, len(y_paths_clean) > 1
