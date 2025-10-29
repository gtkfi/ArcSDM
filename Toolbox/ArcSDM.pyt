import arcpy
import importlib
import os

from imp import reload

import arcsdm.agterbergchengci
import arcsdm.areafrequency
import arcsdm.calculateresponse_arcpy_wip
import arcsdm.calculateresponse
import arcsdm.calculateweights
import arcsdm.categoricalreclass
import arcsdm.fuzzyroc2
import arcsdm.mlp
import arcsdm.pca
import arcsdm.roctool
import arcsdm.symbolize
import arcsdm.splitting
import arcsdm.thinning
import arcsdm.tocfuzzification
import arcsdm.wofe_common

from arcsdm.common import execute_tool


# Toolsets and sub-toolsets within ArcSDM toolbox
TS_EXPLORATORY_DATA_ANALYSIS = "Exploratory Data Analysis"
TS_PREPROCESSING = "Preprocessing"
TS_EVIDENCE_DATA_PROCESSING = "Evidence Data Processing"
TS_FUZZY = "Fuzzy Membership"
TS_TRAINING_DATA_PROCESSING = "Training Data Processing"
TS_PREDICTIVE_MODELING = "Predictive Modeling"
TS_MACHINE_LEARNING = "Machine Learning"
TS_MODELING = "Modeling"
TS_CLASSIFIER_TESTING = "Classifier Testing"
TS_REGRESSOR_TESTING = "Regressor Testing"
TS_CLASSIFIER_APPLICATION = "Classifier Application"
TS_REGRESSOR_APPLICATION = "Regressor Application"
TS_WOFE = "Weights of Evidence"
TS_VALIDATION = "Validation"


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
            CategoricalAndReclassTool,
            FuzzyROC2,
            GetSDMValues,
            PCARaster,
            PCAVector,
            ROCTool,
            SplittingTool,
            ThinningTool,
            # Symbolize,
            TOCFuzzificationTool,
            TrainMLPClassifierTool,
            TrainMLPRegressorTool,
            MLPRegressorTestTool,
            MLPClassifierTestTool,
            RegressorPredictTool,
            ClassifierPredictTool
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
        try:
            importlib.reload(arcsdm.wofe_common)
        except:
            pass

        arcsdm.wofe_common.execute(self, parameters, messages)
        return
        
        
class AreaFrequencyTable(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Area Frequency Table"
        self.description = "Create a table for charting area of evidence classes vs number of training sites."
        self.canRunInBackground = False
        self.category = TS_VALIDATION

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
            displayName="True Positives",
            name="positive_points",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")
        positives_param.filter.list = ["Point", "Multipoint"]

        negatives_param = arcpy.Parameter(
            displayName="True Negatives",
            name="negative_points",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input")
        negatives_param.filter.list = ["Point", "Multipoint"]

        models_param = arcpy.Parameter(
            displayName="Classification Models",
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
        #execute_tool(arcsdm.roctool.execute, self, parameters, messages)
        try:
            importlib.reload(arcsdm.roctool)
        except:
            reload(arcsdm.roctool)
        arcsdm.roctool.execute(self, parameters, messages)
        return


class Symbolize(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Symbolize Raster with Prior Probability (Classified Values)"
        self.description = "This tool allows symbolizing prior probablity raster with predefined colorscheme from local raster_classified.lyr file"
        self.canRunInBackground = False
        self.category = "Utilities"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
        displayName="Raster Layer to symbolize",
        name="evidence_raster_layer",
        datatype="GPRasterLayer",
        parameterType="Required",
        direction="Input")
        
        param2 = arcpy.Parameter(
        displayName="Training sites (for prior prob)",
        name="training_sites",
        #datatype="DEFeatureClass",
        datatype="GPFeatureLayer",
        parameterType="Required",
        direction="Input")
        
        param5 = arcpy.Parameter(
        displayName="Unit area (km2)",
        name="Unit_Area__sq_km_",
        datatype="GPDouble",
        parameterType="Required",
        direction="Input")
        param5.value = "1"
                                  
        params = [param0, param2, param5]
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
        execute_tool(arcsdm.symbolize.execute, self, parameters, messages)
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


class SplittingTool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Splitting Tool"
        self.description = "Split training sites into training and testing datasets based on a random percentage."
        self.canRunInBackground = False
        self.category = f"{TS_PREPROCESSING}\\{TS_TRAINING_DATA_PROCESSING}"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param_input_layer = arcpy.Parameter(
            displayName="Training sites layer",
            name="training_sites_layer",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )

        param_random_percentage = arcpy.Parameter(
            displayName="Random percentage selection",
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
        param_output_layer.value = "reduced_sites_train"

        param_inverse_output_layer = arcpy.Parameter(
            displayName="Output testing layer (optional)",
            name="output_testing_layer",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Output"
        )
        param_inverse_output_layer.value = "reduced_sites_test"

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


class ThinningTool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Thinning Tool"
        self.description = "Selects subset of the training points based on a thinning value and minimum distance."
        self.canRunInBackground = False
        self.category = f"{TS_PREPROCESSING}\\{TS_TRAINING_DATA_PROCESSING}"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param_input_layer = arcpy.Parameter(
            displayName="Training sites layer",
            name="Training_Sites_layer",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")

        param_unit_area = arcpy.Parameter(
            displayName="Unit area",
            name="Unit_Area",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input")
        param_unit_area.value = 500

        param_area_unit = arcpy.Parameter(
            displayName="Area Unit",
            name="Area_Unit",
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
            name="Min_Distance",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input")

        param_output = arcpy.Parameter(
            displayName="Output layer",
            name="layerSelection",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Output")
        param_output.value = "thinned_sites"

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


class CategoricalAndReclassTool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Categorical & Reclass"
        self.description = "Create fuzzy memberships for categorical data by first reclassification to integers and then division by an appropriate value."
        self.canRunInBackground = False
        self.category = f"{TS_PREPROCESSING}\\{TS_EVIDENCE_DATA_PROCESSING}\\{TS_FUZZY}"

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


class TOCFuzzificationTool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "TOC Fuzzification"
        self.description = "This fuzzification method utilized the symbolization of the input raster that has been applied in the map document table of contents (TOC). The symbolization in the TOC defines the number of classes and this tool rescales those classes (1...N) to the range [0,1] by (C - 1)/(N-1) where C is the class value and N is the number of classes."
        self.canRunInBackground = False
        self.category = f"{TS_PREPROCESSING}\\{TS_EVIDENCE_DATA_PROCESSING}\\{TS_FUZZY}"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param_input_raster = arcpy.Parameter(
        displayName="Input Raster",
        name="input_raster",
        datatype="GPRasterLayer",
        parameterType="Required",
        direction="Input")
        
        param_reclass_field = arcpy.Parameter(
        displayName="Reclass Field",
        name="reclass_field",
        datatype="Field",
        parameterType="Required",
        direction="Input")

        param_reclassification = arcpy.Parameter(
        displayName="Reclassification",
        name="reclassification",
        datatype="remap",
        parameterType="Required",
        direction="Input")

        param_num_classes = arcpy.Parameter(
        displayName="Number of Classes",
        name="classes",
        datatype="GPLong",
        parameterType="Required",
        direction="Input")

        param_output_raster = arcpy.Parameter(
        displayName="Output Fuzzy Membership Raster",
        name="fmtoc",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output")
        param_output_raster.value = "%Workspace%\FMTOC"
        
        param_reclass_field.value = "VALUE"
        param_reclass_field.enabled = False
        param_reclassification.enabled = False
        
        param_reclass_field.parameterDependencies = [param_input_raster.name]  
        param_reclassification.parameterDependencies = [param_input_raster.name,param_reclass_field.name]
        params = [param_input_raster,param_reclass_field,param_reclassification,param_num_classes,param_output_raster]
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

        input_file_type = os.path.splitext(parameters[0].valueAsText.lower())[1]
        output_path = parameters[4].valueAsText.lower()

        if ".gdb" not in output_path:
            parameters[4].value = os.path.join(os.path.dirname(output_path), "FMTOC" + input_file_type)

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        
        if parameters[3].value and parameters[3].value < 1:
            parameters[3].setErrorMessage("'Classes' must be greater than 1.")
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        execute_tool(arcsdm.tocfuzzification.Calculate, self, parameters, messages)
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


class FuzzyROC2(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Fuzzy ROC"
        self.description = "Fuzzy Membership + Fuzzy Overlay + ROC (Receiver Operator Characteristic)"
        self.canRunInBackground = False
        self.category = TS_PREDICTIVE_MODELING

    def getParameterInfo(self):
        """Define parameter definitions"""
        
        param_inputs = arcpy.Parameter(
        displayName="Input rasters, Fuzzy Membership functions and parameters",
            name="inputrasters",
            datatype="DETable",
            multiValue=1,
            parameterType="Required",
            direction="Input")
        param_inputs.columns = [['GPRasterLayer', 'Input raster name'], ['String', 'Membership type'], ['String', 'Midpoint Min'], ['String', 'Midpoint Max'], ['String', 'Midpoint Count'], ['String', 'Spread Min'], ['String', 'Spread Max'], ['String', 'Spread Count']]
        param_inputs.filters[1].type = 'ValueList'
        param_inputs.filters[1].list = ['Small', 'Large']

        param_draw = arcpy.Parameter(
            displayName="Draw only Fuzzy Membership plots",
            name="plots",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input")
        param_draw.value = False

        param_true_positives = arcpy.Parameter(
            displayName="\nTrue Positives Feature Class",
            name="truepositives",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Input")

        param_output_folder = arcpy.Parameter(
            displayName="Output Folder",
            name="output_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input")
        param_output_folder.filter.list = ["File System"]

        if arcpy.env.workspace:
            param_output_folder.value = os.path.join(os.path.dirname(arcpy.env.workspace), "FuzzyROC")

        param_overlay_type = arcpy.Parameter(
            displayName="Fuzzy Overlay Type",
            name="overlay_type",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        param_overlay_type.filter.type = "ValueList"
        param_overlay_type.filter.list = ['And', 'Or', 'Product', 'Sum', 'Gamma']
        param_overlay_type.value = 'And'

        param_overlay_parameter = arcpy.Parameter(
            displayName="Fuzzy Overlay Parameter",
            name="overlay_param",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )

        param_overlay_parameter.value = 0.0
        param_overlay_parameter.filter.type = "Range"
        param_overlay_parameter.filter.list = [0.0, 1.0]

        param_display_method = arcpy.Parameter(
        displayName="Plot display method",
        name="display_method",
        datatype="GPString",
        parameterType="Required",
        direction="Input",
        enabled=False,
        category='Plotting')
        param_display_method.filter.type = "ValueList"
        param_display_method.filter.list = ["To PDF file(s)", "To PNG file(s)"]
        param_display_method.value = "To PDF file(s)"

        params = [param_inputs, param_draw, param_true_positives, param_output_folder, param_overlay_type, param_overlay_parameter, param_display_method]
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
        if (parameters[1].value):
            parameters[4].enabled = False
            parameters[6].enabled = True
        else:
            parameters[4].enabled = True
            parameters[6].enabled = False

        if parameters[4].value == "Gamma" and parameters[4].enabled:
            parameters[5].enabled = True
        else:
            parameters[5].enabled = False

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        default_folder = os.path.join(os.path.dirname(arcpy.env.workspace), "FuzzyROC")
        if default_folder and os.path.normpath(parameters[3].valueAsText) == os.path.normpath(default_folder):
            parameters[3].clearMessage()

        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        default_folder = os.path.join(os.path.dirname(arcpy.env.workspace), "FuzzyROC")
        if default_folder and os.path.normpath(parameters[3].valueAsText) == os.path.normpath(default_folder):
            if not os.path.exists(default_folder):
                os.makedirs(default_folder)
        execute_tool(arcsdm.fuzzyroc2.Execute, self, parameters, messages)
        return


class TrainMLPClassifierTool(object):
    def __init__(self):
        """Train a Multi-Layer Perceptron (MLP) classifier with the given parameters."""
        self.label = "Train MLP Classifier"
        self.description = "Train a Multi-Layer Perceptron (MLP) classifier with the given parameters."
        self.canRunInBackground = False
        self.category = f"{TS_PREDICTIVE_MODELING}\\{TS_MACHINE_LEARNING}\\{TS_MODELING}"

    def getParameterInfo(self):
        """Define parameter definitions"""

        param_X = arcpy.Parameter(
            displayName="Input Features",
            name="X",
            datatype=["GPRasterLayer"],
            parameterType="Required",
            multiValue=True,
            direction="Input")

        param_y = arcpy.Parameter(
            displayName="Target Labels",
            name="y",
            datatype=["GPRasterLayer", "GPFeatureLayer"],
            parameterType="Required",
            direction="Input")
        
        param_y_attribute = arcpy.Parameter(
            displayName="Target Labels attribute",
            name="y_attribute",
            datatype="Field",
            parameterType="Optional",
            direction="Input")
        
        param_y_attribute.parameterDependencies = [param_y.name]
        
        param_X_nodata_value = arcpy.Parameter(
            displayName="Input Feature NoData Value",
            name="X_nodata_value",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input")
        param_X_nodata_value.value = -99
        
        param_y_nodata_value = arcpy.Parameter(
            displayName="Label NoData Value",
            name="y_nodata_value",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input")

        param_neurons = arcpy.Parameter(
            displayName="Neurons per Layer. A comma separeted list of integers: e.g. 10,5,10",
            name="neurons",
            datatype="GPString",
            parameterType="Required",
            direction="Input")

        param_validation_split = arcpy.Parameter(
            displayName="Validation Split",
            name="validation_split",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input")
        param_validation_split.value = 0.2

        param_validation_data = arcpy.Parameter(
            displayName="Validation Data",
            name="validation_data",
            datatype="GPTableView",
            parameterType="Optional",
            direction="Input")

        param_activation = arcpy.Parameter(
            displayName="Activation Function",
            name="activation",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        param_activation.filter.type = "ValueList"
        param_activation.filter.list = ["relu", "linear", "sigmoid", "tanh"]
        param_activation.value = "relu"

        param_output_neurons = arcpy.Parameter(
            displayName="Output Neurons",
            name="output_neurons",
            datatype="GPLong",
            parameterType="Required",
            direction="Input")
        param_output_neurons.value = 1

        param_last_activation = arcpy.Parameter(
            displayName="Last Layer Activation Function",
            name="last_activation",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        param_last_activation.filter.type = "ValueList"
        param_last_activation.filter.list = ["sigmoid", "softmax"]
        param_last_activation.value = "sigmoid"

        param_epochs = arcpy.Parameter(
            displayName="Epochs",
            name="epochs",
            datatype="GPLong",
            parameterType="Required",
            direction="Input")
        param_epochs.value = 50

        param_batch_size = arcpy.Parameter(
            displayName="Batch Size",
            name="batch_size",
            datatype="GPLong",
            parameterType="Required",
            direction="Input")
        param_batch_size.value = 32

        param_optimizer = arcpy.Parameter(
            displayName="Optimizer",
            name="optimizer",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        param_optimizer.filter.type = "ValueList"
        param_optimizer.filter.list = ["adam", "adagrad", "rmsprop", "sdg"]
        param_optimizer.value = "adam"

        param_learning_rate = arcpy.Parameter(
            displayName="Learning Rate",
            name="learning_rate",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input")
        param_learning_rate.value = 0.001

        param_loss_function = arcpy.Parameter(
            displayName="Loss Function",
            name="loss_function",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        param_loss_function.filter.type = "ValueList"
        param_loss_function.filter.list = ["binary_crossentropy", "categorical_crossentropy"]
        param_loss_function.value = "binary_crossentropy"

        param_dropout_rate = arcpy.Parameter(
            displayName="Dropout Rate",
            name="dropout_rate",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input")

        param_early_stopping = arcpy.Parameter(
            displayName="Early Stopping",
            name="early_stopping",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input")
        param_early_stopping.value = True

        param_es_patience = arcpy.Parameter(
            displayName="Early Stopping Patience",
            name="es_patience",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input")
        param_es_patience.value = 5

        param_metrics = arcpy.Parameter(
            displayName="Validation Metrics",
            name="validation_metrics",
            datatype="GPString",
            parameterType="Optional",
            direction="Input")
        param_metrics.filter.type = "ValueList"
        param_metrics.filter.list = ["accuracy", "precision", "recall"]
        param_metrics.value = "accuracy"

        param_random_state = arcpy.Parameter(
            displayName="Random State",
            name="random_state",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input")
        
        param_output_file = arcpy.Parameter(
            displayName="Output Model File",
            name="output_file",
            datatype="DEFile",
            parameterType="Required",
            direction="Output")
        param_output_file.value = "model"

        params = [param_X,
                  param_y,
                  param_y_attribute,
                  param_X_nodata_value,
                  param_y_nodata_value,
                  param_neurons,
                  param_validation_split,
                  param_validation_data,
                  param_activation,
                  param_output_neurons,
                  param_last_activation,
                  param_epochs,
                  param_batch_size,
                  param_optimizer,
                  param_learning_rate,
                  param_loss_function,
                  param_dropout_rate,
                  param_early_stopping,
                  param_es_patience,
                  param_metrics,
                  param_random_state,
                  param_output_file
                ]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal validation is performed. This method is called whenever a parameter has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """Execute the tool."""
        execute_tool(arcsdm.mlp.Execute_MLP_classifier, self, parameters, messages)


class TrainMLPRegressorTool(object):
    def __init__(self):
        """Train a Multi-Layer Perceptron (MLP) regressor with the given parameters."""
        self.label = "Train MLP Regressor"
        self.description = "Train a Multi-Layer Perceptron (MLP) regressor with the given parameters."
        self.canRunInBackground = False
        self.category = f"{TS_PREDICTIVE_MODELING}\\{TS_MACHINE_LEARNING}\\{TS_MODELING}"

    def getParameterInfo(self):
        """Define parameter definitions"""

        param_X = arcpy.Parameter(
            displayName="Input Features",
            name="X",
            datatype=["GPRasterLayer"],
            parameterType="Required",
            multiValue=True,
            direction="Input")

        param_y = arcpy.Parameter(
            displayName="Target Labels",
            name="y",
            datatype=["GPRasterLayer", "GPFeatureLayer"],
            parameterType="Required",
            direction="Input")
        
        param_y_attribute = arcpy.Parameter(
            displayName="Target Labels attribute",
            name="y_attribute",
            datatype="Field",
            parameterType="Optional",
            direction="Input")
        
        param_y_attribute.parameterDependencies = [param_y.name]
        
        param_X_nodata_value = arcpy.Parameter(
            displayName="Input Feature NoData Value",
            name="X_nodata_value",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input")
        param_X_nodata_value.value = -99
        
        param_y_nodata_value = arcpy.Parameter(
            displayName="Label NoData Value",
            name="y_nodata_value",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input")

        param_neurons = arcpy.Parameter(
            displayName="Neurons per Layer. A comma separeted list of integers: e.g. 10,5,10",
            name="neurons",
            datatype="GPString",
            parameterType="Required",
            direction="Input")

        param_validation_split = arcpy.Parameter(
            displayName="Validation Split",
            name="validation_split",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input")
        param_validation_split.value = 0.2

        param_validation_data = arcpy.Parameter(
            displayName="Validation Data",
            name="validation_data",
            datatype="GPTableView",
            parameterType="Optional",
            direction="Input")

        param_activation = arcpy.Parameter(
            displayName="Activation Function",
            name="activation",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        param_activation.filter.type = "ValueList"
        param_activation.filter.list = ["relu", "linear", "sigmoid", "tanh"]
        param_activation.value = "relu"

        param_output_neurons = arcpy.Parameter(
            displayName="Output Neurons",
            name="output_neurons",
            datatype="GPLong",
            parameterType="Required",
            direction="Input")
        param_output_neurons.value = 1

        param_last_activation = arcpy.Parameter(
            displayName="Last Layer Activation Function",
            name="last_activation",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        param_last_activation.filter.type = "ValueList"
        param_last_activation.filter.list = ["sigmoid", "softmax"]
        param_last_activation.value = "sigmoid"

        param_epochs = arcpy.Parameter(
            displayName="Epochs",
            name="epochs",
            datatype="GPLong",
            parameterType="Required",
            direction="Input")
        param_epochs.value = 50

        param_batch_size = arcpy.Parameter(
            displayName="Batch Size",
            name="batch_size",
            datatype="GPLong",
            parameterType="Required",
            direction="Input")
        param_batch_size.value = 32

        param_optimizer = arcpy.Parameter(
            displayName="Optimizer",
            name="optimizer",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        param_optimizer.filter.type = "ValueList"
        param_optimizer.filter.list = ["adam", "adagrad", "rmsprop", "sdg"]
        param_optimizer.value = "adam"

        param_learning_rate = arcpy.Parameter(
            displayName="Learning Rate",
            name="learning_rate",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input")
        param_learning_rate.value = 0.001

        param_loss_function = arcpy.Parameter(
            displayName="Loss Function",
            name="loss_function",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        param_loss_function.filter.type = "ValueList"
        param_loss_function.filter.list = ["mse", "mae", "hinge", "huber"]
        param_loss_function.value = "mse"

        param_dropout_rate = arcpy.Parameter(
            displayName="Dropout Rate",
            name="dropout_rate",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input")

        param_early_stopping = arcpy.Parameter(
            displayName="Early Stopping",
            name="early_stopping",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input")
        param_early_stopping.value = True

        param_es_patience = arcpy.Parameter(
            displayName="Early Stopping Patience",
            name="es_patience",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input")
        param_es_patience.value = 5

        param_metrics = arcpy.Parameter(
            displayName="Validation Metrics",
            name="validation_metrics",
            datatype="GPString",
            parameterType="Optional",
            direction="Input")
        param_metrics.filter.type = "ValueList"
        param_metrics.filter.list = ["mse", "rmse", "mae", "r2"]
        param_metrics.value = "mse"

        param_random_state = arcpy.Parameter(
            displayName="Random State",
            name="random_state",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input")
        
        param_output_file = arcpy.Parameter(
            displayName="Output Model File",
            name="output_file",
            datatype="DEFile",
            parameterType="Required",
            direction="Output")
        param_output_file.value = "model"

        params = [param_X,
                  param_y,
                  param_y_attribute,
                  param_X_nodata_value,
                  param_y_nodata_value,
                  param_neurons,
                  param_validation_split,
                  param_validation_data,
                  param_activation,
                  param_output_neurons,
                  param_last_activation,
                  param_epochs,
                  param_batch_size,
                  param_optimizer,
                  param_learning_rate,
                  param_loss_function,
                  param_dropout_rate,
                  param_early_stopping,
                  param_es_patience,
                  param_metrics,
                  param_random_state,
                  param_output_file
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
        """Modify the values and properties of parameters before internal validation is performed. This method is called whenever a parameter has been changed."""

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """Execute the tool."""
        execute_tool(arcsdm.mlp.Execute_MLP_regressor, self, parameters, messages)


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
        param_transformed_data.value = 'transformed_raster'

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
            parameters[4].value = "transformed_raster"

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
        param_transformed_data.value = 'transformed_vector'

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
            parameters[6].value = "transformed_vector"
        
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
    

   
class MLPRegressorTestTool(object):
    def __init__(self):
        """Test trained machine learning regressor model by predicting and scoring."""
        self.label = "Test MLP Regressor"
        self.description = "Test trained machine learning regressor model by predicting and scoring."
        self.canRunInBackground = False
        self.category = f"{TS_PREDICTIVE_MODELING}\\{TS_MACHINE_LEARNING}\\{TS_MODELING}"

    def getParameterInfo(self):
        """Define parameter definitions"""

        param_X = arcpy.Parameter(
            displayName="Input Features",
            name="X",
            datatype=["GPRasterLayer"],
            parameterType="Required",
            multiValue=True,
            direction="Input")

        param_y = arcpy.Parameter(
            displayName="Target Labels",
            name="y",
            datatype=["GPRasterLayer", "GPFeatureLayer"],
            parameterType="Required",
            direction="Input")
        
        param_y_attribute = arcpy.Parameter(
            displayName="Target Labels attribute",
            name="y_attribute",
            datatype="Field",
            parameterType="Optional",
            direction="Input")
        
        param_y_attribute.parameterDependencies = [param_y.name]
 
        param_X_nodata_value = arcpy.Parameter(
            displayName="Input Feature NoData Value",
            name="X_nodata_value",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input")
        param_X_nodata_value.value = -99
        
        param_y_nodata_value = arcpy.Parameter(
            displayName="Label NoData Value",
            name="y_nodata_value",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input")

        param_model_file = arcpy.Parameter(
            displayName="Input Model File",
            name="model_file",
            datatype="DEFile",
            parameterType="Required",
            direction="input")
        
        param_output_raster = arcpy.Parameter(
            displayName="Save output raster to a file",
            name="raster_file",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"
        )

        param_test_metrics = arcpy.Parameter(
            displayName="Test metrics",
            name="test_metrics",
            datatype="String",
            parameterType="Required",
            direction="Input",
            multiValue=True
        )

        param_test_metrics.filter.type = "ValueList"
        param_test_metrics.filter.list = ["mse", "rmse", "mae", "r2"]
        param_test_metrics.value = "mse"

        params = [param_X,
                  param_y,
                  param_y_attribute,
                  param_X_nodata_value,
                  param_y_nodata_value,
                  param_model_file,
                  param_output_raster,
                  param_test_metrics
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
        """Modify the values and properties of parameters before internal validation is performed. This method is called whenever a parameter has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """Execute the tool."""
        execute_tool(arcsdm.mlp.Execute_MLP_regressor_test, self, parameters, messages)



class MLPClassifierTestTool(object):
    def __init__(self):
        """Test trained machine learning classifier model by predicting and scoring."""
        self.label = "Test MLP Classifier"
        self.description = "Test trained machine learning classifier model by predicting and scoring."
        self.canRunInBackground = False
        self.category = f"{TS_PREDICTIVE_MODELING}\\{TS_MACHINE_LEARNING}\\{TS_MODELING}"


    def getParameterInfo(self):
        """Define parameter definitions"""

        param_X = arcpy.Parameter(
            displayName="Input Features",
            name="X",
            datatype=["GPRasterLayer"],
            parameterType="Required",
            multiValue=True,
            direction="Input")

        param_y = arcpy.Parameter(
            displayName="Target Labels",
            name="y",
            datatype=["GPRasterLayer", "GPFeatureLayer"],
            parameterType="Required",
            direction="Input")

        param_y_attribute = arcpy.Parameter(
            displayName="Target Labels attribute",
            name="y_attribute",
            datatype="Field",
            parameterType="Optional",
            direction="Input")
        
        param_y_attribute.parameterDependencies = [param_y.name]

        param_X_nodata_value = arcpy.Parameter(
            displayName="Input Feature NoData Value",
            name="X_nodata_value",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input")
        param_X_nodata_value.value = -99

        param_y_nodata_value = arcpy.Parameter(
            displayName="Label NoData Value",
            name="y_nodata_value",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input")

        param_model_file = arcpy.Parameter(
            displayName="Input Model File",
            name="model_file",
            datatype="DEFile",
            parameterType="Required",
            direction="input")

        param_classification_threshold = arcpy.Parameter(
            displayName="Classification threshold",
            name="Output_classification_threshold",
            datatype="GPDouble",
            parameterType="Required",
            direction="Output"
        )
        param_classification_threshold.value = 0.5 

        param_pred_probability_raster_output = arcpy.Parameter(
            displayName="Output predicted values probability raster",
            name="Output_Predicted_values_probability_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"
        )
        param_pred_probability_raster_output.value = "classifier_probability_test_result"

        param_pred_classified_raster_output = arcpy.Parameter(
            displayName="Output predicted values classified raster",
            name="Output_Predicted_values_classified_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"
        )
        param_pred_classified_raster_output.value = "classifier_classified_test_result"

        param_test_metrics = arcpy.Parameter(
            displayName="Test Metrics",
            name="test_metrics",
            datatype="String",
            parameterType="Required",
            direction="Input",
            multiValue=True
        )

        param_test_metrics.filter.type = "ValueList"
        param_test_metrics.filter.list = ["accuracy", "precision", "recall", "f1"]

        params = [param_X,
                  param_y,
                  param_y_attribute,
                  param_X_nodata_value,
                  param_y_nodata_value,
                  param_model_file,
                  param_classification_threshold,
                  param_pred_probability_raster_output,
                  param_pred_classified_raster_output,
                  param_test_metrics,
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
        """Modify the values and properties of parameters before internal validation is performed. This method is called whenever a parameter has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """Execute the tool."""
        execute_tool(arcsdm.mlp.Execute_MLP_classifier_test, self, parameters, messages)


class RegressorPredictTool(object):
    def __init__(self):
        """Predict with a trained machine learning regressor model."""
        self.label = "Predict Regressor"
        self.description = "Predict with a trained machine learning regressor model."
        self.canRunInBackground = False
        self.category = f"{TS_PREDICTIVE_MODELING}\\{TS_MACHINE_LEARNING}\\{TS_MODELING}"

    def getParameterInfo(self):
        """Define parameter definitions"""

        param_X = arcpy.Parameter(
            displayName="Input Features",
            name="X",
            datatype=["GPRasterLayer"],
            parameterType="Required",
            multiValue=True,
            direction="Input")

        param_X_nodata_value = arcpy.Parameter(
            displayName="Input Feature NoData Value",
            name="X_nodata_value",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input")
        param_X_nodata_value.value = -99

        param_y_nodata_value = arcpy.Parameter(
            displayName="Label NoData Value",
            name="y_nodata_value",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input")

        param_model_file = arcpy.Parameter(
            displayName="Input Model File",
            name="model_file",
            datatype="DEFile",
            parameterType="Required",
            direction="input")

        param_pred_probability_raster_output = arcpy.Parameter(
            displayName="Output predicted values probability raster",
            name="Output_Predicted_values_probability_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"
        )
        param_pred_probability_raster_output.value = "classifier_probability_test_result"

        params = [param_X,
                  param_X_nodata_value,
                  param_y_nodata_value,
                  param_model_file,
                  param_pred_probability_raster_output
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
        """Modify the values and properties of parameters before internal validation is performed. This method is called whenever a parameter has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """Execute the tool."""
        execute_tool(arcsdm.mlp.Execute_regressor_predict, self, parameters, messages)
        

class ClassifierPredictTool(object):
    def __init__(self):
        """Predict with a trained machine learning classifier model."""
        self.label = "Predict Classifier"
        self.description = "Predict with a trained machine learning classifier model."
        self.canRunInBackground = False
        self.category = f"{TS_PREDICTIVE_MODELING}\\{TS_MACHINE_LEARNING}\\{TS_MODELING}"

    def getParameterInfo(self):
        """Define parameter definitions"""

        param_X = arcpy.Parameter(
            displayName="Input Features",
            name="X",
            datatype=["GPRasterLayer"],
            parameterType="Required",
            multiValue=True,
            direction="Input")

        param_X_nodata_value = arcpy.Parameter(
            displayName="Input Feature NoData Value",
            name="X_nodata_value",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input")
        param_X_nodata_value.value = -99

        param_y_nodata_value = arcpy.Parameter(
            displayName="Label NoData Value",
            name="y_nodata_value",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input")

        param_model_file = arcpy.Parameter(
            displayName="Input Model File",
            name="model_file",
            datatype="DEFile",
            parameterType="Required",
            direction="input")

        param_classification_threshold = arcpy.Parameter(
            displayName="Classification threshold",
            name="Output_classification_threshold",
            datatype="GPDouble",
            parameterType="Required",
            direction="Output"
        )
        param_classification_threshold.value = 0.5 

        param_pred_probability_raster_output = arcpy.Parameter(
            displayName="Output predicted values probability raster",
            name="Output_Predicted_values_probability_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"
        )
        param_pred_probability_raster_output.value = "classifier_probability_test_result"

        param_pred_classified_raster_output = arcpy.Parameter(
            displayName="Output predicted values classified raster",
            name="Output_Predicted_values_classified_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"
        )
        param_pred_classified_raster_output.value = "classifier_classified_test_result"

        params = [param_X,
                  param_X_nodata_value,
                  param_y_nodata_value,
                  param_model_file,
                  param_classification_threshold,
                  param_pred_probability_raster_output,
                  param_pred_classified_raster_output
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
        """Modify the values and properties of parameters before internal validation is performed. This method is called whenever a parameter has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """Execute the tool."""
        execute_tool(arcsdm.mlp.Execute_classifier_predict, self, parameters, messages)
