import sys
import arcpy

import arcsdm.sitereduction
import arcsdm.logisticregression
import arcsdm.calculateweights
import arcsdm.categoricalreclass
import arcsdm.categoricalmembership
import arcsdm.tocfuzzification
import arcsdm.calculateresponse
import arcsdm.symbolize
import arcsdm.roctool
import arcsdm.acterbergchengci
import arcsdm.rescale_raster;
from arcsdm.areafrequency import Execute
import arcsdm.nninputfiles
import arcsdm.grand_wofe_lr
import arcsdm.fuzzyroc
import arcsdm.fuzzyroc2
import arcsdm.mlp
from arcsdm.common import execute_tool


import importlib
from imp import reload;



class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the .pyt file)."""
        
        self.label = "ArcSDM Tools"
        self.alias = "ArcSDM" 

        # List of tool classes associated with this toolbox
        self.tools = [PartitionNNInputFiles, CombineNNOutputFiles, NeuralNetworkOutputFiles, NeuralNetworkInputFiles, 
        CalculateWeightsTool,SiteReductionTool,CategoricalMembershipTool,
        CategoricalAndReclassTool, TOCFuzzificationTool, CalculateResponse, FuzzyROC, FuzzyROC2, LogisticRegressionTool, Symbolize, 
        ROCTool, AgterbergChengCITest, AreaFrequencyTable, GetSDMValues, GrandWofe, TrainMLPClassifierTool]


class GrandWofe(object):
    
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Grand WOFE"
        self.description = "From list of Evidence layers generate weights tables and output rasters from Calculate Respons and Logistic Regression."
        self.canRunInBackground = False
        self.category = "Weights of Evidence"

    def getParameterInfo(self):
        """Define parameter definitions"""
        
        param1 = arcpy.Parameter(
        displayName="Grand WOFE name",
        name="wofename",
        #datatype="DEFeatureClass",
        datatype="String",
        parameterType="Required",
        direction="Input")
        
        param2 = arcpy.Parameter(
        displayName="Input raster names",
        name="rasternames",
        #datatype="DEFeatureClass",
        datatype="GPRasterLayer",
        multiValue=1,        
        parameterType="Required",
        direction="Input")
        
        param3 = arcpy.Parameter(
        displayName="Input raster types (use a, d, or c and separate by semicolon ;)",
        name="rastertypes",
        #datatype="DEFeatureClass",
        datatype="String",
        #multiValue=1,        
        parameterType="Required",
        direction="Input")
        
        paramTrainingPoints = arcpy.Parameter(
        displayName="Training points",
        name="Training_points",
        datatype="GPFeatureLayer",
        parameterType="Required",
        direction="Input")
        
        paramIgnoreMissing = arcpy.Parameter(
        displayName="Ignore missing data (must be set to -99)",
        name="Ignore missing data (-99)",
        datatype="Boolean",
        parameterType="Optional",
        direction="Input")
        #paramIgnoreMissing.value= false;
        
        paramContrast = arcpy.Parameter(
        displayName="Contrast Confidence Level",
        name="contrast",
        datatype="GPDouble",
        parameterType="Required",
        direction="Input")
        paramContrast.value = "2"
        
        paramUnitArea = arcpy.Parameter(
        displayName="Unit area (km2)",
        name="Unit_Area__sq_km_",
        datatype="GPDouble",
        parameterType="Required",
        direction="Input")
        paramUnitArea.value = "1"
        
       
        
        
        
        params = [param1, param2, param3, paramTrainingPoints, paramIgnoreMissing, paramContrast, paramUnitArea]
        return params

    def isLicensed(self):    
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False  # tool cannot be executed
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
        #try:
        #    importlib.reload (arcsdm.grand_wofe_lr)
        #except :
        #    reload(arcsdm.grand_wofe_lr);
        #arcsdm.grand_wofe_lr.execute(self, parameters, messages)
        execute_tool(arcsdm.grand_wofe_lr.execute, self, parameters, messages) #AL 090620
        return
                
        
        

class PartitionNNInputFiles(object):
    
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Partition NNInput Files"
        self.description = "Partitions Neural Network class.dta of more than 200,000 records into files of 200,000 or less."
        self.canRunInBackground = False
        self.category = "Neural network"

    def getParameterInfo(self):
        """Define parameter definitions"""
        
        paramInputFiles = arcpy.Parameter(
        displayName="Input class.dat file",
        name="inputfiles",
        #datatype="DEFeatureClass",
        datatype="File",
        parameterType="Required",
        direction="Input")
        
        params = [paramInputFiles]
        return params

    def isLicensed(self):    
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False  # tool cannot be executed
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
            importlib.reload (arcsdm.partition_inputnnfiles)
        except :
            reload(arcsdm.partition_inputnnfiles);
        
        arcsdm.partition_inputnnfiles.execute(self, parameters, messages)
        return
                
        
        
class CombineNNOutputFiles(object):
    
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Combine NNOutput Files "
        self.description = "Combines PNN, FUZ, and RBN files generated from partitions of the class.dta file."
        self.canRunInBackground = False
        self.category = "Neural network"

    def getParameterInfo(self):
        """Define parameter definitions"""
        
        paramInputFiles = arcpy.Parameter(
        displayName="Input RBN, FUZ, PNN files",
        name="inputfiles",
        #datatype="DEFeatureClass",
        datatype="File",
        multiValue=1,
        parameterType="Required",
        direction="Input")
        

        paramOutputFile = arcpy.Parameter(
        displayName="Output file",
        name="outputfile",
        datatype="file",
        parameterType="Required",
        direction="Output")
        params = [paramInputFiles, paramOutputFile]
        return params

    def isLicensed(self):    
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False  # tool cannot be executed
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
            importlib.reload (arcsdm.combine_outputnnfiles)
        except :
            reload(arcsdm.combine_outputnnfiles);
        arcsdm.combine_outputnnfiles.execute(self, parameters, messages)
        return
        
                
        
        
class NeuralNetworkOutputFiles(object):
    
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Neural network output files"
        self.description = "Generate files from output files of GeoXplore"
        self.canRunInBackground = False
        self.category = "Neural network"

    def getParameterInfo(self):
        """Define parameter definitions"""
        
        paramInputRaster = arcpy.Parameter(
        displayName="Unique Conditions raster",
        name="inputraster",
        #datatype="DEFeatureClass",
        datatype="GPRasterLayer",
        parameterType="Required",
        direction="Input")
        
        paramRBFNFile = arcpy.Parameter(
        displayName="RBFN file name, .rbn file",
        name="rbfnfile",
        datatype="File",
        parameterType="Optional",
        direction="Input")     
        
        paramPNNFile = arcpy.Parameter(
        displayName="PNN file name, .pnn file",
        name="pnnfile",
        datatype="File",
        parameterType="Optional",
        direction="Input")     
        
        paramFuzFile = arcpy.Parameter(
        displayName="Fuzzy Classification file name, .fuz file",
        name="fuzfile",
        datatype="File",
        parameterType="Optional",
        direction="Input")     
        
        paramOutputTable = arcpy.Parameter(
        displayName="Output result table",
        name="resulttable",
        datatype="DeTable",
        parameterType="Required",
        direction="Output")     

        param_outras = arcpy.Parameter(
        displayName="Output raster",
        name="outputraster",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output")
        # Is this needed?
        #param_pprb.value = "%Workspace%\neuralnetwork_outras"                      
        # End                                       
        params = [paramInputRaster, paramRBFNFile, paramPNNFile, paramFuzFile, paramOutputTable, param_outras]
        return params

    def isLicensed(self):    
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False  # tool cannot be executed
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
            importlib.reload (arcsdm.nnoutputfiles)
        except :
            reload(arcsdm.nnoutputfiles);
        arcsdm.nnoutputfiles.execute(self, parameters, messages)
        return
        
                
        
                
         
class NeuralNetworkInputFiles(object):
    
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Neural network input files"
        self.description = "Use this tool to create the input ASCII files for the GeoXplore neural network. Before using this tool, the evidence must be combined into a unique conditions raster with the Combine tool and the band statistics must be obtained for all the evidence using the Band Collection Statistics tool. If desired fuzzy membership attribute can be added to each of the training sites. See the ArcMap Tools Options discussion in Usage Tips in the Help about adjusting default setting for this tool."
        self.canRunInBackground = False
        self.category = "Neural network"

    def getParameterInfo(self):
        """Define parameter definitions"""
        
        paramInputRaster = arcpy.Parameter(
        displayName="Input Unique Conditions raster",
        name="inputraster",
        #datatype="DEFeatureClass",
        datatype="GPRasterLayer",
        parameterType="Required",
        direction="Input")
        
        paramTrainingSites = arcpy.Parameter(
        displayName="Training sites",
        name="training_sites",
        datatype="GPFeatureLayer",
        parameterType="Required",
        direction="Input")

        
        paramFZMField = arcpy.Parameter(
        displayName="TP Fuzzy membership field",
        name="fzmfield",
        datatype="Field",
        parameterType="Optional",
        direction="Input")               
        paramFZMField.parameterDependencies = [paramTrainingSites.name]                
        
        paramNDTrainingSites = arcpy.Parameter(
        displayName="ND Training sites",
        name="ndtraining_sites",
        datatype="GPFeatureLayer",
        parameterType="Required",
        direction="Input")

        paramNDFZMField = arcpy.Parameter(
        displayName="ND TP Fuzzy membership field",
        name="ndfzmfield",
        datatype="Field",
        parameterType="Optional",
        direction="Input")               
        paramNDFZMField.parameterDependencies = [paramNDTrainingSites.name]                
        
        paramTrainingFilePrefix = arcpy.Parameter(
        displayName="Training file prefix",
        name="trainingfileprefix",
        datatype="String",
        parameterType="Required",
        direction="Input")     
        
        paramClassificationFile = arcpy.Parameter(
        displayName="Classification file",
        name="classificationfile",
        datatype="Boolean",
        parameterType="Optional",
        direction="Input")     
        
        paramBandStatisticsFile = arcpy.Parameter(
        displayName="Band statistics file",
        name="bandstatisticsfile",
        datatype="File",
        parameterType="Optional",
        direction="Input")     
        
        paramTrainFileOutput = arcpy.Parameter(
        displayName="Train file output",
        name="trainfileoutput",
        datatype="File",
        parameterType="Required",
        direction="Output")     
        
        paramClassFileOutput = arcpy.Parameter(
        displayName="Class file output",
        name="classfileoutput",
        datatype="File",
        parameterType="Required",
        direction="Output")     
        
        
        
        
        
        paramUnitArea = arcpy.Parameter(
        displayName="Unit area (km2)",
        name="Unit_Area__sq_km_",
        datatype="GPDouble",
        parameterType="Required",
        direction="Input")
        paramUnitArea.value = "1"
        
                                  
        params = [paramInputRaster, paramTrainingSites, paramFZMField, paramNDTrainingSites, paramNDFZMField, 
            paramTrainingFilePrefix, paramClassificationFile,
            paramBandStatisticsFile, paramTrainFileOutput, paramClassFileOutput]
        return params

    def isLicensed(self):    
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False  # tool cannot be executed
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
            importlib.reload (arcsdm.nninputfiles)
        except :
            reload(arcsdm.nninputfiles);
        #messages.AddMessage("Waiting for debugger")
        #wait_for_debugger(15);
        arcsdm.nninputfiles.execute(self, parameters, messages)
        return
        
        
class GetSDMValues(object):                
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Get SDM parameters"
        self.description = "This tool is used to view the Environment and SDM modeling parameters that have been set by the user. All of the values reported by this tool must be set to values specific to the model to be made. Using the ESRI default values will cause SDM to fail. If the Environment is not completely set, then an error message stating \"Improper SDM setup\" will occur. The successful running of this tool does not assure that the setup is correct; only that the default values have been changed. See the Environment Settings section of the Help file for Calculate Weights for the details."

        self.canRunInBackground = False
        self.category = "Utilities"

    def getParameterInfo(self):
        """Define parameter definitions"""
               
        
        paramTrainingSites = arcpy.Parameter(
        displayName="Training sites",
        name="training_sites",
        datatype="GPFeatureLayer",
        parameterType="Required",
        direction="Input")

        
        
        paramUnitArea = arcpy.Parameter(
        displayName="Unit area (km2)",
        name="Unit_Area__sq_km_",
        datatype="GPDouble",
        parameterType="Required",
        direction="Input")
        paramUnitArea.value = "1"
        
                                  
        params = [paramTrainingSites, paramUnitArea]
        return params

    def isLicensed(self):    
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False  # tool cannot be executed
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
            importlib.reload (arcsdm.sdmvalues)
        except :
            reload(arcsdm.sdmvalues);
        #messages.AddMessage("Waiting for debugger")
        #wait_for_debugger(15);
        arcsdm.sdmvalues.execute(self, parameters, messages)
        return
                                                             
        
        
class AreaFrequencyTable(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Area Frequency Table"
        self.description = "Create a table for charting area of evidence classes vs number of training sites."
        self.canRunInBackground = False
        self.category = "Weights of Evidence"

    def getParameterInfo(self):
        """Define parameter definitions"""
        
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
        parameterType="Optional",
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
        paramOutputTable.value = "%Workspace%\AreaFrequencyTable"
                
    
        
        
                                  
        params = [paramTrainingSites, paramRaster, paramField, paramUnitArea, paramOutputTable]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False  # tool cannot be executed
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
        arcsdm.areafrequency.Execute(self, parameters, messages)
        return
        
        
        
        

class ROCTool(object):
    def __init__(self):
        self.label = "Calculate ROC Curves and AUC Values"
        self.description = "Calculates Receiver Operator Characteristic curves and Areas Under the Curves"
        self.category = "ROC Tool"
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
            return False  # tool cannot be executed
        return True

    def execute(self, parameters, messages):        
        #execute_tool(arcsdm.roctool.execute, self, parameters, messages)
        try:
            importlib.reload (arcsdm.roctool)
        except :
            reload(arcsdm.roctool);
        arcsdm.roctool.execute (self, parameters, messages);
        return
        
class Symbolize(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Symbolize raster with priorprobability (classified values)"
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
        param5.value = "1";
                                  
        params = [param0, param2, param5]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False  # tool cannot be executed
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
        execute_tool(arcsdm.symbolize.execute, self, parameters, messages)
        return
        
        
class CalculateResponse(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Calculate response"
        self.description = "Use this tool to combine the evidence weighted by their associated generalization in the weights-of-evidence table. This tool calculates the posterior probability, standard deviation (uncertainty) due to weights, variance (uncertainty) due to missing data, and the total standard deviation (uncertainty) based on the evidence and how the evidence is generalized in the associated weights-of-evidence tables.The calculations use the Weight and W_Std in the weights table from Calculate Weights."
        self.canRunInBackground = False
        self.category = "Weights of Evidence"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
        displayName="Input Raster Layer(s)",
        name="Input_evidence_raster_layers",
        datatype="GPValueTable",
        parameterType="Required",
        direction="Input")
        param0.columns = [['GPRasterLayer', 'Evidence raster']]
        
        param1 = arcpy.Parameter(
        displayName="Evidence type",
        name="Evidence_Type",
        datatype="GPValueTable",
        parameterType="Required",
        direction="Input")
        param1.columns = [['GPString', 'Evidence type']]
        param1.parameterDependencies = ["0"];
        
        #param1.filter.type = "ValueList";
        #param1.filter.list = ["o", "c"];
        #param1.value = "o";
        
        paramInputWeights = arcpy.Parameter(
        displayName="Input weights tables",
        name="input_weights_tables",
        datatype="GPValueTable",
        parameterType="Required",
        direction="Input")
        paramInputWeights.columns = [['DETable', 'Weights table']]
        
        param2 = arcpy.Parameter(
        displayName="Training sites",
        name="training_sites",
        #datatype="DEFeatureClass",
        datatype="GPFeatureLayer",
        parameterType="Required",
        direction="Input")
        
        paramIgnoreMissing = arcpy.Parameter(
        displayName="Ignore missing data",
        name="Ignore missing data",
        datatype="Boolean",
        parameterType="Optional",
        direction="Input")
        #paramIgnoreMissing.value= false;
        
        param3 = arcpy.Parameter(
        displayName="Missing data value",
        name="Missing_Data_Value",
        datatype="GPLong",
        
        #parameterType="Required",
        direction="Input")
        param3.value= -99;

        param4 = arcpy.Parameter(
        displayName="Unit area (km^2)",
        name="Unit_Area_sq_km",
        datatype="GPDouble",
        parameterType="Required",
        direction="Input")
        param4.value = "1";
        
        param_pprb = arcpy.Parameter(
        displayName="Output post probablity raster",
        name="Output_Post_Probability_raster",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output")
        param_pprb.value = "%Workspace%\W_pprb"
        
        param_std = arcpy.Parameter(
        displayName="Output standard deviation raster",
        name="Output_Standard_Deviation_raster",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output")
        param_std.value = "%Workspace%\W_std"
        
        param_md_varianceraster = arcpy.Parameter(
        displayName="Output MD variance raster",
        name="output_md_variance_raster",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output")
        param_md_varianceraster.value = "%Workspace%\W_MDvar"
                
        param_totstddev = arcpy.Parameter(
        displayName="Output Total Std Deviation Raster",
        name="output_total_std_dev_raster",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output")
        param_totstddev.value = "%Workspace%\W_Tstd"
        
        param_Confraster = arcpy.Parameter(
        displayName="Output confidence raster",
        name="Output_Confidence_raster",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output")
        param_Confraster.value = "%Workspace%\W_conf"
                                  
        params = [param0, paramInputWeights, param2, paramIgnoreMissing, param3, param4,  param_pprb, param_std, param_md_varianceraster, param_totstddev,  param_Confraster]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False  # tool cannot be executed        
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
        execute_tool(arcsdm.calculateresponse.Execute, self, parameters, messages)
        return
        

class CalculateWeightsTool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Calculate Weights"
        self.description = "Calculate weight rasters from the inputs"
        self.canRunInBackground = False
        self.category = "Weights of Evidence"

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
        param2.value = "";
        
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
        param5.value = "1";
        
        param6 = arcpy.Parameter(
        displayName="Missing data value",
        name="Missing_Data_Value",
        datatype="GPLong",
        parameterType="Required",
        direction="Input")
        param6.value = "-99";
                          

        paramSuccess = arcpy.Parameter(
        displayName="Success",
        name="success",
        datatype="Boolean",
        parameterType="Optional",
        direction="Output")


                          
        params = [param0, param1, paramTrainingPoints, param2, param3, param4, param5, param6, paramSuccess]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False  # tool cannot be executed        
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        if parameters[0].value and parameters[3].value:
            if (parameters[0].altered or paramaters[3].altered) and not parameters[4].altered:
                layer = parameters[0].valueAsText;
                desc = arcpy.Describe(layer)
                name = desc.file;
                type = parameters[3].valueAsText;
                char = type[:1];
                if (char != 'U'):
                    if (char != 'C'):
                        char = 'C' + char; #Output  _C + first letter of type unless it is U
                    else:
                        char = 'CT'; # Unless it is C, then it is CT... 
                #Update name accordingly
                resulttmp = "%WORKSPACE%\\" + name + "_" + char;
                #parameters[4].value = resulttmp.replace(".","");  #Remove illegal characters
                resulttmp = resulttmp.replace(".","");
                #Add .dbf to Weights Table Name if Workspace is not File Geodatabase #AL 250820
                #If using GDB database, remove numbers and underscore from the beginning of the name (else block) #AL 071020
                if not ".gdb" in arcpy.env.workspace:
                    resulttmp = resulttmp + ".dbf"
                else:
                    wtsbase = os.path.basename(resulttmp)
                    while len(wtsbase) > 0 and (wtsbase[:1] <= "9" or wtsbase[:1] == "_"):
                        wtsbase = wtsbase[1:]
                    resulttmp = os.path.dirname(resulttmp) + "\\" + wtsbase
                parameters[4].value = resulttmp
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
     
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        execute_tool(arcsdm.calculateweights.Calculate, self, parameters, messages)
        return

        
class SiteReductionTool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Training sites reduction"
        self.description = "Selects subset of the training points"
        self.canRunInBackground = False
        self.category = "Weights of Evidence"

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
        datatype="GPDouble",
        parameterType="Optional",
        direction="Input")

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

        param4.filter.type = "Range"
        param4.filter.list = [1, 100]
        
        param5 = arcpy.Parameter(
        displayName="Save selection as a new layer",
        name="layerSelection",
        datatype="GPFeatureLayer",
        parameterType="Optional",
        direction="Derived")
                                            
        params = [param0, param1, param2, param3, param4, param5]
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
        validation is performed.  This method is called whenever a parameter
        has been changed."""

        parameters[2].enabled = parameters[1].value
        parameters[4].enabled = parameters[3].value

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""

        if not (parameters[1].value or parameters[3].value):
            parameters[1].setErrorMessage("You have to select at least one!")
            parameters[3].setErrorMessage("You have to select at least one!")
        else:
            if parameters[1].value:
                if not parameters[2].valueAsText:
                    parameters[2].setErrorMessage("Thinning value required!")
            
            if parameters[3].value:
                if not parameters[4].valueAsText:
                    parameters[4].SetErrorMessage("Percentage value required!")
        
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        #For full debugging this disabled:
        #execute_tool(arcsdm.sitereduction.ReduceSites, self, parameters, messages)
        try:
            importlib.reload (arcsdm.sitereduction)
        except :
            reload(arcsdm.sitereduction)
        arcsdm.sitereduction.ReduceSites(self, parameters, messages)
        return
        
class CategoricalMembershipTool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Categorical Membership"
        self.description = "Create fuzzy memberships for categorical data by first reclassification to integers and then division by an appropriate value"
        self.canRunInBackground = False
        self.category = "Fuzzy Logic\\Fuzzy Membership"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
        displayName="Categorical evidence raster",
        name="categorical_evidence",
        datatype="GPRasterLayer",
        parameterType="Required",
        direction="Input")

        param1 = arcpy.Parameter(
        displayName="Reclassification",
        name="reclassification",
        datatype="GPTableView",
        parameterType="Required",
        direction="Input")

        param2 = arcpy.Parameter(
        displayName="Rescale Constant",
        name="rescale_constant",
        datatype="GPLong",
        parameterType="Required",
        direction="Input")

        param3 = arcpy.Parameter(
        displayName="FMCat",
        name="fmcat",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output")
        
        params = [param0, param1, param2, param3]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False  # tool cannot be executed        
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
        execute_tool(arcsdm.categoricalmembership.Calculate, self, parameters, messages)
        return


class CategoricalAndReclassTool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Categorical & Reclass"
        self.description = "Create fuzzy memberships for categorical data by first reclassification to integers and then division by an appropriate value."
        self.canRunInBackground = False
        self.category = "Fuzzy Logic\\Fuzzy Membership"

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
            return False  # tool cannot be executed        
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
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
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        execute_tool(arcsdm.categoricalreclass.Calculate, self, parameters, messages)
        return

class TOCFuzzificationTool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "TOC Fuzzification"
        self.description = "This fuzzification method utilized the symbolization of the input raster that has been applied in the map document table of contects (TOC). The symbolization in the TOC defines the number of classes and this tool rescales those classes (1...N) to the range [0,1] by (C - 1)/(N-1) where C is the class value and N is the number of classes."
        self.canRunInBackground = False
        self.category = "Fuzzy Logic\\Fuzzy Membership"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
        displayName="Input Raster",
        name="input_raster",
        datatype="GPRasterLayer",
        parameterType="Required",
        direction="Input")
        
        param1 = arcpy.Parameter(
        displayName="Reclass Field",
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
        displayName="Number of Classes",
        name="classes",
        datatype="GPLong",
        parameterType="Required",
        direction="Input")

        param4 = arcpy.Parameter(
        displayName="Fuzzy Membership Raster",
        name="fmtoc",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output")
        
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
            return False  # tool cannot be executed        
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
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
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        execute_tool(arcsdm.tocfuzzification.Calculate, self, parameters, messages)
        return
        

class LogisticRegressionTool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Logistic regression"
        self.description = "This tool is a useful complement to Weights-of-Evidence Calculate Response tool as Logistic Regression does not make the assumption of conditional independence of the evidence with regards to the training sites. Using the evidence and assocaited weights tables, this tool creates the outputs the response and standard deviation rasters. The calculations are based on the Gen_Class attribute in the weights table and the type of evidence. Please note that the Logistic Regression tool accepts a maximum of 6,000 unique conditions or it fails. Also note that there is an upper limit of 100,000 unit cells per class in each evidence raster layer. If a class in an evidence raster goes above this, the script contains a function to increase the unit cell size to ensure an upper limit of 100,000. These issues are unable to be fixed due to a hard coded limitation in the Logistic Regression executable sdmlr.exe."
        self.canRunInBackground = False
        self.category = "Weights of Evidence"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
        displayName="Input Raster Layer(s)",
        name="Input_evidence_raster_layers",
        datatype="GPValueTable",
        parameterType="Required",
        direction="Input")
        param0.columns = [['GPRasterLayer', 'Evidence raster']]
        
        param1 = arcpy.Parameter(
        displayName="Evidence types (use a, c, d, or o and separate by semicolon ;)",
        name="Evidence_Type",
        #datatype="GPValueTable",
        datatype="String",
        parameterType="Required",
        direction="Input")
        #param1.columns = [['GPString', 'Evidence type']]
        #param1.parameterDependencies = ["0"];
        #param1.filter.type = "ValueList";
        #param1.filter.list = ["o", "c", "a", "d"];
        #param1.value = "o";

        paramInputWeights = arcpy.Parameter(
        displayName="Input weights tables",
        name="input_weights_tables",
        datatype="GPValueTable",
        parameterType="Required",
        direction="Input")
        paramInputWeights.columns = [['DETable', 'Weights table']]

        param2 = arcpy.Parameter(
        displayName="Training sites",
        name="training_sites",
        #datatype="DEFeatureClass",
        datatype="GPFeatureLayer",
        parameterType="Required",
        direction="Input")
        
        param3 = arcpy.Parameter(
        displayName="Missing data value",
        name="Missing_Data_Value",
        datatype="GPLong",
        parameterType="Required",
        direction="Input")
        param3.value= -99;

        param4 = arcpy.Parameter(
        displayName="Unit area (km^2)",
        name="Unit_Area_sq_km",
        datatype="GPDouble",
        parameterType="Required",
        direction="Input")
        param4.value = "1";
        
        param5 = arcpy.Parameter(
        displayName="Output polynomial table",
        name="Output_Polynomial_Table",
        datatype="DEDbaseTable",
        parameterType="Required",
        direction="Output")
        param5.value = "%Workspace%\LR_logpol"
                
        param52 = arcpy.Parameter(
        displayName="Output coefficients table",
        name="Output_Coefficients_Table",
        datatype="DEDbaseTable",
        parameterType="Required",
        direction="Output")
        param52.value = "%Workspace%\LR_coeff"
        
        param6 = arcpy.Parameter(
        displayName="Output post probablity raster",
        name="Output_Post_Probability_raster",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output")
        param6.value = "%Workspace%\LR_pprb"
        
        param62 = arcpy.Parameter(
        displayName="Output standard deviation raster",
        name="Output_Standard_Deviation_raster",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output")
        param62.value = "%Workspace%\LR_std"
        
        param63 = arcpy.Parameter(
        displayName="Output confidence raster",
        name="Output_Confidence_raster",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output")
        param63.value = "%Workspace%\LR_conf"
                                  
        params = [param0, param1, paramInputWeights, param2, param3, param4, param5, param52, param6, param62, param63]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False  # tool cannot be executed        
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
        try:
            importlib.reload (arcsdm.logisticregression)
        except :
            reload(arcsdm.logisticregression);
        arcsdm.logisticregression.Execute(self, parameters, messages)
        return
        
        #execute_tool(arcsdm.logisticregression.Execute, self, parameters, messages)

class AgterbergChengCITest(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Agterberg-Cheng CI Test"
        self.description = ""
        self.canRunInBackground = False
        self.category = "Weights of Evidence"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
        displayName="Post Probability raster",
        name="pp_raster",
        datatype="GPRasterLayer",
        parameterType="Required",
        direction="Input")

        param1 = arcpy.Parameter(
        displayName="Probability Std raster",
        name="ps_raster",
        datatype="GPRasterLayer",
        parameterType="Required",
        direction="Input")

        param2 = arcpy.Parameter(
        displayName="Training sites",
        name="training_sites",
        datatype="GPFeatureLayer",
        parameterType="Required",
        direction="Input")

        param3 = arcpy.Parameter(
        displayName="Unit area (km2)",
        name="Unit_Area__sq_km",
        datatype="GPDouble",
        parameterType="Required",
        direction="Input")

        param4 = arcpy.Parameter(
        displayName="Output CI Test File",
        name="ci_test_file",
        datatype="DEFile",
        parameterType="Optional",
        direction="Output")
                                  
        params = [param0, param1, param2, param3, param4]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False  # tool cannot be executed        
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
        execute_tool(arcsdm.acterbergchengci.Calculate, self, parameters, messages)
        return

class FuzzyROC(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Fuzzy ROC"
        self.description = "Fuzzy Membership + Fuzzy Overlay + ROC"
        self.canRunInBackground = False
        self.category = "Fuzzy Logic\\Fuzzy Membership"

    def getParameterInfo(self):
        """Define parameter definitions"""
        
        param0 = arcpy.Parameter(
        displayName="Input raster names",
        name="inputrasters",
        datatype="GPRasterLayer",
        multiValue=1,
        parameterType="Required",
        direction="Input")
        
        param1 = arcpy.Parameter(
        displayName="Fuzzy Membership Parameters",
        name="fmparams",
        datatype="DETable",
        parameterType="Required",
        direction="Input")

        param1.columns = [['String', 'Membership type'], ['String', 'Midpoint Min'], ['String', 'Midpoint Max'], ['String', 'Midpoint Count'], ['String', 'Spread Min'], ['String', 'Spread Max'], ['String', 'Spread Count']]
        param1.filters[0].type = 'ValueList'
        param1.filters[0].list = ['Small', 'Large']
        
        param2 = arcpy.Parameter(
        displayName="Fuzzy Overlay Parameters",
        name="foparams",
        datatype="DETable",
        parameterType="Required",
        direction="Input")

        param2.columns = [['String', 'Overlay type'], ['String', 'Parameter']]
        param2.filters[0].type = 'ValueList'
        param2.filters[0].list = ['And', 'Or', 'Product', 'Sum', 'Gamma']
        
        param3 = arcpy.Parameter(
        displayName="ROC True Positives Feature Class",
        name="truepositives",
        datatype="DEFeatureClass",
        parameterType="Required",
        direction="Input")

        param4 = arcpy.Parameter(
        displayName="ROC Destination Folder",
        name="dest_folder",
        datatype="DEFolder",
        parameterType="Required",
        direction="Input")
        param4.filter.list = ["File System"]

        params = [param0, param1, param2, param3, param4]
        return params

    def isLicensed(self):    
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False  # tool cannot be executed
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
        execute_tool(arcsdm.fuzzyroc.Execute, self, parameters, messages)
        return

class FuzzyROC2(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Fuzzy ROC 2"
        self.description = "Fuzzy Membership + Fuzzy Overlay + ROC (Receiver Operator Characteristic)"
        self.canRunInBackground = False
        self.category = "Fuzzy Logic\\Fuzzy Membership"

    def getParameterInfo(self):
        """Define parameter definitions"""
        
        param0 = arcpy.Parameter(
        displayName="Input rasters, Fuzzy Membership functions and parameters",
        name="inputrasters",
        datatype="DETable",
        multiValue=1,
        parameterType="Required",
        direction="Input")
        param0.columns = [['GPRasterLayer', 'Input raster name'], ['String', 'Membership type'], ['String', 'Midpoint Min'], ['String', 'Midpoint Max'], ['String', 'Midpoint Count'], ['String', 'Spread Min'], ['String', 'Spread Max'], ['String', 'Spread Count']]
        param0.filters[1].type = 'ValueList'
        param0.filters[1].list = ['Small', 'Large']

        param1 = arcpy.Parameter(
        displayName="Draw only Fuzzy Membership plots",
        name="plots",
        datatype="GPBoolean",
        parameterType="Optional",
        direction="Input")
        param1.value = False;
        
        param2 = arcpy.Parameter(
        displayName="\nTrue Positives Feature Class",
        name="truepositives",
        datatype="DEFeatureClass",
        parameterType="Required",
        direction="Input")

        param3 = arcpy.Parameter(
        displayName="Output Folder",
        name="output_folder",
        datatype="DEFolder",
        parameterType="Required",
        direction="Input")
        param3.filter.list = ["File System"]

        param4 = arcpy.Parameter(
        displayName="Fuzzy Overlay Parameters",
        name="foparams",
        datatype="DETable",
        parameterType="Required",
        direction="Input",
        enabled=True,
        category="Calculation")

        param4.columns = [['String', 'Overlay type'], ['String', 'Parameter']]
        param4.filters[0].type = 'ValueList'
        param4.filters[0].list = ['And', 'Or', 'Product', 'Sum', 'Gamma']
        param4.values = [['And', '0']]

        param5 = arcpy.Parameter(
        displayName="Plot display method",
        name="display_method",
        datatype="GPString",
        parameterType="Required",
        direction="Input",
        enabled=False,
        category='Plotting')
        param5.filter.type = "ValueList";
        param5.filter.list = ["To Window(s)", "To PDF file(s)", "To PNG file(s)"];
        param5.value = "To Window(s)";

        params = [param0, param1, param2, param3, param4, param5]
        return params

    def isLicensed(self):    
        """Set whether tool is licensed to execute."""
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
        except Exception:
            return False  # tool cannot be executed
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        if (parameters[1].value):
            parameters[4].enabled = False
            parameters[5].enabled = True
        else:
            parameters[4].enabled = True
            parameters[5].enabled = False
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
     
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        execute_tool(arcsdm.fuzzyroc2.Execute, self, parameters, messages)
        return
class TrainMLPClassifierTool(object):
    def __init__(self):
        """Train a Multi-Layer Perceptron (MLP) classifier with the given parameters."""
        self.label = "Train MLP Classifier"
        self.description = "Train a Multi-Layer Perceptron (MLP) classifier with the given parameters."
        self.canRunInBackground = False
        self.category = "Prediction"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param_X = arcpy.Parameter(
            displayName="Input Features",
            name="X",
            datatype="GPTableView",
            parameterType="Required",
            direction="Input")

        param_y = arcpy.Parameter(
            displayName="Target Variable",
            name="y",
            datatype="Field",
            parameterType="Required",
            direction="Input")
        param_y.parameterDependencies = [param_X.name]

        param_neurons = arcpy.Parameter(
            displayName="Neurons per Layer. A comma separeted list of integers: e.g. 10,5,10",
            name="neurons",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        param_neurons.info = "Enter the number of neurons for each layer separated by commas. For example, 100,50,25."

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

        params = [param_X,
                  param_y,
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
        execute_tool(arcsdm.mlp.Execute_train_MLP_classifier, self, parameters, messages)
        return
