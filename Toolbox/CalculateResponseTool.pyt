import sys
import arcpy

import arcsdm;

from arcsdm import *
from arcsdm.sitereduction import ReduceSites
from arcsdm.calculateweights import Calculate
from arcsdm.categoricalmembership import Calculate
from arcsdm.calculateresponse import Execute

   

import importlib
from imp import reload;


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        
        self.label = "Calculate reseponse (tmp) toolbox"
        self.alias = "Calculate response tmp toolbox" 

        # List of tool classes associated with this toolbox
        self.tools = [CalculateResponse]


class CalculateResponse(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Calculate response"
        self.description = "TODO: Copy this from old toolbox"
        self.canRunInBackground = False
        #self.category = "Weights of Evidence"

    def getParameterInfo(self):
        """Define parameter definitions"""
        # TODO: Multiple rasters?
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
        parameterType="Required",
        direction="Output")
        #paramIgnoreMissing.value= false;
        
        
        param3 = arcpy.Parameter(
        displayName="Missing data value",
        name="Missing_Data_Value",
        datatype="GPLong",
        
        #parameterType="Required",
        direction="Output")
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
        datatype="DEDbaseTable",
        parameterType="Required",
        direction="Output")
        param_md_varianceraster.value = "%Workspace%\W_MDvar"
                
        param_totstddev = arcpy.Parameter(
        displayName="Output Total Std Deviation Raster",
        name="output_total_std_dev_raster",
        datatype="DEDbaseTable",
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
            importlib.reload (arcsdm.calculateresponse)
        except :
            reload(arcsdm.calculateresponse);
        # To list what functions does module contain
        #messages.addWarningMessage(dir(arcsdm.SiteReduction));
        #arcsdm.CalculateWeights.Calculate(self, parameters, messages);
        #messages.AddMessage("Waiting for debugger")
        #wait_for_debugger(15);
        #No do yet
        arcsdm.calculateresponse.Execute(self, parameters, messages)
        return
        
        