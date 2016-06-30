import arcpy
import arcsdm.SiteReduction as SiteReduction;
import importlib;

class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        
        self.label = "ArcSDM"
        self.alias = "ArcSDM python toolbox"
        # List of tool classes associated with this toolbox
        self.tools = [SiteReductionTool]


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
        parameterType="Required",
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
        parameterType="Required",
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
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
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
