# -*- coding: utf-8 -*-

# ArcSDM 5 Common and general functions
#
# Janne Kallunki, GTK, 2017
# Tero Ronkko, GTK, 2018
# Arto Laiho, GTK, 29.4.2020 (see addToDisplay)

import sys, os, traceback
import arcpy
import arcsdm.config as cfg
from arcsdm.exceptions import SDMError

PY2 = sys.version_info[0] == 2
PY34 = sys.version_info[0:2] >= (3, 4)

if PY2:
    from imp import reload;
if PY34:
    import importlib

    
def testandwarn_arcgispro():
    installinfo = arcpy.GetInstallInfo ();
    if (installinfo['ProductName'] == "ArcGISPro"):
        arcpy.AddWarning("This tool does not work properly on ArcGISPro!");
        return True;
    else:
        return False;
        
        
def testandwarn_filegeodatabase_environment():
    desc = arcpy.Describe(arcpy.env.workspace)
    #arcpy.AddMessage("%-20s %s (%s)" % ("Workspace: ", arcpy.env.workspace, desc.workspaceType));
    wdesc = arcpy.Describe(arcpy.env.scratchWorkspace)       
    #arcpy.AddMessage("%-20s %s (%s)" % ("Scratch workspace:",  arcpy.env.scratchWorkspace, wdesc.workspaceType))
    if desc.workspaceType == "LocalDatabase" or wdesc.workspaceType == "LocalDatabase":
        arcpy.AddWarning("For this tool workspaces cannot be filegeodatabases!");
        return True;
    else:
        return False;
   
def testandwarn_filegeodatabase_source(resourcename):
    desc = arcpy.Describe(resourcename);
    #arcpy.AddMessage(desc.catalogpath);
    workspace = os.path.dirname(desc.catalogpath)
    #arcpy.AddMessage(workspace);        
    if [any(ext) for ext in ('.gdb', '.mdb', '.sde') if ext in os.path.splitext(workspace)]:
        workspace = workspace;
        arcpy.AddWarning("For this tool the source data cannot be in geodatabase format!");        
        return True;
    else:
        return False;
    

   
    
def reload_arcsdm_modules(messages):
    arcsdm_modules = [m.__name__ for m in sys.modules.values() if m and m.__name__.startswith(__package__)]
    for m in arcsdm_modules:
        try:
            reload_module(sys.modules[m])
        except Exception as e:
            messages.AddMessage("Failed to reload module %s. Reason:%s" %(m, e.message))
    messages.AddMessage("Reloaded %s modules" % __package__)

def reload_module(name):
    if PY2:
        reload(name)
    if PY34:
        importlib.reload(name)

def execute_tool(func, self, parameters, messages):
    if cfg.RELOAD_MODULES:
        # reload arcsdm.* modules
        reload_arcsdm_modules(messages)
        # update func ref to use reloaded code
        func.__code__ = getattr(sys.modules[func.__module__],  func.__name__).__code__
    if cfg.USE_PTVS_DEBUGGER:
        messages.AddMessage("Waiting for debugger..")
        try:
            from arcsdm.debug_ptvs import wait_for_debugger
            wait_for_debugger()
        except:
            messages.AddMessage("Failed to import debug_ptvs. Is ptvsd package installed?")
    try:
        # run the tool
        func(self, parameters, messages)
    except:
        tb = sys.exc_info()[2]
        errors = "Unhandled exception caught\n" + traceback.format_exc()
        arcpy.AddError(errors)         

def addToDisplay(layer, name, position):
    result = arcpy.MakeRasterLayer_management(layer, name)
    lyr = result.getOutput(0)
    product = arcpy.GetInstallInfo()['ProductName']
    if "Desktop" in product:
        mxd = arcpy.mapping.MapDocument("CURRENT")
        dataframe = arcpy.mapping.ListDataFrames(mxd)[0]
        arcpy.mapping.AddLayer(dataframe, lyr, position)
    elif "Pro" in product:
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        m = aprx.listMaps("Map")[0] 
        #m.addLayer(lyr, position)
        #AL removed row above and added next three rows
        m.addDataFromPath(layer)
        layer0 = m.listLayers()[0]
        layer0.name = name
