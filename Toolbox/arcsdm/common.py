# -*- coding: utf-8 -*-
import sys, os, traceback
import arcpy
import arcsdm.config as cfg
from arcsdm.exceptions import SDMError
from arcsdm.debug_ptvs import wait_for_debugger

PY2 = sys.version_info[0] == 2
PY34 = sys.version_info[0:2] >= (3, 4)

if PY2:
    from imp import reload;
if PY34:
    import importlib

def reload_module(name, messages):
    if not cfg.RELOAD_MODULES: return
    try:
        if PY2:
            reload(name)
        if PY34:
            importlib.reload(name)
        messages.AddMessage("Reloaded module {}".format(name))
    except Exception as e:
        messages.AddMessage("Failed to reload module %s. Reason:%s" %(name, e.message))

def execute_tool(func, self, parameters, messages):
    if cfg.USE_PTVS_DEBUGGER:
        messages.AddMessage("Waiting for debugger..")
        wait_for_debugger()
    try:
        func(self, parameters, messages)
    except arcpy.ExecuteError as e:
        msgs = arcpy.GetMessages(2)  
        arcpy.AddError(msgs) 
        # get the traceback object
        tb = sys.exc_info()[2]
        # tbinfo contains the line number that the code failed on and the code from that line
        tbinfo = traceback.format_tb(tb)[0]
        # concatenate information together concerning the error into a message string
        trace = "Traceback\n" + tbinfo + "\nError Info:\n" + str(sys.exc_info()[1])
        print(trace)
        print(msgs)
    except SDMError as e:
        arcpy.AddError(e.value)
        # get the traceback object
        tb = sys.exc_info()[2]
        # tbinfo contains the line number that the code failed on and the code from that line
        tbinfo = traceback.format_tb(tb)[0]
        # concatenate information together concerning the error into a message string
        trace = "Traceback\n" + tbinfo
        arcpy.AddError('jiihaa2')
        arcpy.AddError(trace)
        print(trace)
        print(e.value)
    except:
        # get the traceback object
        arccpy.AddError('Unexcpected exception caught')
        tb = sys.exc_info()[2]
        # tbinfo contains the line number that the code failed on and the code from that line
        tbinfo = traceback.format_tb(tb)[0]
        # concatenate information together concerning the error into a message string
        trace = "Traceback\n" + tbinfo + "\nError Info:\n" + str(sys.exc_info()[1])
        msgs = "ArcPy ERRORS:\n" + arcpy.GetMessages(2) + "\n"
        arcpy.AddError(trace)
        arcpy.AddError(msgs)
        print(trace)
        print(msgs)

