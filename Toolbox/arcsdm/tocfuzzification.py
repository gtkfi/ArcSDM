# -*- coding: utf-8 -*-
import sys, string, os, math, traceback, math
import arcpy
from arcpy.sa import Float, Divide

def Calculate(self, parameters, messages):
    try:
         messages.addMessage("Starting toc fuzzification calculation");
    except Exception as Msg:
        # get the traceback object
        tb = sys.exc_info()[2]
        # tbinfo contains the line number that the code failed on and the code from that line
        tbinfo = traceback.format_tb(tb)[0]
        # concatenate information together concerning the error into a message string
        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
            str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
        messages.addErrorMessage(pymsg); #msgs)
        # print messages for use in Python/PythonWin
        print (pymsg)
        raise
