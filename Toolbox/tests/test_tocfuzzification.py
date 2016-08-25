# -*- coding: utf-8 -*-
import unittest
import platform
import os
import arcpy

class TestTOCFuzzification(unittest.TestCase):
    
    ARCSDM_TOOLBOX = "../ArcSDMPythonToolbox.pyt"
    GDBDIR = "../../work/"
    GDB = "database.gdb"
    MASK = os.path.join(GDBDIR, GDB + "/study_area") 
    
    @classmethod
    def setUpClass(cls):
        print("Using Python version %s" % platform.python_version())
        print("Setting up env:")
        print("\tARCSDM_TOOLBOX=%s" % cls.ARCSDM_TOOLBOX)
        arcpy.env.mask =  cls.MASK
        print("\tarcpy.env.mask=%s" % arcpy.env.mask)
        arcpy.env.overwriteOutput = True
        print("\tarcpy.env.overwriteOutput=%s" % arcpy.env.overwriteOutput)
        print("Checking out Spatial extension..")
        arcpy.CheckOutExtension("spatial")
        print("Importing ArcSDM toolbox..")
        arcpy.ImportToolbox(cls.ARCSDM_TOOLBOX)

    @classmethod
    def tearDownClass(cls):
        arcpy.CheckInExtension("spatial")

    def test1(self):
        pass

if __name__ == '__main__':
    unittest.main()