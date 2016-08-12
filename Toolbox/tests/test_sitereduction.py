import unittest
import platform
import os
import arcpy


class TestSiteReduction(unittest.TestCase):
    ARCSDM_TOOLBOX = "../ArcSDMPythonToolbox.pyt"
    GDBDIR = "../../work/"
    GDB = "database.gdb"
    TRAINING_SITE_FILEPATH = os.path.join(GDBDIR, GDB + "/gold_deposits") 
    TRAINING_SITE_LYR_STR = "traininig_site_lyr"	
    TRAINING_SITE_LYR = arcpy.MakeFeatureLayer_management(TRAINING_SITE_FILEPATH, TRAINING_SITE_LYR_STR)	
    MASK = os.path.join(GDBDIR, GDB + "/study_area") 
    @classmethod
    def setUpClass(cls):
        print("Using Python version %s" % platform.python_version())
        print("Setting up env:")
        print("\tARCSDM_TOOLBOX=%s" % cls.ARCSDM_TOOLBOX)
        print("\tTRAINING_SITE_FILEPATH=%s" % cls.TRAINING_SITE_FILEPATH)
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
    
    def test_thin_sel_0(self):
        arcpy.SiteReductionTool_ArcSDM(self.TRAINING_SITE_LYR, True, "0", "", "")
        desc=arcpy.Describe(self.TRAINING_SITE_LYR_STR)
        sel_count = len(desc.fidSet.split(";"))
        self.assertEqual(sel_count, 60)

    def test_thin_sel_100(self):
        arcpy.SiteReductionTool_ArcSDM(self.TRAINING_SITE_LYR, True, "100", "", "")
        desc=arcpy.Describe(self.TRAINING_SITE_LYR_STR)
        sel_count = len(desc.fidSet.split(";"))
        self.assertEqual(sel_count, 44)
    
    def test_thin_sel_1000(self):
        arcpy.SiteReductionTool_ArcSDM(self.TRAINING_SITE_LYR, True, "1000", "", "")
        desc=arcpy.Describe(self.TRAINING_SITE_LYR_STR)
        sel_count = len(desc.fidSet.split(";"))
        self.assertEqual(sel_count, 26)
    
    def test_thin__sel_5000(self):
        arcpy.SiteReductionTool_ArcSDM(self.TRAINING_SITE_LYR, True, "5000", "", "")
        desc=arcpy.Describe(self.TRAINING_SITE_LYR_STR)
        sel_count = len(desc.fidSet.split(";"))
        self.assertEqual(sel_count, 14)
    
    def test_random_sel_1(self):
        arcpy.SiteReductionTool_ArcSDM(self.TRAINING_SITE_LYR, False, "", True, "1")
        desc=arcpy.Describe(self.TRAINING_SITE_LYR_STR)
        sel_count = len(desc.fidSet.split(";"))
        #THROWS!
    
    def test_random_sel_2(self):
        arcpy.SiteReductionTool_ArcSDM(self.TRAINING_SITE_LYR, False, "", True, "15")
        desc=arcpy.Describe(self.TRAINING_SITE_LYR_STR)
        sel_count = len(desc.fidSet.split(";"))
        self.assertEqual(sel_count, 9)
    
    def test_random_sel_25(self):
        arcpy.SiteReductionTool_ArcSDM(self.TRAINING_SITE_LYR, False, "", True, "25")
        desc=arcpy.Describe(self.TRAINING_SITE_LYR_STR)
        sel_count = len(desc.fidSet.split(";"))
        self.assertEqual(sel_count, 15)
    
    def test_random_sel_80(self):
        arcpy.SiteReductionTool_ArcSDM(self.TRAINING_SITE_LYR, False, "", True, "80")
        desc=arcpy.Describe(self.TRAINING_SITE_LYR_STR)
        sel_count = len(desc.fidSet.split(";"))
        self.assertEqual(sel_count, 48)

    def test_random_sel_100(self):
        arcpy.SiteReductionTool_ArcSDM(self.TRAINING_SITE_LYR, False, "", True, "100")
        desc=arcpy.Describe(self.TRAINING_SITE_LYR_STR)
        sel_count = len(desc.fidSet.split(";"))
        #THROWS
    
    def test_thin_random_sel_5000_50(self):
        arcpy.SiteReductionTool_ArcSDM(self.TRAINING_SITE_LYR, True, "5000", True, "50")
        desc=arcpy.Describe(self.TRAINING_SITE_LYR_STR)
        sel_count = len(desc.fidSet.split(";"))
        self.assertEqual(sel_count, 7)

    def test_thin_random_sel_1000_80(self):
        arcpy.SiteReductionTool_ArcSDM(self.TRAINING_SITE_LYR, True, "1000", True, "80")
        desc=arcpy.Describe(self.TRAINING_SITE_LYR_STR)
        sel_count = len(desc.fidSet.split(";"))
        self.assertEqual(sel_count, 20)

if __name__ == '__main__':
    unittest.main()
