import sys
import arcpy

import arcsdm.sitereduction
import arcsdm.logisticregression
import arcsdm.calculateweights
import arcsdm.categoricalreclass
import arcsdm.categoricalmembership
import arcsdm.tocfuzzification
import arcsdm.logisticregression
import arcsdm.calculateresponse
import arcsdm.symbolize
import arcsdm.roctool
import arcsdm.acterbergchengci
import arcsdm.rescale_raster;
from arcsdm.areafrequency import Execute

from arcsdm.common import execute_tool


import importlib
from imp import reload;


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        
        self.label = "Experimental SDM toolbox"
        self.alias = "experimentaltools" 

        # List of tool classes associated with this toolbox
        self.tools = [rastersom]
        
        
