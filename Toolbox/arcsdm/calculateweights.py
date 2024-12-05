# -*- coding: utf-8 -*-

import arcpy
import importlib
import math
import os
import sys
import traceback

import arcsdm.sdmvalues
import arcsdm.wofe_common

from arcsdm.wofe_common import check_input_data, get_evidence_values_at_training_points


ASCENDING = "Ascending"
DESCENDING = "Descending"
CATEGORICAL = "Categorical"
UNIQUE = "Unique"


def extract_layer_from_raster_band(evidence_layer, evidence_descr):
    if evidence_descr.dataType == "RasterBand":
    # Try to change RasterBand to RasterDataset
        evidence1 = os.path.split(evidence_layer)
        evidence2 = os.path.split(evidence1[0])
        if (evidence1[1] == evidence2[1]) or (evidence1[1][:4] == "Band"):
            new_evidence_layer = evidence1[0]
            new_evidence_descr = arcpy.Describe(evidence_layer)
            arcpy.AddMessage("Evidence Layer is now " + new_evidence_layer + " and its data type is " + new_evidence_descr.dataType)
            return new_evidence_layer, new_evidence_descr
        else:
            arcpy.ExecuteError("ERROR: Data Type of Evidence Layer cannot be RasterBand, use Raster Dataset.")
    else:
        return evidence_layer, evidence_descr


def calculate_unique_weights(evidence_raster, training_site_raster):
    # TODO: select unique classes
    pass


def calculate_cumulative_weights():
    pass


def Calculate(self, parameters, messages):
    # TODO: make relevant checks to input parameters
    # TODO: convert evidence feature to raster
    # TODO: make sure mask is used in all steps that it needs to be used in
    # TODO: calculate weights based on weights type

    # TODO: Remove this after testing is done!
    # Make sure imported modules are refreshed if the toolbox is refreshed.
    importlib.reload(arcsdm.wofe_common)
    importlib.reload(arcsdm.sdmvalues)

    arcpy.AddMessage("Starting weight calculation")
    arcpy.AddMessage("------------------------------")
    try:
        arcpy.env.overwriteOutput = True
        arcpy.AddMessage("overwriteOutput set to True")

        # Input parameters are as follows:
        # 0: EvidenceRasterLayer
        # 1: EvidenceRasterCodefield (what is this?)
        # 2: TrainingPoints
        # 3: Type
        # 4: OutputWeightsTable
        # 5: ConfidenceLevelOfStudentizedContrast
        # 6: UnitAreaKm2
        # 7: MissingDataValue
        # 8: Success

        evidence_raster = parameters[0].valueAsText
        code_name =  parameters[1].valueAsText
        training_sites_feature = parameters[2].valueAsText
        selected_weight_type =  parameters[3].valueAsText
        output_weights_table = parameters[4].valueAsText
        studentized_contrast_threshold = parameters[5].value
        unit_area_sq_km = parameters[6].value
        nodata_value = parameters[7].value

        # Test data type of Evidence Layer
        evidence_descr = arcpy.Describe(evidence_raster)

        evidence_raster, evidence_descr = extract_layer_from_raster_band(evidence_raster, evidence_descr)

        check_input_data([evidence_raster], training_sites_feature)
        
        # If using non gdb database, lets add .dbf
        # If using GDB database, remove numbers and underscore from the beginning of the Weights table name (else block)
        workspace_descr = arcpy.Describe(arcpy.env.workspace)
        if workspace_descr.workspaceType == "FileSystem":
            if not(output_weights_table.endswith(".dbf")):
                output_weights_table += ".dbf"
        else:
            wtsbase = os.path.basename(output_weights_table)
            while len(wtsbase) > 0 and (wtsbase[:1] <= "9" or wtsbase[:1] == "_"):
                wtsbase = wtsbase[1:]
            output_weights_table = os.path.dirname(output_weights_table) + "\\" + wtsbase

        mask = arcpy.env.mask
        if mask:
            if not arcpy.Exists(mask):
                raise arcpy.ExecuteError("Mask doesn't exist! Set Mask under Analysis/Environments.")
            
            evidence_raster = arcpy.sa.ExtractByMask(evidence_raster, mask)

        arcsdm.sdmvalues.appendSDMValues(unit_area_sq_km, training_sites_feature)
        arcpy.AddMessage("=" * 10 + " Calculate weights " + "=" * 10)

        arcpy.AddMessage ("%-20s %s (%s)" % ("Creating table: ", output_weights_table, selected_weight_type))

        # Extract points from training sites feature layer to a raster
        # A new field named RASTERVALU is added to the output to store the extracted values
        assert isinstance(evidence_raster, object)

        values_at_training_points = get_evidence_values_at_training_points(evidence_raster, training_sites_feature)

        # TODO: in both categorical cases:
        # TODO: get both the evidence and the training data as raster
        # TODO: in case of training data: set default value as 1 & fill value as 0
        # TODO: get the unique values in the evidence raster (should already be classified, so give a warning if there are a lot of classes)
        # TODO: -> those are the classes
        # TODO: for each class, get the weights
        # TODO: (probably faster to read the data into a numpy array instead of trying to work with cursors)
        # TODO: (note: probably easier to work with )

        if selected_weight_type in [UNIQUE, CATEGORICAL]:
            calculate_unique_weights(evidence_raster, values_at_training_points)
        elif selected_weight_type == ASCENDING:
            calculate_cumulative_weights()
        elif selected_weight_type == DESCENDING:
            calculate_cumulative_weights()

    except arcpy.ExecuteError:
        arcpy.AddError(arcpy.GetMessages(2))
    except Exception:
        e = sys.exc_info()[1]
        print(e.args[0])
        arcpy.AddError(e.args[0])
