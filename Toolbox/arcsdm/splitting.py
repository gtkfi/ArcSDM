"""
    ArcSDM 6 ToolBox for ArcGIS Pro

    Splitting Tool for Training Sites Reduction
"""

import sys
import traceback
import arcpy
import random
from utils.site_reduction_functions import create_output_layer, get_identifier

def SplitSites(self, parameters, messages):
    arcpy.AddMessage("Starting Splitting Tool")
    arcpy.AddMessage("------------------------------")

    try:
        # Read parameters
        input_features = parameters[0].valueAsText
        random_percentage = parameters[1].value
        output_layer = parameters[2].valueAsText or "reduced_sites_train"
        inverse_output_layer = parameters[3].valueAsText or "reduced_sites_test"

        # Validate random percentage
        if not (0 < random_percentage < 100):
            arcpy.AddError("Random percentage must be between 0 and 100.")
            return

        # Get identifier field (OBJECTID or FID)
        identifier = get_identifier(input_features)

        # Collect all features
        features = []
        with arcpy.da.SearchCursor(input_features, [identifier, "SHAPE@"]) as cursor:
            for row in cursor:
                features.append((row[0], row[1]))

        arcpy.AddMessage(f"Total features: {len(features)}")

        # Randomly select features for training
        random.shuffle(features)
        split_index = int(len(features) * (random_percentage / 100))
        training_features = features[:split_index]
        testing_features = features[split_index:]

        arcpy.AddMessage(f"Training features: {len(training_features)}")
        arcpy.AddMessage(f"Testing features: {len(testing_features)}")

        # Create inverse (testing) layer if specified
        if inverse_output_layer:
            create_output_layer([f[0] for f in testing_features], input_features, inverse_output_layer, identifier)
        else:
            create_output_layer([f[0] for f in training_features], input_features, output_layer, identifier)


        arcpy.AddMessage("Splitting completed successfully.")

    except arcpy.ExecuteError:
        e = sys.exc_info()[1]
        arcpy.AddError(f"Splitting Tool caught arcpy.ExecuteError: {e.args[0]}")

    except Exception as e:
        tb_info = traceback.format_tb(sys.exc_info()[2])[0]
        error_message = f"PYTHON ERRORS:\nTraceback Info:\n{tb_info}\nError Info:\n{type(e).__name__}: {e}\n"
        messages.AddError(error_message)
