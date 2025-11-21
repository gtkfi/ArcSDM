import arcpy
from arcsdm.csi.core.calculation import calculation


def execute(self, parameters, messages):
    """Execute the CSI calculation"""
    try:
        # Get parameters
        labelled_path = parameters[0].valueAsText
        csv_nodata = float(parameters[1].value) if parameters[1].value else -9999.0
        label_field_names = parameters[2].valueAsText.split(';') if parameters[2].valueAsText else []
        coordinate_field_names = parameters[3].valueAsText.split(';') if parameters[3].valueAsText else []
        feature_field_names = parameters[4].valueAsText.split(';') if parameters[4].valueAsText else []
        evidence_type = parameters[5].valueAsText if parameters[5].value else None
        rasters_list = parameters[6].valueAsText.split(';') if parameters[6].valueAsText else []
        vector_list = parameters[7].valueAsText.split(';') if parameters[7].valueAsText else []
        out_labelled_pairwise_csv = parameters[8].valueAsText
        out_class_centroid = parameters[9].valueAsText if parameters[9].value else None
        out_evidence_table_csv = parameters[10].valueAsText if parameters[10].value else None
        out_raster_folder = parameters[11].valueAsText if parameters[11].value else None
        # Filter labeled points by field value
        selected_label_field = parameters[11].valueAsText if len(parameters) > 11 and parameters[11].value else None
        
        arcpy.AddMessage("Starting CSI Analysis...")
        arcpy.AddMessage("="*60)
        
        calculation(
            selected_label_field,
            labelled_path,
            label_field_names,
            coordinate_field_names,
            feature_field_names,
            rasters_list,
            evidence_type,
            csv_nodata,
            out_labelled_pairwise_csv,
            out_class_centroid,
            out_evidence_table_csv,
            out_raster_folder
        )

        arcpy.AddMessage("Completed CSI Analysis.")

    except arcpy.ExecuteError:
        arcpy.AddError(arcpy.GetMessages(2))
    except Exception as e:
        arcpy.AddError(f"Error in CSI calculation: {e}")
        import traceback
        arcpy.AddError(traceback.format_exc())