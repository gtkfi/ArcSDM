import arcpy
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from arcsdm.csi.helpers.detect_xy_columns import detect_xy_columns

def load_labeled_data(
    labelled_path: str,
    label_field_names: Optional[List[str]],
    feature_field_names: Optional[List[str]]
) -> Tuple[Optional[pd.DataFrame], Optional[List[str]], bool]:

    """Load labeled data from feature class or table"""
    try:

        if not feature_field_names:
            field_objects = arcpy.ListFields(labelled_path)
            all_numeric_fields = [
                f.name for f in field_objects
                if f.type in ['Double', 'Float', 'Integer', 'SmallInteger']
            ]
            exclude_fields = ['OBJECTID', 'OID', 'FID', 'Shape_Length', 'Shape_Area', 'SHAPE']
            all_numeric_fields = [f for f in all_numeric_fields if f not in exclude_fields]

            # Add any remaining fields if needed
            remaining_fields = [f for f in all_numeric_fields if f not in feature_field_names]
            feature_field_names.extend(remaining_fields)

            arcpy.AddMessage(f"Auto-detected feature fields: {feature_field_names}")

        fields = feature_field_names.copy()
        if label_field_names:
            fields.extend([f for f in label_field_names if f not in fields])

        desc = arcpy.Describe(labelled_path)
        has_geometry = hasattr(desc, 'shapeType')

        # For non-geometry tables, try to detect coordinate fields
        if not has_geometry:
            sample_fields = [f.name for f in arcpy.ListFields(labelled_path)]
            temp_data = []
            with arcpy.da.SearchCursor(labelled_path, sample_fields[:10]) as cursor:
                for i, row in enumerate(cursor):
                    if i >= 5:
                        break
                    temp_data.append(row)

            if temp_data:
                temp_df = pd.DataFrame(temp_data, columns=sample_fields[:10])
                xcol, ycol = detect_xy_columns(temp_df)
                if xcol and ycol and xcol not in fields:
                    fields.append(xcol)
                if xcol and ycol and ycol not in fields:
                    fields.append(ycol)
        else:
            fields.append('SHAPE@XY')

        data = []
        with arcpy.da.SearchCursor(labelled_path, fields) as cursor:
            for row in cursor:
                data.append(row)

        df = pd.DataFrame(data, columns=fields)
        df.replace({None: np.nan}, inplace=True)

        arcpy.AddMessage(f"Loaded {len(df)} labeled points with {len(feature_field_names)} features")
        return df, feature_field_names, has_geometry

    except Exception as e:
        arcpy.AddError(f"Error loading labeled data: {e}")
        return None, None, False


def load_raster_data(
    rasters_list: List[str]
) -> List[str]:
    """Load raster data"""
    try:
        raster_data = []
        for raster_path in rasters_list:
            if arcpy.Exists(raster_path):
                raster_data.append(raster_path)
            else:
                arcpy.AddWarning(f"Raster does not exist: {raster_path}")

        arcpy.AddMessage(f"Loaded {len(raster_data)} raster layers")
        return raster_data

    except Exception as e:
        arcpy.AddError(f"Error loading raster data: {e}")
        return []