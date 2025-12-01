import arcpy
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

def load_labeled_data(
    labelled_path: str,
    label_field_names: Optional[List[str]],
    feature_field_names: Optional[List[str]],
    coordinate_field_names: Optional[List[str]] = None
) -> Tuple[Optional[pd.DataFrame], Optional[List[str]], bool]:

    """Load labeled data from feature class or table.
    
    Returns:
        tuple of (DataFrame with labeled data, list of feature field names, bool indicating if geometry is present)
    """
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

        # For non-geometry tables, use provided coordinate_field_names
        if has_geometry is False:
            if coordinate_field_names:
                for col in coordinate_field_names:
                    if col not in fields:
                        fields.append(col)
        else:
            arcpy.AddWarning(f"SHAPE FOUND - using geometry for coordinates")
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