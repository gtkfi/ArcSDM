import arcpy
import numpy as np

def input_to_numpy_array(data):
        desc = arcpy.Describe(data)

        if desc.datasetType == 'RasterDataset' or desc == 'RasterLayer':
            input_as_array = arcpy.RasterToNumPyArray(data)
            input_as_array = np.array([list(row) for row in input_as_array])
        elif desc.datasetType == 'FeatureClass' or desc == 'FeatureLayer':
            input_as_array = arcpy.da.FeatureClassToNumPyArray(data, [field.name for field in arcpy.ListFields(x) if field.type != 'OID'])
            input_as_array = np.array(input_as_array.tolist())
        else:
            arcpy.AddError("Input data is not a raster or a feature class.")
            raise arcpy.ExecuteError
            
        return input_as_array