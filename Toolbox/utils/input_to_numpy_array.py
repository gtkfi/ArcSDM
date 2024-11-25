import arcpy
import numpy as np

def input_to_numpy_arrays(input_data):

        datasets = [] 
        for data in input_data: 
            desc = arcpy.Describe(data) 
            if desc.dataType == "RasterDataset" or desc.dataType == "RasterLayer": 
                if desc.bandCount == 1: 
                    dataset = arcpy.RasterToNumPyArray(data) 
                else: 
                    out_bands_raster = [arcpy.ia.ExtractBand(data, band_ids=i)for i in range(desc.bandCount)] 
                    dataset = [arcpy.RasterToNumPyArray(band) for band in out_bands_raster] 

            elif desc.datasetType == 'FeatureClass' or desc == 'FeatureLayer':
                dataset = arcpy.da.FeatureClassToNumPyArray(data)
            else:
                arcpy.AddError("Input data is not a raster or a feature class.")
                raise arcpy.ExecuteError

            datasets.append(dataset)

        return datasets