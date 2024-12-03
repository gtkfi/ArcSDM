import arcpy
import numpy as np
from typing import List

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
    
def read_and_stack_rasters(input_data: List[str], nodata_value = -99, nodata_handling = "convert_to_nan"):
    bands = []
    nodata_values = []

    for raster_file in input_data:
        # Describe the raster file to get its properties
        desc = arcpy.Describe(raster_file)
        
        # Check if the input data is a raster dataset or raster layer
        if desc.dataType == "RasterDataset" or desc.dataType == "RasterLayer":
            raster = arcpy.Raster(raster_file)
            
            # Extract raster profile information
            profile = {
                "nodata": raster.noDataValue,
                "extent": raster.extent,
                "width": raster.width,
                "height": raster.height,
                "bandCount": raster.bandCount
            }
            
            # Get band numbers and extract bands
            band_nums = [i for i in range(1, raster.bandCount + 1)]
            
            for band_num in band_nums:
                band = arcpy.ia.ExtractBand(raster, band_ids=band_num)
                # Convert raster bands to numpy float array 
                # Float array is used as np.nan is not supported in integer arrays
                band_data = arcpy.RasterToNumPyArray(band, nodata_to_value = nodata_value)
                band_data = band_data.astype(float)

                # Handle nodata values based on the specified method
                if nodata_handling == "convert_to_nan":
                    band_data[band_data == nodata_value] = np.nan
                elif nodata_handling == "unify":
                    band_data[nodata_value] = -9999
                elif nodata_handling == "raise_exception":
                    nodata_values.append(profile["nodata"])
                    if len(set(nodata_values)) > 1:
                        raise arcpy.ExecuteError("Input rasters have varying nodata values.")
                elif nodata_handling == "none":
                    pass
                
                bands.append(band_data)
        else:
            arcpy.AddError("Input data is not a raster.")
            raise arcpy.ExecuteError

    # Stack all bands into a single 3D array
    stacked_arrays = np.stack(bands, axis=0)
    return stacked_arrays