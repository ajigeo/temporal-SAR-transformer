from osgeo import gdal, osr
#%%
def read_image(filename):
    image = gdal.Open(filename)
    image_transform = image.GetGeoTransform()
    image_projection = image.GetProjectionRef()
    image_array = image.ReadAsArray()
    #image_shape = image_array.shape()
    return [image_array,image_projection,image_transform]

#%% Assigning projection to the matrix and saving as tif
#https://gis.stackexchange.com/questions/37238/writing-numpy-array-to-raster-file
#from osgeo import
def plot_output(input_reshaped_array, transformation, output_location='E:/final_plot.tif'): 
    rows,cols = input_reshaped_array.shape
    output_raster = gdal.GetDriverByName('GTiff').Create(output_location,cols, rows, 1 ,gdal.GDT_Float32)  # Open the file
    output_raster.SetGeoTransform(transformation)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    output_raster.SetProjection( srs.ExportToWkt() )
    output_raster.GetRasterBand(1).WriteArray(input_reshaped_array)
    output_raster.FlushCache()