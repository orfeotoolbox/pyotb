import numpy as np
import pyotb
from osgeo import osr

filepath = 'image.tif'

# Test to_numpy array
inp = pyotb.Input(filepath)
array = inp.to_numpy()
assert array.dtype == np.uint8
assert array.shape == (304, 251, 4)

# Test to_numpy array with slicer
inp = pyotb.Input(filepath)[:100, :200, :3]
array = inp.to_numpy()
assert array.dtype == np.uint8
assert array.shape == (100, 200, 3)

# Test conversion to numpy array
array = np.array(inp)
assert isinstance(array, np.ndarray)
assert inp.shape == array.shape

# Test image + noise README example
white_noise = np.random.normal(0, 50, size=inp.shape)
noisy_image = inp + white_noise
assert isinstance(noisy_image, pyotb.App)
assert noisy_image.shape == inp.shape

# Test to_rasterio
array, profile = inp.to_rasterio()
# Data type and shape
assert array.dtype == profile['dtype'] == np.uint8
assert array.shape == (3, 100, 200)
# Array statistics
assert array.min() == 35
assert array.max() == 255
# Spatial reference
assert profile['transform'] == (6.0, 0.0, 760056.0, 0.0, -6.0, 6946092.0)
crs = osr.SpatialReference()
crs.ImportFromEPSG(2154)
dest_crs = osr.SpatialReference()
dest_crs.ImportFromWkt(profile['crs'])
wkt1, wkt2 = crs.ExportToWkt(), dest_crs.ExportToWkt()
print(f"{wkt1=}, {wkt2=}")
assert wkt1 == wkt2
