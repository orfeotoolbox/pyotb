import numpy as np
import pyotb

filepath = 'Data/Input/QB_MUL_ROI_1000_100.tif'
inp = pyotb.Input(filepath)

array, profile = inp.to_rasterio()

# Check data type and shape
assert array.dtype == np.float64
assert array.shape == (4, 100, 100)

# Check array statistics
assert array.min() == 126.0
assert array.max() == 1973.0

# Check rasterio profile
assert profile['transform'] == (1.0, 0.0, 1000.0, 0.0, 1.0, 1000.0)
