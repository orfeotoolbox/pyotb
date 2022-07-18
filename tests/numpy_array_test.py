import numpy as np
import pyotb

filepath = 'image.tif'

# Test to_numpy array
inp = pyotb.Input(filepath)
array = inp.to_numpy()
assert array.dtype == np.float64
assert array.shape == (100, 100, 4)

# Test to_numpy array with slicer
inp = pyotb.Input(filepath)[:50, :60, :3]
array = inp.to_numpy()
assert array.dtype == np.float64
assert array.shape == (50, 60, 3)

# Test conversion to numpy array
array = np.array(inp)
assert isinstance(array, np.ndarray)
assert inp.shape == array.shape

# Test image + noise README example
white_noise = np.random.normal(0, 50, size=inp.shape)
noisy_image = inp + white_noise
assert isinstance(noisy_image, pyotb.App)
assert noisy_image.shape == inp.shape

# Check to_rasterio
array, profile = inp.to_rasterio()

# Check data type and shape
assert array.dtype == profile['dtype'] == np.float64
assert array.shape == (3, 50, 60)

# Check array statistics
assert array.min() == 136.0
assert array.max() == 591.0

# Check rasterio profile
assert profile['transform'] == (1.0, 0.0, 1000.0, 0.0, 1.0, 1000.0)
