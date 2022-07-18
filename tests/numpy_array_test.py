from difflib import SequenceMatcher
import numpy as np
import pyotb

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

# Check to_rasterio
array, profile = inp.to_rasterio()

# Check data type and shape
assert array.dtype == profile['dtype'] == np.uint8
assert array.shape == (3, 100, 200)

# Check array statistics
assert array.min() == 35
assert array.max() == 255

# Check rasterio profile
image_crs = """PROJCS["RGF93 v1 / Lambert-93",
    GEOGCS["RGF93 v1",
        DATUM["Reseau_Geodesique_Francais_1993_v1",
            SPHEROID["GRS 1980",6378137,298.257222101,
                AUTHORITY["EPSG","7019"]],
            AUTHORITY["EPSG","6171"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4171"]],
    PROJECTION["Lambert_Conformal_Conic_2SP"],
    PARAMETER["latitude_of_origin",46.5],
    PARAMETER["central_meridian",3],
    PARAMETER["standard_parallel_1",49],
    PARAMETER["standard_parallel_2",44],
    PARAMETER["false_easting",700000],
    PARAMETER["false_northing",6600000],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH],
    AUTHORITY["EPSG","2154"]]"""

# Check string similarity ratio since direct string equality is not working in the docker image
matcher = SequenceMatcher(a=image_crs, b=profile['crs'])
pyotb.logger.debug("CRS string match ratio is %f", matcher.ratio())
assert matcher.ratio() > 0.99
assert profile['transform'] == (6.0, 0.0, 760056.0, 0.0, -6.0, 6946092.0)
