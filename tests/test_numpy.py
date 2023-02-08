import numpy as np
import pyotb
from tests_data import INPUT


def test_export():
    INPUT.export()
    array = INPUT.exports_dic[INPUT.output_image_key]["array"]
    assert isinstance(array, np.ndarray)
    assert array.dtype == "uint8"
    del INPUT.exports_dic["out"]


def test_output_export():
    INPUT["out"].export()
    assert INPUT["out"].output_image_key in INPUT["out"].exports_dic


def test_to_numpy():
    array = INPUT.to_numpy()
    assert array.dtype == np.uint8
    assert array.shape == INPUT.shape
    assert array.min() == 33
    assert array.max() == 255


def test_to_numpy_sliced():
    sliced = INPUT[:100, :200, :3]
    array = sliced.to_numpy()
    assert array.dtype == np.uint8
    assert array.shape == (100, 200, 3)


def test_convert_to_array():
    array = np.array(INPUT)
    assert isinstance(array, np.ndarray)
    assert INPUT.shape == array.shape


def test_pixel_coords_otb_equals_numpy():
    assert INPUT[19, 7] == list(INPUT.to_numpy()[19, 7])


def test_add_noise_array():
    white_noise = np.random.normal(0, 50, size=INPUT.shape)
    noisy_image = INPUT + white_noise
    assert isinstance(noisy_image, pyotb.core.App)
    assert noisy_image.shape == INPUT.shape


def test_to_rasterio():
    array, profile = INPUT.to_rasterio()
    assert array.dtype == profile["dtype"] == np.uint8
    assert array.shape == (4, 304, 251)
    assert profile["transform"] == (6.0, 0.0, 760056.0, 0.0, -6.0, 6946092.0)

    # CRS test requires GDAL python bindings
    try:
        from osgeo import osr

        crs = osr.SpatialReference()
        crs.ImportFromEPSG(2154)
        dest_crs = osr.SpatialReference()
        dest_crs.ImportFromWkt(profile["crs"])
        assert dest_crs.IsSame(crs)
    except ImportError:
        pass
