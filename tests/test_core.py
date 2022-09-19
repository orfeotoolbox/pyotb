import os
import pyotb
from ast import literal_eval
from pathlib import Path


FILEPATH = os.environ["TEST_INPUT_IMAGE"]
INPUT = pyotb.Input(FILEPATH)


# Basic tests
def test_dtype():
    assert INPUT.dtype == "uint8"


def test_shape():
    assert INPUT.shape == (304, 251, 4)


def test_slicer_shape():
    extract = INPUT[:50, :60, :3]
    assert extract.shape == (50, 60, 3)


def test_slicer_preserve_dtype():
    extract = INPUT[:50, :60, :3]
    assert extract.dtype == "uint8"


# More complex tests
def test_operation():
    op = INPUT / 255 * 128
    assert op.exp == "((im1b1 / 255) * 128);((im1b2 / 255) * 128);((im1b3 / 255) * 128);((im1b4 / 255) * 128)"


def test_sum_bands():
    # Sum of bands
    summed = sum(INPUT[:, :, b] for b in range(INPUT.shape[-1]))
    assert summed.exp == "((((0 + im1b1) + im1b2) + im1b3) + im1b4)"


def test_binary_mask_where():
    # Create binary mask based on several possible values
    values = [1, 2, 3, 4]
    res = pyotb.where(pyotb.any(INPUT[:, :, 0] == value for value in values), 255, 0)
    assert res.exp == "(((((im1b1 == 1) || (im1b1 == 2)) || (im1b1 == 3)) || (im1b1 == 4)) ? 255 : 0)"


# Apps
def test_app_readimageinfo():
    info = pyotb.ReadImageInfo(INPUT, quiet=True)
    assert info.sizex == 251
    assert info.sizey == 304
    assert info["numberbands"] == info.numberbands == 4


def test_app_computeimagestats():
    stats = pyotb.ComputeImagesStatistics([INPUT], quiet=True)
    assert stats["out.min"] == "[33, 64, 91, 47]"


def test_app_computeimagestats_sliced():
    slicer_stats = pyotb.ComputeImagesStatistics(il=[INPUT[:10, :10, 0]], quiet=True)
    assert slicer_stats["out.min"] == "[180]"


# NDVI
def test_ndvi_comparison():
    ndvi_bandmath = (INPUT[:, :, -1] - INPUT[:, :, [0]]) / (INPUT[:, :, -1] + INPUT[:, :, 0])
    ndvi_indices = pyotb.RadiometricIndices(
        {"in": INPUT, "list": "Vegetation:NDVI", "channels.red": 1, "channels.nir": 4}
    )
    assert ndvi_bandmath.exp == "((im1b4 - im1b1) / (im1b4 + im1b1))"

    ndvi_bandmath.write("/tmp/ndvi_bandmath.tif", pixel_type="float")
    assert Path("/tmp/ndvi_bandmath.tif").exists()
    ndvi_indices.write("/tmp/ndvi_indices.tif", pixel_type="float")
    assert Path("/tmp/ndvi_indices.tif").exists()

    compared = pyotb.CompareImages({"ref.in": ndvi_indices, "meas.in": "/tmp/ndvi_bandmath.tif"})
    assert compared.count == 0
    assert compared.mse == 0

    thresholded_indices = pyotb.where(ndvi_indices >= 0.3, 1, 0)
    assert thresholded_indices.exp == "((im1b1 >= 0.3) ? 1 : 0)"

    thresholded_bandmath = pyotb.where(ndvi_bandmath >= 0.3, 1, 0)
    assert thresholded_bandmath.exp == "((((im1b4 - im1b1) / (im1b4 + im1b1)) >= 0.3) ? 1 : 0)"
