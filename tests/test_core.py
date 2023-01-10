import os
import pyotb
from ast import literal_eval
from pathlib import Path

import pytest


FILEPATH = os.environ["TEST_INPUT_IMAGE"]
INPUT = pyotb.Input(FILEPATH)


# Input settings
def test_parameters():
    assert (INPUT.parameters["sizex"], INPUT.parameters["sizey"]) == (251, 304)


# Catch errors before exec
def test_missing_input_file():
    with pytest.raises(FileNotFoundError):
        pyotb.Input("missing_file.tif")


def test_wrong_key():
    with pytest.raises(KeyError):
        pyotb.BandMath(INPUT, expression="im1b1")


# OTBObject's properties
def test_name():
    assert INPUT.name == "Input from tests/image.tif"
    INPUT.name = "Test input"
    assert INPUT.name == "Test input"


def test_key_input():
    assert INPUT.key_input == INPUT.key_input_image == "in"


def test_key_output():
    assert INPUT.key_output_image == "out"


def test_dtype():
    assert INPUT.dtype == "uint8"


def test_shape():
    assert INPUT.shape == (304, 251, 4)


def test_transform():
    assert INPUT.transform == (6.0, 0.0, 760056.0, 0.0, -6.0, 6946092.0)


def test_slicer_shape():
    extract = INPUT[:50, :60, :3]
    assert extract.shape == (50, 60, 3)
    assert extract.parameters["cl"] == ("Channel1", "Channel2", "Channel3")


def test_slicer_preserve_dtype():
    extract = INPUT[:50, :60, :3]
    assert extract.dtype == "uint8"


# Arithmetic
def test_operation():
    op = INPUT / 255 * 128
    assert op.exp == "((im1b1 / 255) * 128);((im1b2 / 255) * 128);((im1b3 / 255) * 128);((im1b4 / 255) * 128)"
    assert op.dtype == "float32"


def test_func_abs_expression():
    assert abs(INPUT).exp == "(abs(im1b1));(abs(im1b2));(abs(im1b3));(abs(im1b4))"


def test_sum_bands():
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
    assert (info.sizex, info.sizey) == (251, 304)
    assert info["numberbands"] == info.numberbands == 4


def test_app_computeimagestats():
    stats = pyotb.ComputeImagesStatistics([INPUT], quiet=True)
    assert stats["out.min"] == "[33, 64, 91, 47]"


def test_app_computeimagestats_sliced():
    slicer_stats = pyotb.ComputeImagesStatistics(il=[INPUT[:10, :10, 0]], quiet=True)
    assert slicer_stats["out.min"] == "[180]"


def test_read_values_at_coords():
    assert INPUT[0, 0, 0] == 180
    assert INPUT[10, 20, :] == [196, 188, 172, 255]


# XY => RowCol
def test_xy_to_rowcol():
    assert INPUT.xy_to_rowcol(760100, 6946210) == (19, 7)


# Create dir before write
def test_write():
    INPUT.write("/tmp/missing_dir/test_write.tif")
    assert INPUT.out.filepath.exists()


# BandMath NDVI == RadiometricIndices NDVI ?
@pytest.mark.xfail(reason="Regression in OTB 8.2, waiting for Rémi's patch to be merged.")
def test_ndvi_comparison():
    ndvi_bandmath = (INPUT[:, :, -1] - INPUT[:, :, [0]]) / (INPUT[:, :, -1] + INPUT[:, :, 0])
    ndvi_indices = pyotb.RadiometricIndices(INPUT, {"list": "Vegetation:NDVI", "channels.red": 1, "channels.nir": 4})
    assert ndvi_bandmath.exp == "((im1b4 - im1b1) / (im1b4 + im1b1))"

    ndvi_bandmath.write("/tmp/ndvi_bandmath.tif", pixel_type="float")
    assert ndvi_bandmath.out.filepath.exists()
    ndvi_indices.write("/tmp/ndvi_indices.tif", pixel_type="float")
    assert ndvi_indices.out.filepath.exists()

    compared = pyotb.CompareImages({"ref.in": ndvi_indices, "meas.in": "/tmp/ndvi_bandmath.tif"})
    assert (compared.count, compared.mse) == (0, 0)

    thresholded_indices = pyotb.where(ndvi_indices >= 0.3, 1, 0)
    assert thresholded_indices.exp == "((im1b1 >= 0.3) ? 1 : 0)"

    thresholded_bandmath = pyotb.where(ndvi_bandmath >= 0.3, 1, 0)
    assert thresholded_bandmath.exp == "((((im1b4 - im1b1) / (im1b4 + im1b1)) >= 0.3) ? 1 : 0)"
