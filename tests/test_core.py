import pytest

import pyotb
from tests_data import *


# Input settings
def test_parameters():
    assert INPUT.parameters
    assert INPUT.parameters["in"] == FILEPATH
    assert (INPUT.parameters["sizex"], INPUT.parameters["sizey"]) == (251, 304)


def test_input_vsi():
    # Simple remote file
    info = pyotb.ReadImageInfo("https://fake.com/image.tif", frozen=True)
    assert info.app.GetParameterValue("in") == "/vsicurl/https://fake.com/image.tif"
    assert info.parameters["in"] == "https://fake.com/image.tif"
    # Compressed remote file
    info = pyotb.ReadImageInfo("https://fake.com/image.tif.zip", frozen=True)
    assert info.app.GetParameterValue("in") == "/vsizip//vsicurl/https://fake.com/image.tif.zip"
    assert info.parameters["in"] == "https://fake.com/image.tif.zip"
    # Piped curl --> zip --> tiff
    ziped_tif_urls = (
        "https://github.com/OSGeo/gdal/raw/master"
        "/autotest/gcore/data/byte.tif.zip",  # without /vsi
        "/vsizip/vsicurl/https://github.com/OSGeo/gdal/raw/master"
        "/autotest/gcore/data/byte.tif.zip",  # with /vsi
    )
    for ziped_tif_url in ziped_tif_urls:
        info = pyotb.ReadImageInfo(ziped_tif_url)
        assert info["sizex"] == 20


def test_input_vsi_from_user():
    # Ensure old way is still working: ExtractROI will raise RuntimeError if a path is malformed
    pyotb.Input("/vsicurl/" + FILEPATH)


def test_wrong_key():
    with pytest.raises(KeyError):
        pyotb.BandMath(INPUT, expression="im1b1")


# OTBObject properties
def test_name():
    app = pyotb.App("BandMath", [INPUT], exp="im1b1", name="TestName")
    assert app.name == "TestName"


def test_key_input():
    assert INPUT.input_key == INPUT.input_image_key == "in"


def test_key_output():
    assert INPUT.output_image_key == "out"


def test_dtype():
    assert INPUT.dtype == "uint8"


def test_shape():
    assert INPUT.shape == (304, 251, 4)


def test_transform():
    assert INPUT.transform == (6.0, 0.0, 760056.0, 0.0, -6.0, 6946092.0)


def test_data():
    assert pyotb.ComputeImagesStatistics(INPUT).data == TEST_IMAGE_STATS


def test_metadata():
    assert INPUT.metadata["Metadata_1"] == "TIFFTAG_SOFTWARE=CSinG - 13 SEPTEMBRE 2012"


def test_nonraster_property():
    with pytest.raises(TypeError):
        assert pyotb.ReadImageInfo(INPUT).dtype == "uint8"


def test_elapsed_time():
    assert 0 < pyotb.ReadImageInfo(INPUT).elapsed_time < 1


# Other functions
def test_get_info():
    infos = INPUT.get_info()
    assert (infos["sizex"], infos["sizey"]) == (251, 304)
    bm_infos = pyotb.BandMathX([INPUT], exp="im1")["out"].get_info()
    assert infos == bm_infos


def test_get_statistics():
    assert INPUT.get_statistics() == TEST_IMAGE_STATS


def test_xy_to_rowcol():
    assert INPUT.get_rowcol_from_xy(760101, 6945977) == (19, 7)


def test_write():
    assert INPUT.write("/tmp/test_write.tif", ext_fname="nodata=0")
    INPUT["out"].filepath.unlink()


def test_frozen_app_write():
    app = pyotb.BandMath(INPUT, exp="im1b1", frozen=True)
    assert app.write("/tmp/test_frozen_app_write.tif")
    app["out"].filepath.unlink()

    app = pyotb.BandMath(INPUT, exp="im1b1", out="/tmp/test_frozen_app_write.tif", frozen=True)
    assert app.write()
    app["out"].filepath.unlink()


def test_output_write():
    assert INPUT["out"].write("/tmp/test_output_write.tif")
    INPUT["out"].filepath.unlink()


def test_frozen_output_write():
    app = pyotb.BandMath(INPUT, exp="im1b1", frozen=True)
    assert app["out"].write("/tmp/test_frozen_app_write.tif")
    app["out"].filepath.unlink()


def test_output_in_arg():
    info = pyotb.ReadImageInfo(INPUT["out"])
    assert info.data


# Slicer
def test_slicer_shape():
    extract = INPUT[:50, :60, :3]
    assert extract.shape == (50, 60, 3)
    assert extract.parameters["cl"] == ["Channel1", "Channel2", "Channel3"]


def test_slicer_preserve_dtype():
    extract = INPUT[:50, :60, :3]
    assert extract.dtype == "uint8"


def test_slicer_negative_band_index():
    assert INPUT[:50, :60, :-2].shape == (50, 60, 2)


def test_slicer_in_output():
    slc = pyotb.BandMath([INPUT], exp="im1b1")["out"][:50, :60, :-2]
    assert isinstance(slc, pyotb.core.Slicer)


# Arithmetic
def test_rational_operators():
    def _test(func, exp):
        meas = func(INPUT)
        ref = pyotb.BandMathX({"il": [FILEPATH], "exp": exp})
        for i in range(1, 5):
            compared = pyotb.CompareImages({"ref.in": ref, "meas.in": meas, "ref.channel": i, "meas.channel": i})
            assert (compared["count"], compared["mse"]) == (0, 0)

    _test(lambda x: x + x, "im1 + im1")
    _test(lambda x: x - x, "im1 - im1")
    _test(lambda x: x / x, "im1 div im1")
    _test(lambda x: x * x, "im1 mult im1")
    _test(lambda x: x + FILEPATH, "im1 + im1")
    _test(lambda x: x - FILEPATH, "im1 - im1")
    _test(lambda x: x / FILEPATH, "im1 div im1")
    _test(lambda x: x * FILEPATH, "im1 mult im1")
    _test(lambda x: FILEPATH + x, "im1 + im1")
    _test(lambda x: FILEPATH - x, "im1 - im1")
    _test(lambda x: FILEPATH / x, "im1 div im1")
    _test(lambda x: FILEPATH * x, "im1 mult im1")
    _test(lambda x: x + 2, "im1 + {2;2;2;2}")
    _test(lambda x: x - 2, "im1 - {2;2;2;2}")
    _test(lambda x: x / 2, "0.5 * im1")
    _test(lambda x: x * 2, "im1 * 2")
    _test(lambda x: x + 2.0, "im1 + {2.0;2.0;2.0;2.0}")
    _test(lambda x: x - 2.0, "im1 - {2.0;2.0;2.0;2.0}")
    _test(lambda x: x / 2.0, "0.5 * im1")
    _test(lambda x: x * 2.0, "im1 * 2.0")
    _test(lambda x: 2 + x, "{2;2;2;2} + im1")
    _test(lambda x: 2 - x, "{2;2;2;2} - im1")
    _test(lambda x: 2 / x, "{2;2;2;2} div im1")
    _test(lambda x: 2 * x, "2 * im1")
    _test(lambda x: 2.0 + x, "{2.0;2.0;2.0;2.0} + im1")
    _test(lambda x: 2.0 - x, "{2.0;2.0;2.0;2.0} - im1")
    _test(lambda x: 2.0 / x, "{2.0;2.0;2.0;2.0} div im1")
    _test(lambda x: 2.0 * x, "2.0 * im1")


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


# Essential apps
def test_app_readimageinfo():
    info = pyotb.ReadImageInfo(INPUT, quiet=True)
    assert (info["sizex"], info["sizey"]) == (251, 304)
    assert info["numberbands"] == 4


def test_app_computeimagestats():
    stats = pyotb.ComputeImagesStatistics([INPUT], quiet=True)
    assert stats["out.min"] == TEST_IMAGE_STATS["out.min"]


def test_app_computeimagestats_sliced():
    slicer_stats = pyotb.ComputeImagesStatistics(il=[INPUT[:10, :10, 0]], quiet=True)
    assert slicer_stats["out.min"] == [180]


def test_read_values_at_coords():
    assert INPUT[0, 0, 0] == 180
    assert INPUT[10, 20, :] == [207, 192, 172, 255]


# BandMath NDVI == RadiometricIndices NDVI ?
def test_ndvi_comparison():
    ndvi_bandmath = (INPUT[:, :, -1] - INPUT[:, :, [0]]) / (INPUT[:, :, -1] + INPUT[:, :, 0])
    ndvi_indices = pyotb.RadiometricIndices(INPUT, {"list": ["Vegetation:NDVI"], "channels.red": 1, "channels.nir": 4})
    assert ndvi_bandmath.exp == "((im1b4 - im1b1) / (im1b4 + im1b1))"
    assert ndvi_bandmath.write("/tmp/ndvi_bandmath.tif", pixel_type="float")
    assert ndvi_indices.write("/tmp/ndvi_indices.tif", pixel_type="float")

    compared = pyotb.CompareImages({"ref.in": ndvi_indices, "meas.in": "/tmp/ndvi_bandmath.tif"})
    assert (compared["count"], compared["mse"]) == (0, 0)
    thresholded_indices = pyotb.where(ndvi_indices >= 0.3, 1, 0)
    assert thresholded_indices["exp"] == "((im1b1 >= 0.3) ? 1 : 0)"
    thresholded_bandmath = pyotb.where(ndvi_bandmath >= 0.3, 1, 0)
    assert thresholded_bandmath["exp"] == "((((im1b4 - im1b1) / (im1b4 + im1b1)) >= 0.3) ? 1 : 0)"


def test_summarize_output():
    assert pyotb.summarize(INPUT["out"])


def test_summarize_strip_output():
    in_fn = FILEPATH
    in_fn_w_ext = FILEPATH + "?&skipcarto=1"
    out_fn = "/tmp/output.tif"
    out_fn_w_ext = out_fn + "?&box=10:10:10:10"

    baseline = [
        (in_fn, out_fn_w_ext, "out", {}, out_fn_w_ext),
        (in_fn, out_fn_w_ext, "out", {"strip_output_paths": True}, out_fn),
        (in_fn_w_ext, out_fn, "in", {}, in_fn_w_ext),
        (in_fn_w_ext, out_fn, "in", {"strip_input_paths": True}, in_fn)
    ]

    for inp, out, key, extra_args, expected in baseline:
        app = pyotb.ExtractROI({"in": inp, "out": out})
        summary = pyotb.summarize(app, **extra_args)
        assert summary["parameters"][key] == expected, \
            f"Failed for input {inp}, output {out}, args {extra_args}"


def test_pipeline_simple():
    # BandMath -> OrthoRectification -> ManageNoData
    app1 = pyotb.BandMath({"il": [FILEPATH], "exp": "im1b1"})
    app2 = pyotb.OrthoRectification({"io.in": app1})
    app3 = pyotb.ManageNoData({"in": app2})
    summary = pyotb.summarize(app3)
    assert summary == SIMPLE_SERIALIZATION


def test_pipeline_diamond():
    # Diamond graph
    app1 = pyotb.BandMath({"il": [FILEPATH], "exp": "im1b1"})
    app2 = pyotb.OrthoRectification({"io.in": app1})
    app3 = pyotb.ManageNoData({"in": app2})
    app4 = pyotb.BandMathX({"il": [app2, app3], "exp": "im1+im2"})
    summary = pyotb.summarize(app4)
    assert summary == COMPLEX_SERIALIZATION
