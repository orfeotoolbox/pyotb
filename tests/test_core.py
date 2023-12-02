import pytest
import numpy as np

import pyotb
from tests_data import *


def test_app_parameters():
    # Input / ExtractROI
    assert INPUT.parameters
    assert (INPUT.parameters["sizex"], INPUT.parameters["sizey"]) == (251, 304)
    # OrthoRectification
    app = pyotb.OrthoRectification(INPUT)
    assert isinstance(app.parameters["map"], str)
    assert app.parameters["map"] == "utm"
    assert "map" in app._auto_parameters
    app.set_parameters({"map": "epsg", "map.epsg.code": 2154})
    assert app.parameters["map"] == "epsg"
    assert "map" in app._settings and "map" not in app._auto_parameters
    assert app.parameters["map.epsg.code"] == app.app.GetParameters()["map.epsg.code"]
    # Orthorectification with underscore kwargs
    app = pyotb.OrthoRectification(io_in=INPUT, map_epsg_code=2154)
    assert app.parameters["map.epsg.code"] == 2154
    # ManageNoData
    app = pyotb.ManageNoData(INPUT)
    assert "usenan" in app._auto_parameters
    assert "mode.buildmask.inv" in app._auto_parameters
    # OpticalCalibration
    app = pyotb.OpticalCalibration(pyotb.Input(PLEIADES_IMG_URL), level="toa")
    assert "milli" in app._auto_parameters
    assert "clamp" in app._auto_parameters
    assert app._auto_parameters["acqui.year"] == 2012
    assert app._auto_parameters["acqui.sun.elev"] == 23.836299896240234


def test_app_properties():
    assert INPUT.input_key == INPUT.input_image_key == "in"
    assert INPUT.output_key == INPUT.output_image_key == "out"
    with pytest.raises(KeyError):
        pyotb.BandMath(INPUT, expression="im1b1")
    # Test user can set custom name
    app = pyotb.App("BandMath", [INPUT], exp="im1b1", name="TestName")
    assert app.name == "TestName"
    # Test data dict is not empty
    app = pyotb.ReadImageInfo(INPUT)
    assert app.data
    # Test elapsed time is not null
    assert 0 < app.elapsed_time < 1


def test_app_input_vsi():
    # Ensure old way is still working: ExtractROI will raise RuntimeError if a path is malformed
    pyotb.Input("/vsicurl/" + SPOT_IMG_URL)
    # Simple remote file
    info = pyotb.ReadImageInfo("https://fake.com/image.tif", frozen=True)
    assert (
        info.app.GetParameterValue("in")
        == info.parameters["in"]
        == "/vsicurl/https://fake.com/image.tif"
    )
    # Compressed single file archive
    info = pyotb.ReadImageInfo("image.tif.zip", frozen=True)
    assert (
        info.app.GetParameterValue("in")
        == info.parameters["in"]
        == "/vsizip/image.tif.zip"
    )
    # File within compressed remote archive
    info = pyotb.ReadImageInfo("https://fake.com/archive.tar.gz/image.tif", frozen=True)
    assert (
        info.app.GetParameterValue("in")
        == info.parameters["in"]
        == "/vsitar//vsicurl/https://fake.com/archive.tar.gz/image.tif"
    )
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


def test_img_properties():
    assert INPUT.dtype == "uint8"
    assert INPUT.shape == (304, 251, 4)
    assert INPUT.transform == (6.0, 0.0, 760056.0, 0.0, -6.0, 6946092.0)
    with pytest.raises(TypeError):
        assert pyotb.ReadImageInfo(INPUT).dtype == "uint8"


def test_img_metadata():
    assert "ProjectionRef" in INPUT.metadata
    assert "TIFFTAG_SOFTWARE" in INPUT.metadata
    inp2 = pyotb.Input(
        "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/"
        "47/Q/RU/2021/12/S2B_47QRU_20211227_0_L2A/B04.tif"
    )
    assert "ProjectionRef" in inp2.metadata
    assert "OVR_RESAMPLING_ALG" in inp2.metadata
    # Metadata with numeric values (e.g. TileHintX)
    fp = (
        "https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/raw/develop/"
        "Data/Input/radarsat2/RADARSAT2_ALTONA_300_300_VV.tif?inline=false"
    )
    app = pyotb.BandMath({"il": [fp], "exp": "im1b1"})
    assert "TileHintX" in app.metadata


def test_essential_apps():
    readimageinfo = pyotb.ReadImageInfo(INPUT, quiet=True)
    assert (readimageinfo["sizex"], readimageinfo["sizey"]) == (251, 304)
    assert readimageinfo["numberbands"] == 4
    computeimagestats = pyotb.ComputeImagesStatistics([INPUT], quiet=True)
    assert computeimagestats["out.min"] == TEST_IMAGE_STATS["out.min"]
    slicer_computeimagestats = pyotb.ComputeImagesStatistics(
        il=[INPUT[:10, :10, 0]], quiet=True
    )
    assert slicer_computeimagestats["out.min"] == [180]


def test_get_statistics():
    stats_data = pyotb.ComputeImagesStatistics(INPUT).data
    assert stats_data == TEST_IMAGE_STATS
    assert INPUT.get_statistics() == TEST_IMAGE_STATS


def test_get_info():
    infos = INPUT.get_info()
    assert (infos["sizex"], infos["sizey"]) == (251, 304)
    bm_infos = pyotb.BandMathX([INPUT], exp="im1")["out"].get_info()
    assert infos == bm_infos


def test_read_values_at_coords():
    assert INPUT[0, 0, 0] == 180
    assert INPUT[10, 20, :] == [207, 192, 172, 255]


def test_xy_to_rowcol():
    assert INPUT.get_rowcol_from_xy(760101, 6945977) == (19, 7)


def test_write():
    # Write string filepath
    assert INPUT.write("/dev/shm/test_write.tif")
    INPUT["out"].filepath.unlink()
    # With Path filepath
    assert INPUT.write(Path("/dev/shm/test_write.tif"))
    INPUT["out"].filepath.unlink()
    # Write to uint8
    assert INPUT.write(Path("/dev/shm/test_write.tif"), pixel_type="uint8")
    assert INPUT["out"].dtype == "uint8"
    INPUT["out"].filepath.unlink()
    # Write frozen app
    frozen_app = pyotb.BandMath(INPUT, exp="im1b1", frozen=True)
    assert frozen_app.write("/dev/shm/test_frozen_app_write.tif")
    frozen_app["out"].filepath.unlink()
    frozen_app_init_with_outfile = pyotb.BandMath(
        INPUT, exp="im1b1", out="/dev/shm/test_frozen_app_write.tif", frozen=True
    )
    assert frozen_app_init_with_outfile.write(pixel_type="uint16")
    assert frozen_app_init_with_outfile.dtype == "uint16"
    frozen_app_init_with_outfile["out"].filepath.unlink()


def test_write_multi_output():
    mss = pyotb.MeanShiftSmoothing(
        SPOT_IMG_URL,
        fout="/dev/shm/test_ext_fn_fout.tif",
        foutpos="/dev/shm/test_ext_fn_foutpos.tif",
    )

    mss = pyotb.MeanShiftSmoothing(SPOT_IMG_URL)
    assert mss.write(
        {
            "fout": "/dev/shm/test_ext_fn_fout.tif",
            "foutpos": "/dev/shm/test_ext_fn_foutpos.tif",
        },
        ext_fname={"nodata": 0, "gdal:co:COMPRESS": "DEFLATE"},
    )

    dr = pyotb.DimensionalityReduction(
        SPOT_IMG_URL, out="/dev/shm/1.tif", outinv="/dev/shm/2.tif"
    )
    dr = pyotb.DimensionalityReduction(SPOT_IMG_URL)
    assert dr.write(
        {"out": "/dev/shm/1.tif", "outinv": "/dev/shm/2.tif"}
    )


def test_write_ext_fname():
    def _check(expected: str, key: str = "out", app=INPUT.app):
        fn = app.GetParameterString(key)
        assert "?&" in fn
        assert fn.split("?&", 1)[1] == expected

    assert INPUT.write("/dev/shm/test_write.tif", ext_fname="nodata=0")
    _check("nodata=0")
    assert INPUT.write("/dev/shm/test_write.tif", ext_fname={"nodata": "0"})
    _check("nodata=0")
    assert INPUT.write("/dev/shm/test_write.tif", ext_fname={"nodata": 0})
    _check("nodata=0")
    assert INPUT.write(
        "/dev/shm/test_write.tif",
        ext_fname={"nodata": 0, "gdal:co:COMPRESS": "DEFLATE"},
    )
    _check("nodata=0&gdal:co:COMPRESS=DEFLATE")
    assert INPUT.write(
        "/dev/shm/test_write.tif", ext_fname="nodata=0&gdal:co:COMPRESS=DEFLATE"
    )
    _check("nodata=0&gdal:co:COMPRESS=DEFLATE")
    assert INPUT.write(
        "/dev/shm/test_write.tif?&box=0:0:10:10",
        ext_fname={"nodata": "0", "gdal:co:COMPRESS": "DEFLATE", "box": "0:0:20:20"},
    )
    # Check that the bbox is the one specified in the filepath, not the one
    # specified in `ext_filename`
    _check("nodata=0&gdal:co:COMPRESS=DEFLATE&box=0:0:10:10")
    assert INPUT.write(
        "/dev/shm/test_write.tif?&box=0:0:10:10",
        ext_fname="nodata=0&gdal:co:COMPRESS=DEFLATE&box=0:0:20:20",
    )
    _check("nodata=0&gdal:co:COMPRESS=DEFLATE&box=0:0:10:10")
    INPUT["out"].filepath.unlink()

    mmsd = pyotb.MorphologicalMultiScaleDecomposition(INPUT)
    mmsd.write(
        {
            "outconvex": "/dev/shm/outconvex.tif?&nodata=1",
            "outconcave": "/dev/shm/outconcave.tif?&nodata=2",
            "outleveling": "/dev/shm/outleveling.tif?&nodata=3",
        },
        ext_fname={"nodata": 0, "gdal:co:COMPRESS": "DEFLATE"},
    )
    _check("nodata=1&gdal:co:COMPRESS=DEFLATE", key="outconvex", app=mmsd.app)
    _check("nodata=2&gdal:co:COMPRESS=DEFLATE", key="outconcave", app=mmsd.app)
    _check("nodata=3&gdal:co:COMPRESS=DEFLATE", key="outleveling", app=mmsd.app)
    mmsd["outconvex"].filepath.unlink()
    mmsd["outconcave"].filepath.unlink()
    mmsd["outleveling"].filepath.unlink()


def test_output():
    assert INPUT["out"].write("/dev/shm/test_output_write.tif")
    INPUT["out"].filepath.unlink()
    frozen_app = pyotb.BandMath(INPUT, exp="im1b1", frozen=True)
    assert frozen_app["out"].write("/dev/shm/test_frozen_app_write.tif")
    frozen_app["out"].filepath.unlink()
    info_from_output_obj = pyotb.ReadImageInfo(INPUT["out"])
    assert info_from_output_obj.data


# Slicer
def test_slicer():
    sliced = INPUT[:50, :60, :3]
    assert sliced.parameters["cl"] == ["Channel1", "Channel2", "Channel3"]
    assert sliced.shape == (50, 60, 3)
    assert sliced.dtype == "uint8"
    sliced_negative_band_idx = INPUT[:50, :60, :-2]
    assert sliced_negative_band_idx.shape == (50, 60, 2)
    sliced_from_output = pyotb.BandMath([INPUT], exp="im1b1")["out"][:50, :60, :-2]
    assert isinstance(sliced_from_output, pyotb.core.Slicer)


# Operation and LogicalOperation
def test_operator_expressions():
    op = INPUT / 255 * 128
    assert (
        op.exp
        == "((im1b1 / 255) * 128);((im1b2 / 255) * 128);((im1b3 / 255) * 128);((im1b4 / 255) * 128)"
    )
    assert op.dtype == "float32"
    assert abs(INPUT).exp == "(abs(im1b1));(abs(im1b2));(abs(im1b3));(abs(im1b4))"
    summed_bands = sum(INPUT[:, :, b] for b in range(INPUT.shape[-1]))
    assert summed_bands.exp == "((((0 + im1b1) + im1b2) + im1b3) + im1b4)"


def operation_test(func, exp):
    meas = func(INPUT)
    ref = pyotb.BandMathX({"il": [SPOT_IMG_URL], "exp": exp})
    for i in range(1, 5):
        compared = pyotb.CompareImages(
            {"ref.in": ref, "meas.in": meas, "ref.channel": i, "meas.channel": i}
        )
        assert (compared["count"], compared["mse"]) == (0, 0)


def test_operation_add():
    operation_test(lambda x: x + x, "im1 + im1")
    operation_test(lambda x: x + INPUT, "im1 + im1")
    operation_test(lambda x: INPUT + x, "im1 + im1")
    operation_test(lambda x: x + 2, "im1 + {2;2;2;2}")
    operation_test(lambda x: x + 2.0, "im1 + {2.0;2.0;2.0;2.0}")
    operation_test(lambda x: 2 + x, "{2;2;2;2} + im1")
    operation_test(lambda x: 2.0 + x, "{2.0;2.0;2.0;2.0} + im1")


def test_operation_sub():
    operation_test(lambda x: x - x, "im1 - im1")
    operation_test(lambda x: x - INPUT, "im1 - im1")
    operation_test(lambda x: INPUT - x, "im1 - im1")
    operation_test(lambda x: x - 2, "im1 - {2;2;2;2}")
    operation_test(lambda x: x - 2.0, "im1 - {2.0;2.0;2.0;2.0}")
    operation_test(lambda x: 2 - x, "{2;2;2;2} - im1")
    operation_test(lambda x: 2.0 - x, "{2.0;2.0;2.0;2.0} - im1")


def test_operation_mult():
    operation_test(lambda x: x * x, "im1 mult im1")
    operation_test(lambda x: x * INPUT, "im1 mult im1")
    operation_test(lambda x: INPUT * x, "im1 mult im1")
    operation_test(lambda x: x * 2, "im1 * 2")
    operation_test(lambda x: x * 2.0, "im1 * 2.0")
    operation_test(lambda x: 2 * x, "2 * im1")
    operation_test(lambda x: 2.0 * x, "2.0 * im1")


def test_operation_div():
    operation_test(lambda x: x / x, "im1 div im1")
    operation_test(lambda x: x / INPUT, "im1 div im1")
    operation_test(lambda x: INPUT / x, "im1 div im1")
    operation_test(lambda x: x / 2, "im1 * 0.5")
    operation_test(lambda x: x / 2.0, "im1 * 0.5")
    operation_test(lambda x: 2 / x, "{2;2;2;2} div im1")
    operation_test(lambda x: 2.0 / x, "{2.0;2.0;2.0;2.0} div im1")


# BandMath NDVI == RadiometricIndices NDVI ?
def test_ndvi_comparison():
    ndvi_bandmath = (INPUT[:, :, -1] - INPUT[:, :, [0]]) / (
        INPUT[:, :, -1] + INPUT[:, :, 0]
    )
    ndvi_indices = pyotb.RadiometricIndices(
        INPUT, {"list": ["Vegetation:NDVI"], "channels.red": 1, "channels.nir": 4}
    )
    assert ndvi_bandmath.exp == "((im1b4 - im1b1) / (im1b4 + im1b1))"
    assert ndvi_bandmath.write("/dev/shm/ndvi_bandmath.tif", "float")
    assert ndvi_indices.write("/dev/shm/ndvi_indices.tif", "float")

    compared = pyotb.CompareImages(
        {"ref.in": ndvi_indices, "meas.in": "/dev/shm/ndvi_bandmath.tif"}
    )
    assert (compared["count"], compared["mse"]) == (0, 0)
    thresholded_indices = pyotb.where(ndvi_indices >= 0.3, 1, 0)
    assert thresholded_indices["exp"] == "((im1b1 >= 0.3) ? 1 : 0)"
    thresholded_bandmath = pyotb.where(ndvi_bandmath >= 0.3, 1, 0)
    assert (
        thresholded_bandmath["exp"]
        == "((((im1b4 - im1b1) / (im1b4 + im1b1)) >= 0.3) ? 1 : 0)"
    )


# Tests for functions.py
def test_binary_mask_where():
    # Create binary mask based on several possible values
    values = [1, 2, 3, 4]
    res = pyotb.where(pyotb.any(INPUT[:, :, 0] == value for value in values), 255, 0)
    assert (
        res.exp
        == "(((((im1b1 == 1) || (im1b1 == 2)) || (im1b1 == 3)) || (im1b1 == 4)) ? 255 : 0)"
    )


# Tests for summarize()
def test_summarize_pipeline_simple():
    app1 = pyotb.OrthoRectification({"io.in": SPOT_IMG_URL})
    app2 = pyotb.BandMath({"il": [app1], "exp": "im1b1"})
    app3 = pyotb.ManageNoData({"in": app2})
    summary = pyotb.summarize(app3)
    assert SIMPLE_SERIALIZATION == summary


def test_summarize_pipeline_diamond():
    app1 = pyotb.BandMath({"il": [SPOT_IMG_URL], "exp": "im1b1"})
    app2 = pyotb.OrthoRectification({"io.in": app1})
    app3 = pyotb.ManageNoData({"in": app2})
    app4 = pyotb.BandMathX({"il": [app2, app3], "exp": "im1+im2"})
    summary = pyotb.summarize(app4)
    assert DIAMOND_SERIALIZATION == summary


def test_summarize_output_obj():
    assert pyotb.summarize(INPUT["out"])


def test_summarize_strip_output():
    in_fn = "/vsicurl/" + SPOT_IMG_URL
    in_fn_w_ext = "/vsicurl/" + SPOT_IMG_URL + "?&skipcarto=1"
    out_fn = "/dev/shm/output.tif"
    out_fn_w_ext = out_fn + "?&box=10:10:10:10"

    baseline = [
        (in_fn, out_fn_w_ext, "out", {}, out_fn_w_ext),
        (in_fn, out_fn_w_ext, "out", {"strip_outpath": True}, out_fn),
        (in_fn_w_ext, out_fn, "in", {}, in_fn_w_ext),
        (in_fn_w_ext, out_fn, "in", {"strip_inpath": True}, in_fn),
    ]

    for inp, out, key, extra_args, expected in baseline:
        app = pyotb.ExtractROI({"in": inp, "out": out})
        summary = pyotb.summarize(app, **extra_args)
        assert (
            summary["parameters"][key] == expected
        ), f"Failed for input {inp}, output {out}, args {extra_args}"


def test_summarize_consistency():
    app_fns = [
        lambda inp: pyotb.ExtractROI(
            {"in": inp, "startx": 10, "starty": 10, "sizex": 50, "sizey": 50}
        ),
        lambda inp: pyotb.ManageNoData({"in": inp, "mode": "changevalue"}),
        lambda inp: pyotb.DynamicConvert({"in": inp}),
        lambda inp: pyotb.Mosaic({"il": [inp]}),
        lambda inp: pyotb.BandMath({"il": [inp], "exp": "im1b1 + 1"}),
        lambda inp: pyotb.BandMathX({"il": [inp], "exp": "im1"}),
        lambda inp: pyotb.OrthoRectification({"io.in": inp}),
    ]

    def operator_test(app_fn):
        """
        Here we create 2 summaries:
        - summary of the app before write()
        - summary of the app after write()
        Then we check that both only differ with the output parameter
        """
        app = app_fn(inp=SPOT_IMG_URL)
        out_file = "/dev/shm/out.tif"
        out_key = app.output_image_key
        summary_wo_wrt = pyotb.summarize(app)
        app.write(out_file)
        summay_w_wrt = pyotb.summarize(app)
        app[out_key].filepath.unlink()
        summary_wo_wrt["parameters"].update({out_key: out_file})
        assert summary_wo_wrt == summay_w_wrt

    for app_fn in app_fns:
        operator_test(app_fn)


# Numpy tests
def test_numpy_exports_dic():
    INPUT.export()
    exported_array = INPUT.exports_dic[INPUT.output_image_key]["array"]
    assert isinstance(exported_array, np.ndarray)
    assert exported_array.dtype == "uint8"
    del INPUT.exports_dic["out"]
    INPUT["out"].export()
    assert INPUT["out"].output_image_key in INPUT["out"].exports_dic


def test_numpy_conversions():
    array = INPUT.to_numpy()
    assert array.dtype == np.uint8
    assert array.shape == INPUT.shape
    assert (array.min(), array.max()) == (33, 255)
    # Sliced img to array
    sliced = INPUT[:100, :200, :3]
    sliced_array = sliced.to_numpy()
    assert sliced_array.dtype == np.uint8
    assert sliced_array.shape == (100, 200, 3)
    # Test auto convert to numpy
    assert isinstance(np.array(INPUT), np.ndarray)
    assert INPUT.shape == np.array(INPUT).shape
    assert INPUT[19, 7] == list(INPUT.to_numpy()[19, 7])
    # Add noise test from the docs
    white_noise = np.random.normal(0, 50, size=INPUT.shape)
    noisy_image = INPUT + white_noise
    assert isinstance(noisy_image, pyotb.core.App)
    assert noisy_image.shape == INPUT.shape


def test_numpy_to_rasterio():
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
