from tests_data import INPUT
import pyotb

INPUT2 = pyotb.Input(
    "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/"
    "47/Q/RU/2021/12/S2B_47QRU_20211227_0_L2A/B04.tif"
)

def test_metadata():
    assert "ProjectionRef", "TIFFTAG_SOFTWARE" in INPUT.metadata
    assert "ProjectionRef", "OVR_RESAMPLING_ALG" in INPUT2.metadata
