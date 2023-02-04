import pyotb
from tests_data import *


def test_pipeline_simple():
    # BandMath -> OrthoRectification -> ManageNoData
    app1 = pyotb.BandMath({"il": [FILEPATH], "exp": "im1b1"})
    app2 = pyotb.OrthoRectification({"io.in": app1})
    app3 = pyotb.ManageNoData({"in": app2})
    summary = app3.summarize()
    assert summary == SIMPLE_SERIALIZATION


def test_pipeline_diamond():
    # Diamond graph
    app1 = pyotb.BandMath({"il": [FILEPATH], "exp": "im1b1"})
    app2 = pyotb.OrthoRectification({"io.in": app1})
    app3 = pyotb.ManageNoData({"in": app2})
    app4 = pyotb.BandMathX({"il": [app2, app3], "exp": "im1+im2"})
    summary = app4.summarize()
    assert summary == COMPLEX_SERIALIZATION
