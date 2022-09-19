import os
import pyotb

filepath = os.environ["TEST_INPUT_IMAGE"]


def test_pipeline_simple():
    # BandMath -> OrthoRectification -> ManageNoData
    app1 = pyotb.BandMath({'il': [filepath], 'exp': 'im1b1'})
    app2 = pyotb.OrthoRectification({'io.in': app1})
    app3 = pyotb.ManageNoData({'in': app2})
    summary = app3.summarize()
    reference = {'name': 'ManageNoData',
                'parameters': {'in': {'name': 'OrthoRectification',
                'parameters': {'io.in': {'name': 'BandMath',
                    'parameters': {'il': ('tests/image.tif',), 'exp': 'im1b1'}},
                    'map': 'utm',
                    'outputs.isotropic': True}},
                'mode': 'buildmask'}}
    assert summary == reference

def test_pipeline_diamond():
    # Diamond graph
    app1 = pyotb.BandMath({'il': [filepath], 'exp': 'im1b1'})
    app2 = pyotb.OrthoRectification({'io.in': app1})
    app3 = pyotb.ManageNoData({'in': app2})
    app4 = pyotb.BandMathX({'il': [app2, app3], 'exp': 'im1+im2'})
    summary = app4.summarize()
    reference = {'name': 'BandMathX',
                'parameters': {'il': [{'name': 'OrthoRectification',
                    'parameters': {'io.in': {'name': 'BandMath',
                    'parameters': {'il': ('tests/image.tif',), 'exp': 'im1b1'}},
                    'map': 'utm',
                    'outputs.isotropic': True}},
                {'name': 'ManageNoData',
                    'parameters': {'in': {'name': 'OrthoRectification',
                    'parameters': {'io.in': {'name': 'BandMath',
                        'parameters': {'il': ('tests/image.tif',), 'exp': 'im1b1'}},
                    'map': 'utm',
                    'outputs.isotropic': True}},
                    'mode': 'buildmask'}}],
                'exp': 'im1+im2'}}
    assert summary == reference
