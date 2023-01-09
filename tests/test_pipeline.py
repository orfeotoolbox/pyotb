import sys
import os
import itertools
import pytest
import pyotb
from pyotb.helpers import logger


# List of buildings blocks, we can add other pyotb objects here
OTBAPPS_BLOCKS = [
    # lambda inp: pyotb.ExtractROI({"in": inp, "startx": 10, "starty": 10, "sizex": 50, "sizey": 50}),
    lambda inp: pyotb.ManageNoData({"in": inp, "mode": "changevalue"}),
    lambda inp: pyotb.DynamicConvert({"in": inp}),
    lambda inp: pyotb.Mosaic({"il": [inp]}),
    lambda inp: pyotb.BandMath({"il": [inp], "exp": "im1b1 + 1"}),
    lambda inp: pyotb.BandMathX({"il": [inp], "exp": "im1"})
]

PYOTB_BLOCKS = [
    lambda inp: 1 / (1 + abs(inp) * 2),
    lambda inp: inp[:80, 10:60, :],
]
PIPELINES_LENGTH = [1, 2, 3]

ALL_BLOCKS = PYOTB_BLOCKS + OTBAPPS_BLOCKS
FILEPATH = os.environ["TEST_INPUT_IMAGE"]
INPUT = pyotb.Input(FILEPATH)


def generate_pipeline(inp, building_blocks):
    """
    Create pipeline formed with the given building blocks

    Args:
      inp: input
      building_blocks: building blocks

    Returns:
        pipeline

    """
    # Create the N apps pipeline
    pipeline = []
    for app in building_blocks:
        new_app_inp = pipeline[-1] if pipeline else inp
        new_app = app(new_app_inp)
        pipeline.append(new_app)
    return pipeline


def combi(building_blocks, length):
    """Returns all possible combinations of N unique buildings blocks

    Args:
        building_blocks: building blocks
        length: length

    Returns:
        list of combinations

    """
    av = list(itertools.combinations(building_blocks, length))
    al = []
    for a in av:
        al += itertools.permutations(a)

    return list(set(al))


def pipeline2str(pipeline):
    """Prints the pipeline blocks

    Args:
      pipeline: pipeline

    Returns:
        a string

    """
    return " > ".join([INPUT.__class__.__name__] + [f"{i}.{app.name.split()[0]}"
                                                       for i, app in enumerate(pipeline)])


def make_pipelines_list():
    """Create a list of pipelines using different lengths and blocks"""
    blocks = {FILEPATH: OTBAPPS_BLOCKS,  # for filepath, we can't use Slicer or Operation
              INPUT: ALL_BLOCKS}
    pipelines = []
    names = []
    for inp, blocks in blocks.items():
        # Generate pipelines of different length
        for length in PIPELINES_LENGTH:
            blocks_combis = combi(building_blocks=blocks, length=length)
            for block in blocks_combis:
                pipe = generate_pipeline(inp, block)
                name = pipeline2str(pipe)
                if name not in names:
                    pipelines.append(pipe)
                    names.append(name)

    return pipelines, names


PIPELINES, NAMES = make_pipelines_list()


@pytest.mark.parametrize("pipe", PIPELINES, ids=NAMES)
def test_pipeline_shape(pipe):
    for i, app in enumerate(pipe):
        print(app.shape)
        assert bool(app.shape)


@pytest.mark.parametrize("pipe", PIPELINES, ids=NAMES)
def test_pipeline_shape_nointermediate(pipe):
    app = [pipe[-1]][0]
    assert bool(app.shape)


@pytest.mark.parametrize("pipe", PIPELINES, ids=NAMES)
def test_pipeline_shape_backward(pipe):
    for i, app in enumerate(reversed(pipe)):
        assert bool(app.shape)


@pytest.mark.parametrize("pipe", PIPELINES, ids=NAMES)
def test_pipeline_write(pipe):
    for i, app in enumerate(pipe):
        out = f"/tmp/out_{i}.tif"
        if os.path.isfile(out):
            os.remove(out)
        app.write(out)
        assert os.path.isfile(out)


@pytest.mark.parametrize("pipe", PIPELINES, ids=NAMES)
def test_pipeline_write_nointermediate(pipe):
    app = [pipe[-1]][0]
    out = f"/tmp/out_0.tif"
    if os.path.isfile(out):
        os.remove(out)
    app.write(out)
    assert os.path.isfile(out)


@pytest.mark.parametrize("pipe", PIPELINES, ids=NAMES)
def test_pipeline_write_backward(pipe):
    for i, app in enumerate(reversed(pipe)):
        out = f"/tmp/out_{i}.tif"
        if os.path.isfile(out):
            os.remove(out)
        app.write(out)
        assert os.path.isfile(out)
