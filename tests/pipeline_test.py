import sys
import itertools
import os
import pyotb

# List of buildings blocks
# We can add other pyotb objects here
OTBAPPS_BLOCKS = [
    #lambda inp: pyotb.ExtractROI({"in": inp, "startx": 10, "starty": 10, "sizex": 50, "sizey": 50}),
    lambda inp: pyotb.ManageNoData({"in": inp, "mode": "changevalue"}),
    lambda inp: pyotb.DynamicConvert({"in": inp}),
    lambda inp: pyotb.Mosaic({"il": [inp]}),
    lambda inp: pyotb.BandMath({"il": [inp], "exp": "im1b1 + 1"}),
    lambda inp: pyotb.BandMathX({"il": [inp], "exp": "im1"})
]

PYOTB_BLOCKS = [
    lambda inp: 1 + abs(inp) * 2,
    lambda inp: inp[:80, 10:60, :],
]

ALL_BLOCKS = PYOTB_BLOCKS + OTBAPPS_BLOCKS

# These apps are problematic when used in pipelines with intermediate outputs
# (cf https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/issues/2290)
PROBLEMATIC_APPS = ['DynamicConvert', 'BandMath']


def backward():
    """
    Return True if backward mode.
    In backward mode applications are tested from the end to the beginning of the pipeline.
    """

def check_app_write(app, out):
    """
    Check that the app write correctly its output
    """
    print(f"Checking app {app.name} writing")

    # Remove output file it already there
    if os.path.isfile(out):
        os.remove(out)
    # Write
    try:
        app.write(out)
    except Exception as e:
        print("\n\033[91mWRITE ERROR\033[0m")
        print(e)
        return False
    # Report
    if not os.path.isfile(out):
        return False
    return True


filepath = 'image.tif'
pyotb_input = pyotb.Input(filepath)
args = [arg.lower() for arg in sys.argv[1:]] if len(sys.argv) > 1 else []

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


def test_pipeline(pipeline):
    """Test the pipeline

    Args:
      pipeline: pipeline (list of pyotb objects)

    """
    report = {"shapes_errs": [], "write_errs": []}

    # Test outputs shapes
    pipeline_items = [pipeline[-1]] if "no-intermediate-result" in args else pipeline
    generator = lambda: enumerate(pipeline_items)
    if "backward" in args:
        print("Perform tests in backward mode")
        generator = lambda: enumerate(reversed(pipeline_items))
    if "shape" in args:
        for i, app in generator():
            try:
                print(f"Trying to access shape of app {app.name} output...")
                shape = app.shape
                print(f"App {app.name} output shape is {shape}")
            except Exception as e:
                print("\n\033[91mGET SHAPE ERROR\033[0m")
                print(e)
                report["shapes_errs"].append(i)

    # Test all pipeline outputs
    if "write" in args:
        for i, app in generator():
            if not check_app_write(app, f"/tmp/out_{i}.tif"):
                report["write_errs"].append(i)

    return report


def pipeline2str(pipeline):
    """Prints the pipeline blocks

    Args:
      pipeline: pipeline

    Returns:
        a string

    """
    return " \u2192 ".join([inp.__class__.__name__] + [f"{i}.{app.name.split(' ')[0]}"
                                                       for i, app in enumerate(pipeline)])


# Generate pipelines of different length
blocks = {filepath: OTBAPPS_BLOCKS,  # for filepath, we can't use Slicer or Operation
          pyotb_input: ALL_BLOCKS}
pipelines = []
for inp, blocks in blocks.items():
    for length in [1, 2, 3]:
        print(f"Testing pipelines of length {length}")
        blocks_combis = combi(building_blocks=blocks, length=length)
        for block_combi in blocks_combis:
            pipelines.append(generate_pipeline(inp, block_combi))

# Test pipelines
pipelines.sort(key=lambda x: f"{len(x)}" "".join([app.name for app in x]))  # Sort by length then alphabetical
results = {}
for pipeline in pipelines:
    print("\033[94m" f"\nTesting the following pipeline: {pipeline2str(pipeline)}\n" "\033[0m")
    results.update({tuple(pipeline): test_pipeline(pipeline)})

# Summary
cols = max([len(pipeline2str(pipeline)) for pipeline in pipelines]) + 1
print(f'Tests summary (\033[93mTest options: {"; ".join(args)}\033[0m)')
print("Pipeline".ljust(cols) + " | Status (reason)")
print("-" * cols + "-|-" + "-" * 20)
nb_fails = 0
allowed_to_fail = 0
for pipeline, errs in results.items():
    has_err = sum(len(value) for key, value in errs.items()) > 0
    graph = pipeline2str(pipeline)
    msg = graph + " ".ljust(cols - len(graph))
    if has_err:
        msg = f"\033[91m{msg}\033[0m"
    msg += " | "
    if has_err:
        causes = [f"{section}: " + ", ".join([f"app{i}" for i in out_ids])
                  for section, out_ids in errs.items() if out_ids]
        msg += "\033[91mFAIL\033[0m (" + "; ".join(causes) + ")"
        if any([app.name in PROBLEMATIC_APPS for app in pipeline]):
            allowed_to_fail += 1
        else:
            nb_fails += 1
    else:
        msg += "\033[92mPASS\033[0m"
    print(msg)
print(f"End of summary ({nb_fails} error(s), {allowed_to_fail} 'allowed to fail' error(s))", flush=True)
assert nb_fails == 0, "One of the pipelines have failed. Please read the report."
