import json
from pathlib import Path
import pyotb

FILEPATH = "https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/raw/develop/Data/Input/SP67_FR_subset_1.tif"
INPUT = pyotb.Input(FILEPATH)
TEST_IMAGE_STATS = {
    'out.mean': [79.5505, 109.225, 115.456, 249.349],
    'out.min': [33, 64, 91, 47],
    'out.max': [255, 255, 230, 255],
    'out.std': [51.0754, 35.3152, 23.4514, 20.3827]
}

json_file = Path(__file__).parent / "serialized_apps.json"
with json_file.open("r") as js:
    data = json.load(js)
SIMPLE_SERIALIZATION = data["SIMPLE"]
COMPLEX_SERIALIZATION = data["COMPLEX"]
