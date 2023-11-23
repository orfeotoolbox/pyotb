import json
from pathlib import Path
import pyotb


SPOT_IMG_URL = "https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/raw/develop/Data/Input/SP67_FR_subset_1.tif"
PLEIADES_IMG_URL = "https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/raw/develop/Data/Baseline/OTB/Images/prTvOrthoRectification_pleiades-1_noDEM.tif"
INPUT = pyotb.Input(SPOT_IMG_URL)


TEST_IMAGE_STATS = {
    "out.mean": [79.5505, 109.225, 115.456, 249.349],
    "out.min": [33, 64, 91, 47],
    "out.max": [255, 255, 230, 255],
    "out.std": [51.0754, 35.3152, 23.4514, 20.3827],
}

json_file = Path(__file__).parent / "pipeline_summary.json"
with json_file.open("r", encoding="utf-8") as js:
    data = json.load(js)
SIMPLE_SERIALIZATION = data["SIMPLE"]
DIAMOND_SERIALIZATION = data["DIAMOND"]
