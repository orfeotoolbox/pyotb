# pyotb: Orfeo ToolBox for Python

[![latest release](https://forgemia.inra.fr/orfeo-toolbox/pyotb/-/badges/release.svg)](https://forgemia.inra.fr/orfeo-toolbox/pyotb/-/releases)
[![pipeline status](https://forgemia.inra.fr/orfeo-toolbox/pyotb/badges/develop/pipeline.svg)](https://forgemia.inra.fr/orfeo-toolbox/pyotb/-/commits/develop)
[![coverage report](https://forgemia.inra.fr/orfeo-toolbox/pyotb/badges/develop/coverage.svg)](https://forgemia.inra.fr/orfeo-toolbox/pyotb/-/commits/develop)
[![read the docs status](https://readthedocs.org/projects/pyotb/badge/?version=master)](https://pyotb.readthedocs.io/en/master/)

**pyotb** wraps the [Orfeo Toolbox](https://www.orfeo-toolbox.org/) in a pythonic, developer friendly 
fashion.  

## Key features

- Easy use of Orfeo ToolBox (OTB) applications from python
- Simplify common sophisticated I/O features of OTB
- Lazy execution of operations thanks to OTB streaming mechanism
- Interoperable with popular python libraries ([numpy](https://numpy.org/) and 
[rasterio](https://rasterio.readthedocs.io/))
- Extensible

Documentation hosted at [pyotb.readthedocs.io](https://pyotb.readthedocs.io/).

## Example

Building a simple pipeline with OTB applications

```py
import pyotb

# RigidTransformResample, with input parameters as dict
resampled = pyotb.RigidTransformResample({
    "in": "https://myserver.ia/input.tif",  # Note: no /vsicurl/
    "interpolator": "linear", 
    "transform.type.id.scaley": 0.5,
    "transform.type.id.scalex": 0.5
})

# OpticalCalibration, with input parameters as args
calib = pyotb.OpticalCalibration(resampled)

# BandMath, with input parameters as kwargs
ndvi = pyotb.BandMath(calib, exp="ndvi(im1b1, im1b4)")

# Pythonic slicing
roi = ndvi[20:586, 9:572]

# Pipeline execution. The actual computation happens here!
roi.write("output.tif", "float")
```
