# pyotb: a pythonic extension of Orfeo Toolbox

[![latest release](https://gitlab.orfeo-toolbox.org/nicolasnn/pyotb/-/badges/release.svg)](https://gitlab.orfeo-toolbox.org/nicolasnn/pyotb/-/releases)
[![pipeline status](https://gitlab.orfeo-toolbox.org/nicolasnn/pyotb/badges/master/pipeline.svg)](https://gitlab.orfeo-toolbox.org/nicolasnn/pyotb/-/commits/master)
[![read the docs status](https://readthedocs.org/projects/pyotb/badge/?version=master)](https://pyotb.readthedocs.io/en/master/)

**pyotb** wraps the [Orfeo Toolbox](https://www.orfeo-toolbox.org/) (OTB)
python bindings to make it more developer friendly.  

## Key features

- Easy use of OTB applications from python
- Simplify common sophisticated I/O features of OTB
- Lazy execution of in-memory pipelines with OTB streaming mechanism
- Interoperable with popular python libraries (numpy, rasterio)
- Extensible

Documentation hosted at [pyotb.readthedocs.io](https://pyotb.readthedocs.io/).

## Example

Building a simple pipeline with OTB applications

```py
import pyotb

# RigidTransformResample application, with input parameters as dict
resampled = pyotb.RigidTransformResample({
    "in": "https://some.remote.data/input.tif",  # Note: no /vsicurl/...
    "interpolator": "linear", 
    "transform.type.id.scaley": 0.5,
    "transform.type.id.scalex": 0.5
})

# OpticalCalibration, with automatic input parameters resolution
calib = pyotb.OpticalCalibration(resampled)

# BandMath, with input parameters passed as kwargs
ndvi = pyotb.BandMath(calib, exp="ndvi(im1b1, im1b4)")

# Pythonic slicing using lazy computation (no memory used)
roi = ndvi[20:586, 9:572]

# Pipeline execution
# The actual computation happens here !
roi.write("output.tif", "float")
```

pyotb's objects also enable easy interoperability with 
[numpy](https://numpy.org/) and [rasterio](https://rasterio.readthedocs.io/):

```python
# Numpy and RasterIO style attributes
print(roi.shape, roi.dtype, roi.transform)
print(roi.metadata)

# Other useful information
print(roi.get_infos())
print(roi.get_statistics())

array = roi.to_numpy()
array, profile = roi.to_rasterio()
```

## Contributing

Contributions are welcome on [Github](https://github.com/orfeotoolbox/pyotb) or the source repository hosted on the Orfeo ToolBox [GitLab](https://gitlab.orfeo-toolbox.org/nicolasnn/pyotb).
