# pyotb: a pythonic extension of OTB

Full documentation is available at [pyotb.readthedocs.io](https://pyotb.readthedocs.io/)

[![Latest Release](https://gitlab.orfeo-toolbox.org/nicolasnn/pyotb/-/badges/release.svg)](https://gitlab.orfeo-toolbox.org/nicolasnn/pyotb/-/releases)
[![pipeline status](https://gitlab.orfeo-toolbox.org/nicolasnn/pyotb/badges/master/pipeline.svg)](https://gitlab.orfeo-toolbox.org/nicolasnn/pyotb/-/commits/master)


## Installation
Requirements:
- python>=3.5 and numpy
- OrfeoToolBox python API

```bash
pip install pyotb --upgrade
```

For Python>=3.6, latest version available is pyotb 1.5.1 For Python 3.5, latest version available is pyotb 1.2.2

## Quickstart: running an OTB application as a oneliner
pyotb has been written so that it is more convenient to run an application in Python.

You can pass the parameters of an application as a dictionary :
```python
import pyotb
resampled = pyotb.RigidTransformResample({'in': 'my_image.tif', 'interpolator': 'linear',
                                          'transform.type.id.scaley': 0.5, 'transform.type.id.scalex': 0.5})
```
Note that pyotb has a 'lazy' evaluation: it only performs operation when it is needed, i.e. results are written to disk.
Thus, the previous line doesn't trigger the application.

To actually trigger the application execution, you need to write the result to disk:

```python
resampled.write('output.tif')  # this is when the application actually runs
```

## Using Python keyword arguments
It is also possible to use the Python keyword arguments notation for passing the parameters:
```python
output = pyotb.SuperImpose(inr='reference_image.tif', inm='image.tif')
```
is equivalent to:
```python
output = pyotb.SuperImpose({'inr': 'reference_image.tif', 'inm': 'image.tif'})
```

Limitations : for this notation, python doesn't accept the parameter `in` or any parameter that contains a `.`. E.g., it is not possible to use `pyotb.RigidTransformResample(in=input_path...)` or `pyotb.VectorDataExtractROI(io.vd=vector_path...)`.




## In-memory connections
The big asset of pyotb is the ease of in-memory connections between apps.

Let's start from our previous example. Consider the case where one wants to apply optical calibration and binary morphological dilatation 
following the undersampling.

Using pyotb, you can pass the output of an app as input of another app :
```python
import pyotb

resampled = pyotb.RigidTransformResample({'in': 'my_image.tif', 'interpolator': 'linear', 
                                          'transform.type.id.scaley': 0.5, 'transform.type.id.scalex': 0.5})

calibrated = pyotb.OpticalCalibration({'in': resampled, 'level': 'toa'}) 

dilated = pyotb.BinaryMorphologicalOperation({'in': calibrated, 'out': 'output.tif', 'filter': 'dilate', 
                                              'structype': 'ball', 'xradius': 3, 'yradius': 3})
dilated.write('result.tif')
```

## Writing the result of an app
Any pyotb object can be written to disk using the `write` method, e.g. :

```python
import pyotb

resampled = pyotb.RigidTransformResample({'in': 'my_image.tif', 'interpolator': 'linear',
                                          'transform.type.id.scaley': 0.5, 'transform.type.id.scalex': 0.5})
# Here you can set optionally pixel type and extended filename variables
resampled.write({'out': 'output.tif'}, pixel_type='uint16', filename_extension='?nodata=65535')
```

Another possibility for writing results is to set the output parameter when initializing the application:
```python
import pyotb

resampled = pyotb.RigidTransformResample({'in': 'my_image.tif', 'interpolator': 'linear', 'out': 'output.tif',
                                          'transform.type.id.scaley': 0.5, 'transform.type.id.scalex': 0.5})
# Here you can set optionally pixel type and extended filename variables
resampled.write(pixel_type='uint16', filename_extension='?nodata=65535')
```

## Arithmetic operations
Every pyotb object supports arithmetic operations, such as addition, subtraction, comparison...
Consider an example where we want to compute a vegeteation mask from NDVI, i.e. the arithmetic operation `(nir - red) / (nir + red) > 0.3`

With pyotb, one can simply do :
```python
import pyotb

# transforming filepaths to pyotb objects
nir, red = pyotb.Input('nir.tif'), pyotb.Input('red.tif')

res = (nir - red) / (nir + red) > 0.3
print(res.exp)  # prints the BandMath expression: "((im1b1 - im2b1) / (im1b1 + im2b1)) > 0.3 ? 1 : 0"
res.write('vegetation_mask.tif', pixel_type='uint8')
```

## Slicing
pyotb objects support slicing in a Python fashion :

```python
import pyotb

# transforming filepath to pyotb object
inp = pyotb.Input('my_image.tif')

inp[:, :, :3]  # selecting first 3 bands
inp[:, :, [0, 1, 4]]  # selecting bands 1, 2 & 5
inp[:1000, :1000]  # selecting 1000x1000 subset, same as inp[:1000, :1000, :] 
inp[:100, :100].write('my_image_roi.tif')  # write cropped image to disk
```

## Numpy-inspired functions
Some functions have been written, entirely based on OTB, to mimic the behavior of some well-known numpy functions. 
### pyotb.where
Equivalent of `numpy.where`.
It is the equivalent of the muparser syntax `condition ? x : y` that can be used in OTB's BandMath.

```python
import pyotb

# transforming filepaths to pyotb objects
labels, image1, image2 = pyotb.Input('labels.tif'), pyotb.Input('image1.tif') , pyotb.Input('image2.tif')

# If labels = 1, returns image1. Else, returns image2 
res = pyotb.where(labels == 1, image1, image2)  # this would also work: pyotb.where(labels == 1, 'image1.tif', 'image2.tif') 

# A more complex example
# If labels = 1, returns image1. If labels = 2, returns image2. If labels = 3, returns 3. Else 0
res = pyotb.where(labels == 1, image1,
                  pyotb.where(labels == 2, image2,
                              pyotb.where(labels == 3, 3, 0)))

```

### pyotb.clip
Equivalent of `numpy.clip`. Clip (limit) the values in a raster to a range.

```python
import pyotb

res = pyotb.clip('my_image.tif', 0, 255)  # clips the values between 0 and 255
```

### pyotb.all
Equivalent of `numpy.all`. 

For only one image, this function checks that all bands of the image are True (i.e. !=0) and outputs
a singleband boolean raster.
For several images, this function checks that all images are True (i.e. !=0) and outputs
a boolean raster, with as many bands as the inputs.


### pyotb.any
Equivalent of `numpy.any`. 

For only one image, this function checks that at least one band of the image is True (i.e. !=0) and outputs
a singleband boolean raster.
For several images, this function checks that at least one of the images is True (i.e. !=0) and outputs
a boolean raster, with as many bands as the inputs.


## Interaction with Numpy

pyotb objects can be transparently used in numpy functions.

For example:

```python
import pyotb
import numpy as np

inp = pyotb.Input('image.tif')  # this is a pyotb object

# Creating a numpy array of noise
white_noise = np.random.normal(0, 50, size=inp.shape)  # this is a numpy object

# Adding the noise to the image
noisy_image = inp + white_noise  # magic: this is a pyotb object that has the same georeference as input. 
                                 # `np.add(inp, white_noise)` would have worked the same
noisy_image.write('image_plus_noise.tif')
```
Limitations : 
- The whole image is loaded into memory
- The georeference can not be modified. Thus, numpy operations can not change the image or pixel size


## Export to rasterio
pyotb objects can also be exported in a format that is usable by rasterio.

For example:

```python
import pyotb
import rasterio
from scipy import ndimage

# Pansharpening + NDVI + creating bare soils mask
pxs = pyotb.BundleToPerfectSensor(inp='panchromatic.tif', inxs='multispectral.tif')
ndvi = pyotb.RadiometricIndices({'in': pxs, 'channels.red': 3, 'channels.nir': 4, 'list': 'Vegetation:NDVI'})
bare_soil_mask = (ndvi < 0.3)

# Exporting the result as array & profile usable by rasterio
mask_array, profile = bare_soil_mask.to_rasterio()

# Doing something in Python that is not possible with OTB, e.g. gathering the contiguous groups of pixels
# with an integer index
labeled_mask_array, nb_groups = ndimage.label(mask_array)

# Writing the result to disk
with rasterio.open('labeled_bare_soil.tif', 'w', **profile) as f:
    f.write(labeled_mask_array)

```
This way of exporting pyotb objects is more flexible that exporting to numpy, as the user gets the `profile` dictionary. 
If the georeference or pixel size is modified, the user can update the `profile` accordingly.


## Interaction with Tensorflow

We saw that numpy operations had some limitations. To bypass those limitations, it is possible to use some Tensorflow operations on pyotb objects.


You need a working installation of OTBTF >=3.0 for this and then the code is like this:

```python
import pyotb

def scalar_product(x1, x2):
    """This is a function composed of tensorflow operations."""
    import tensorflow as tf
    return tf.reduce_sum(tf.multiply(x1, x2), axis=-1)

# Compute the scalar product
res = pyotb.run_tf_function(scalar_product)('image1.tif', 'image2.tif')  # magic: this is a pyotb object
res.write('scalar_product.tif')
```

For some easy syntax, one can use `pyotb.run_tf_function` as a function decorator, such as:
```python
import pyotb

@pyotb.run_tf_function  # The decorator enables the use of pyotb objects as inputs/output of the function
def scalar_product(x1, x2):
    import tensorflow as tf
    return tf.reduce_sum(tf.multiply(x1, x2), axis=-1)

res = scalar_product('image1.tif', 'image2.tif')  # magic: this is a pyotb object
```

Advantages :
- The process supports streaming, hence the whole image is **not** loaded into memory
- Can be integrated in OTB pipelines

Limitations :
- It is not possible to use the tensorflow python API inside a script where OTBTF is used because of compilation issues 
between Tensorflow and OTBTF, i.e. `import tensorflow` doesn't work in a script where OTBTF apps have been initialized


## Some examples
### Compute the mean of several rasters, taking into account NoData
Let's consider we have at disposal 73 NDVI rasters for a year, where clouds have been masked with NoData (nodata value of -10 000 for example).

Goal: compute the mean across time (keeping the spatial dimension) of the NDVIs, excluding cloudy pixels. Piece of code to achieve that:

```python
import pyotb

nodata = -10000
ndvis = [pyotb.Input(path) for path in ndvi_paths]

# For each pixel location, summing all valid NDVI values 
summed = sum([pyotb.where(ndvi != nodata, ndvi, 0) for ndvi in ndvis])

# Printing the generated BandMath expression
print(summed.exp)  # this returns a very long exp: "0 + ((im1b1 != -10000) ? im1b1 : 0) + ((im2b1 != -10000) ? im2b1 : 0) + ... + ((im73b1 != -10000) ? im73b1 : 0)"

# For each pixel location, getting the count of valid pixels
count = sum([pyotb.where(ndvi == nodata, 0, 1) for ndvi in ndvis])

mean = summed / count  # BandMath exp of this is very long: "(0 + ((im1b1 != -10000) ? im1b1 : 0) + ... + ((im73b1 != -10000) ? im73b1 : 0)) / (0 + ((im1b1 == -10000) ? 0 : 1) + ... + ((im73b1 == -10000) ? 0 : 1))"
mean.write('ndvi_annual_mean.tif')
```

Note that no actual computation is executed before the last line where the result is written to disk.

### Process raw Pleiades data
This is a common case of Pleiades data preprocessing : optical calibration -> orthorectification -> pansharpening

```python
import pyotb
srtm = '/media/data/raster/nasa/srtm_30m'
geoid = '/media/data/geoid/egm96.grd'

pan =  pyotb.OpticalCalibration('IMG_PHR1A_P_001/DIM_PHR1A_P_201509011347379_SEN_1791374101-001.XML', level='toa')
ms = pyotb.OpticalCalibration('IMG_PHR1A_MS_002/DIM_PHR1A_MS_201509011347379_SEN_1791374101-002.XML', level='toa')

pan_ortho = pyotb.OrthoRectification({'io.in': pan, 'elev.dem': srtm, 'elev.geoid': geoid})
ms_ortho = pyotb.OrthoRectification({'io.in': ms, 'elev.dem': srtm, 'elev.geoid': geoid})

pxs = pyotb.BundleToPerfectSensor(inp=pan_ortho, inxs=ms_ortho, method='bayes', mode="default")

# Here we trigger every app in the pipeline and the process is blocked until result is written to disk
pxs.write('pxs_image.tif', pixel_type='uint16', filename_extension='?gdal:co:COMPRESS=DEFLATE&gdal:co:PREDICTOR=2')
```
