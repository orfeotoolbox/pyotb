## Export to Numpy

pyotb objects can be exported to numpy array.

```python
import pyotb
import numpy as np

calibrated = pyotb.OpticalCalibration('image.tif', level='toa')  # this is a pyotb object
arr = np.asarray(calibrated)  # same as calibrated.to_numpy()
```

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
