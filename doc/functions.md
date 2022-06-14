Some functions have been written, entirely based on OTB, to mimic the behavior of some well-known numpy functions.
## pyotb.where
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

## pyotb.clip
Equivalent of `numpy.clip`. Clip (limit) the values in a raster to a range.

```python
import pyotb

res = pyotb.clip('my_image.tif', 0, 255)  # clips the values between 0 and 255
```

## pyotb.all
Equivalent of `numpy.all`.

For only one image, this function checks that all bands of the image are True (i.e. !=0) and outputs
a singleband boolean raster.
For several images, this function checks that all images are True (i.e. !=0) and outputs
a boolean raster, with as many bands as the inputs.


## pyotb.any
Equivalent of `numpy.any`.

For only one image, this function checks that at least one band of the image is True (i.e. !=0) and outputs
a singleband boolean raster.
For several images, this function checks that at least one of the images is True (i.e. !=0) and outputs
a boolean raster, with as many bands as the inputs.