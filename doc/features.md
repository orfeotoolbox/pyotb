## Arithmetic operations

Every pyotb object supports arithmetic operations, such as addition, 
subtraction, comparison...
Consider an example where we want to compute a vegeteation mask from NDVI, 
i.e. the arithmetic operation `(nir - red) / (nir + red) > 0.3`

With pyotb, one can simply do :

```python
import pyotb

# transforming filepaths to pyotb objects
nir, red = pyotb.Input('nir.tif'), pyotb.Input('red.tif')

res = (nir - red) / (nir + red) > 0.3
# Prints the BandMath expression:
# "((im1b1 - im2b1) / (im1b1 + im2b1)) > 0.3 ? 1 : 0"
print(res.exp)
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
inp[:, :, 1:-1]  # removing first and last band
inp[:, :, ::2]  # selecting one band every 2 bands
inp[:100, :100]  # selecting 100x100 subset, same as inp[:100, :100, :] 
inp[:100, :100].write('my_image_roi.tif')  # write cropped image to disk
```

## Retrieving a pixel location in image coordinates

One can retrieve a pixel location in image coordinates (i.e. row and column 
indices) using `get_rowcol_from_xy()`:

```python
inp.get_rowcol_from_xy(760086.0, 6948092.0)  # (333, 5)
```

## Reading a pixel value

One can read a pixel value of a pyotb object using brackets, as if it was a 
common array. Returned is a list of pixel values for each band:

```python
inp[10, 10]  # [217, 202, 182, 255]
```

!!! warning

    Accessing multiple pixels values if not computationally efficient. Please 
    use this with moderation, or consider numpy or pyotb applications to 
    process efficiently blocks of pixels.

## Attributes

### Shape

The shape of pyotb objects can be retrieved using `shape`.

```python
print(inp[:1000, :500].shape)  # (1000, 500, 4)
```

### Pixel type

The pixel type of pyotb objects can be retrieved using `dtype`.

```python
inp.dtype  # e.g. 'uint8'
```

!!! note

    The `dtype` returns a `str` corresponding to values accepted by the 
    `pixel_type` of `write()`

### Transform

The transform, as defined in GDAL, can be retrieved with the `transform` 
attribute:

```python
inp.transform  # (6.0, 0.0, 760056.0, 0.0, -6.0, 6946092.0)
```