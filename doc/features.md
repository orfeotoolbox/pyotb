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

### Metadata

Images metadata can be retrieved with the `metadata` attribute:

```python
print(inp.metadata)
```

Gives: 

```
{
  'DataType': 1.0, 
  'DriverLongName': 'GeoTIFF', 
  'DriverShortName': 'GTiff', 
  'GeoTransform': (760056.0, 6.0, 0.0, 6946092.0, 0.0, -6.0),
  'LowerLeftCorner': (760056.0, 6944268.0), 
  'LowerRightCorner': (761562.0, 6944268.0), 
  'AREA_OR_POINT': 'Area', 
  'TIFFTAG_SOFTWARE': 'CSinG - 13 SEPTEMBRE 2012', 
  'ProjectionRef': 'PROJCS["RGF93 v1 / Lambert-93",\n...',
  'ResolutionFactor': 0, 
  'SubDatasetIndex': 0, 
  'UpperLeftCorner': (760056.0, 6946092.0), 
  'UpperRightCorner': (761562.0, 6946092.0), 
  'TileHintX': 251.0, 
  'TileHintY': 8.0
}
```

## Information

The information fetched by the `ReadImageInfo` OTB application is available 
through `get_info()`:

```python
print(inp.get_info())
```

Gives:

```json lines
{
  'indexx': 0, 
  'indexy': 0, 
  'sizex': 251, 
  'sizey': 304, 
  'spacingx': 6.0, 
  'spacingy': -6.0, 
  'originx': 760059.0, 
  'originy': 6946089.0, 
  'estimatedgroundspacingx': 5.978403091430664, 
  'estimatedgroundspacingy': 5.996793270111084, 
  'numberbands': 4, 
  'datatype': 'unsigned_char', 
  'ullat': 0.0, 
  'ullon': 0.0, 
  'urlat': 0.0, 
  'urlon': 0.0, 
  'lrlat': 0.0, 
  'lrlon': 0.0, 
  'lllat': 0.0, 
  'lllon': 0.0, 
  'rgb.r': 0, 
  'rgb.g': 1, 
  'rgb.b': 2, 
  'projectionref': 'PROJCS["RGF93 v1 ..."EPSG","2154"]]',
  'gcp.count': 0
}
```

## Statistics

Image statistics can be computed on-the-fly using `get_statistics()`:

```python
print(inp.get_statistics())
```

Gives:

```json lines
{
  'out.mean': [79.5505, 109.225, 115.456, 249.349], 
  'out.min': [33, 64, 91, 47], 
  'out.max': [255, 255, 230, 255], 
  'out.std': [51.0754, 35.3152, 23.4514, 20.3827]
}
```