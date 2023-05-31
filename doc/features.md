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

## Shape attributes

You can access the shape of any in-memory pyotb object.

```python
import pyotb

# transforming filepath to pyotb object
inp = pyotb.Input('my_image.tif')
print(inp[:1000, :500].shape)  # (1000, 500, 4)
```
