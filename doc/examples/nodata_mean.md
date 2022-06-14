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

