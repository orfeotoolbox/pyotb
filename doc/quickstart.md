## Quickstart: running an OTB application with pyotb

pyotb has been written so that it is more convenient to run an application in 
Python.

You can pass the parameters of an application as a dictionary :

```python
import pyotb
resampled = pyotb.RigidTransformResample({
    'in': 'my_image.tif', 
    'transform.type.id.scaley': 0.5,
    'interpolator': 'linear', 
    'transform.type.id.scalex': 0.5
})
```

Note that pyotb has a 'lazy' evaluation: it only performs operation when it is 
needed, i.e. results are written to disk.
Thus, the previous line doesn't trigger the application.

To actually trigger the application execution, you need to write the result to 
disk:

```python
resampled.write('output.tif')  # this is when the application actually runs
```

### Using Python keyword arguments

It is also possible to use the Python keyword arguments notation for passing 
the parameters:

```python
output = pyotb.SuperImpose(inr='reference_image.tif', inm='image.tif')
```

is equivalent to:

```python
output = pyotb.SuperImpose({'inr': 'reference_image.tif', 'inm': 'image.tif'})
```

Limitations : for this notation, python doesn't accept the parameter `in` or 
any parameter that contains a dots (e.g. `io.in)`.
For `in` and other main input parameters of an OTB app, you may simply pass  
the value as first argument, pyotb will guess the parameter name.  
For parameters that contains dots, you can either use a dictionary, or  replace dots (`.`) with underscores (`_`) as follow :  

```python
resampled = pyotb.RigidTransformResample(
    'my_image.tif', 
    interpolator = 'linear',
    transform_type_id_scaley = 0.5, 
    transform_type_id_scalex = 0.5
)

## In-memory connections

The big asset of pyotb is the ease of in-memory connections between apps.

Let's start from our previous example. Consider the case where one wants to 
apply optical calibration and binary morphological dilatation 
following the undersampling.

Using pyotb, you can pass the output of an app as input of another app :

```python
import pyotb

resampled = pyotb.RigidTransformResample({
    'in': 'my_image.tif', 
    'interpolator': 'linear',
    'transform.type.id.scaley': 0.5, 
    'transform.type.id.scalex': 0.5
})

calibrated = pyotb.OpticalCalibration({
    'in': resampled, 
    'level': 'toa'
}) 

dilated = pyotb.BinaryMorphologicalOperation({
    'in': calibrated, 
    'out': 'output.tif', 
    'filter': 'dilate',
    'structype': 'ball', 
    'xradius': 3, 'yradius': 3
})

dilated.write('result.tif')
```

## Writing the result of an app

Any pyotb object can be written to disk using the `write` method, e.g. :

```python
import pyotb

resampled = pyotb.RigidTransformResample({
    'in': 'my_image.tif', 
    'interpolator': 'linear',
    'transform.type.id.scaley': 0.5,
    'transform.type.id.scalex': 0.5
})

# Here you can set optionally pixel type and extended filename variables
resampled.write(
    {'out': 'output.tif'}, 
    pixel_type='uint16', 
    ext_fname='?nodata=65535'
)
```

Another possibility for writing results is to set the output parameter when 
initializing the application:

```python
import pyotb

resampled = pyotb.RigidTransformResample({
    'in': 'my_image.tif', 
    'interpolator': 'linear', 
    'out': 'output.tif',
    'transform.type.id.scaley': 0.5,
    'transform.type.id.scalex': 0.5
})
```