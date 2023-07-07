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

For now, `resampled` has not been executed. Indeed, pyotb has a 'lazy' 
evaluation: applications are executed only when required. Generally, like in 
this example, executions happen to write output images to disk. 

To actually trigger the application execution, `write()` has to be called:

```python
resampled.write('output.tif')  # this is when the application actually runs
```

### Using Python keyword arguments

One can use the Python keyword arguments notation for passing that parameters:

```python
output = pyotb.SuperImpose(inr='reference_image.tif', inm='image.tif')
```

Which is equivalent to:

```python
output = pyotb.SuperImpose({'inr': 'reference_image.tif', 'inm': 'image.tif'})
```

!!! warning

    For this notation, python doesn't accept the parameter `in` or any 
    parameter that contains a dots (e.g. `io.in`). For `in` or other main 
    input parameters of an OTB application, you may simply pass the value as 
    first argument, pyotb will guess the parameter name. For parameters that 
    contains dots, you can either use a dictionary, or replace dots (`.`) 
    with underscores (`_`). 

    Let's take the example of the `OrthoRectification` application of OTB, 
    with the input image parameter named `io.in`:

    Option #1, keyword-arg-free:

    ```python
    ortho = pyotb.OrthoRectification('my_image.tif')
    ```
    
    Option #2, replacing dots with underscores in parameter name: 

    ```python
    ortho = pyotb.OrthoRectification(io_in='my_image.tif')
    ``` 

## In-memory connections

One nice feature of pyotb is in-memory connection between apps. It relies on 
the so-called [streaming](https://www.orfeo-toolbox.org/CookBook/C++/StreamingAndThreading.html)
mechanism of OTB, that enables to process huge images with a limited memory 
footprint.

pyotb allows to pass any application's output to another. This enables to 
build pipelines composed of several applications.

Let's start from our previous example. Consider the case where one wants to 
resample the image, then apply optical calibration and binary morphological 
dilatation. We can write the following code to build a pipeline that will
generate the output in an end-to-end fashion, without being limited with the 
input image size, without writing temporary files.

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
    'xradius': 3, 
    'yradius': 3
})
```

We just have built our first pipeline! At this point, it's all symbolic since 
no computation has been performed. To trigger our pipeline, one must call the 
`write()` method from the pipeline termination:

```python
dilated.write('output.tif')
```

In the next section, we will detail how `write()` works. 

## Writing the result of an app

Any pyotb object can be written to disk using `write()`.

Let's consider the following pyotb application instance:

```python
import pyotb
resampled = pyotb.RigidTransformResample({
    'in': 'my_image.tif', 
    'interpolator': 'linear',
    'transform.type.id.scaley': 0.5,
    'transform.type.id.scalex': 0.5
})
```

We can then write the output of `resampled` as following:

```python
resampled.write('output.tif')
```

!!! note

    For applications that have multiple outputs, passing a `dict` of filenames 
    can be considered. Let's take the example of `MeanShiftSmoothing` which 
    has 2 output images:

    ```python
    import pyotb
    meanshift = pyotb.MeanShiftSmoothing('my_image.tif')
    meanshift.write({'fout': 'output_1.tif', 'foutpos': 'output_2.tif'})
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

### Pixel type

Setting the pixel type is optional, and can be achieved setting the 
`pixel_type` argument: 

```python
resampled.write('output.tif', pixel_type='uint16')
```

The value of `pixel_type` corresponds to the name of a pixel type from OTB 
applications (e.g. `'uint8'`, `'float'`, etc).

### Extended filenames

Extended filenames can be passed as `str` or `dict`.

As `str`:

```python
resampled.write(
    ...
    ext_fname='nodata=65535&box=0:0:256:256'
)
```

As `dict`:

```python
resampled.write(
    ...
    ext_fname={'nodata': '65535', 'box': '0:0:256:256'}
)
```

!!! info

    When `ext_fname` is provided and the output filenames contain already some 
    extended filename pattern, the ones provided in the filenames take 
    priority over the ones passed in `ext_fname`. This allows to fine-grained 
    tune extended filenames for each output, with a common extended filenames 
    keys/values basis.
