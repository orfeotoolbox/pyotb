# Troubleshooting

## Migration from pyotb 1.5.4 (oct 2022) to 2.x.y

- `otbObject` has ben renamed `OTBObject`
- use `pyotb_app['paramname']` or `pyotb_app.app.GetParameterValue('paramname')` instead of `pyotb_app.GetParameterValue('paramname')` to access parameter `paramname` value
- use `pyotb_app['paramname']` instead of `pyotb_app.paramname` to access parameter `paramname` value
- `App.output_param` has been replaced with `App.output_image_key`
- `App.write()` argument `filename_extension` has been renamed `ext_fname`
- `Output.__init__()` arguments `app` and `output_parameter_key` have been renamed `pyotb_app` and `param_key`
- `Output.pyotb_app` has been renamed `Output.parent_pyotb_app`

## Known limitations with old versions

!!! note

    All defects described below have been fixed since OTB 8.1.2 and pyotb 2.0.0

### Failure of intermediate writing (otb < 8.1, pyotb < 1.5.4)

When chaining applications in-memory, there may be some problems when writing 
intermediate results, depending on the order
the writings are requested. Some examples can be found below:

#### Example of failures involving slicing

For some applications (non-exhaustive know list: OpticalCalibration, 
DynamicConvert, BandMath), we can face unexpected failures when using channels 
slicing

```python
import pyotb

inp = pyotb.DynamicConvert('raster.tif')
one_band = inp[:, :, 1]

# this works
one_band.write('one_band.tif')

# this works
one_band.write('one_band.tif')
inp.write('stretched.tif')

# this does not work
inp.write('stretched.tif')
one_band.write('one_band.tif')  # Failure here
```

When writing is triggered right after the application declaration, no problem occurs:

```python
import pyotb

inp = pyotb.DynamicConvert('raster.tif')
inp.write('stretched.tif')

one_band = inp[:, :, 1]
one_band.write('one_band.tif')  # no failure
```

Also, when using only spatial slicing, no issue has been reported:

```python
import pyotb

inp = pyotb.DynamicConvert('raster.tif')
one_band = inp[:100, :100, :]

# this works 
inp.write('stretched.tif')
one_band.write('one_band.tif')
```

#### Example of failures involving arithmetic operation

One can meet errors when using arithmetic operations at the end of a pipeline 
when DynamicConvert, BandMath or OpticalCalibration is involved:

```python
import pyotb

inp = pyotb.DynamicConvert('raster.tif')
inp_new = pyotb.ManageNoData(inp, mode='changevalue')
absolute = abs(inp)

# this does not work 
inp.write('one_band.tif')
inp_new.write('one_band_nodata.tif')  # Failure here
absolute.write('absolute.tif')  # Failure here
```

When writing only the final result, i.e. the end of the pipeline (`absolute.write('absolute.tif')`), there is no problem.
