## Comparison between otbApplication and pyotb

### Single application execution

Using OTB, the code would be like :

```python
import otbApplication

input_path = 'my_image.tif'
resampled = otbApplication.Registry.CreateApplication('RigidTransformResample')
resampled.SetParameterString('in', input_path)
resampled.SetParameterString('interpolator', 'linear')
resampled.SetParameterFloat('transform.type.id.scalex', 0.5)
resampled.SetParameterFloat('transform.type.id.scaley', 0.5)
resampled.SetParameterString('out', 'output.tif')
resampled.SetParameterOutputImagePixelType(
    'out', otbApplication.ImagePixelType_uint16
)

resampled.ExecuteAndWriteOutput()
```

Using pyotb:

```python
import pyotb

resampled = pyotb.RigidTransformResample({
    'in': 'my_image.tif', 
    'interpolator': 'linear',
    'transform.type.id.scaley': 0.5,
    'transform.type.id.scalex': 0.5
})

resampled.write('output.tif', pixel_type='uint16')
```

### In-memory connections

<table>
<tr>
<th> OTB </th>
<th> pyotb </th>
</tr>
<tr>
<td>

```python
import otbApplication

app1 = otbApplication.Registry.CreateApplication(
    'RigidTransformResample'
)
app1.SetParameterString('in', 'my_image.tif')
app1.SetParameterString('interpolator', 'linear')
app1.SetParameterFloat(
    'transform.type.id.scalex',
    0.5
)
app1.SetParameterFloat(
    'transform.type.id.scaley',
    0.5
)
app1.Execute()

app2 = otbApplication.Registry.CreateApplication(
    'OpticalCalibration'
)
app2.ConnectImage('in', app1, 'out')
app2.SetParameterString('level', 'toa')
app2.Execute()

app3 = otbApplication.Registry.CreateApplication(
    'BinaryMorphologicalOperation'
)
app3.ConnectImage('in', app2, 'out')
app3.SetParameterString('filter', 'dilate')
app3.SetParameterString('structype', 'ball')
app3.SetParameterInt('xradius', 3)
app3.SetParameterInt('yradius', 3)
app3.SetParameterString('out', 'output.tif')
app3.SetParameterOutputImagePixelType(
    'out', 
    otbApplication.ImagePixelType_uint16
)
app3.ExecuteAndWriteOutput()
```

</td>
<td>

```python
import pyotb

app1 = pyotb.RigidTransformResample({
    'in': 'my_image.tif', 
    'interpolator': 'linear',
    'transform.type.id.scaley': 0.5, 
    'transform.type.id.scalex': 0.5
})

app2 = pyotb.OpticalCalibration({
    'in': app1, 
    'level': 'toa'
}) 

app3 = pyotb.BinaryMorphologicalOperation({
    'in': app2, 
    'out': 'output.tif', 
    'filter': 'dilate',
    'structype': 'ball', 
    'xradius': 3, 
    'yradius': 3
})

app3.write(
    'result.tif', 
    pixel_type='uint16'
)
```

</td>
</tr>
</table>

### Arithmetic operations

Every pyotb object supports arithmetic operations, such as addition, subtraction, comparison...
Consider an example where we want to perform the arithmetic operation `image1 * image2 - 2*image3`

Using OTB, the following code works for 3-bands images :

```python
import otbApplication

bmx = otbApplication.Registry.CreateApplication('BandMathX')
bmx.SetParameterStringList('il', ['image1.tif', 'image2.tif', 'image3.tif'])  # all images are 3-bands
exp = 'im1b1*im2b1 - 2*im3b1; im1b2*im2b2 - 2*im3b2; im1b3*im2b3 - 2*im3b3'
bmx.SetParameterString('exp', exp)
bmx.SetParameterString('out', 'output.tif')
bmx.SetParameterOutputImagePixelType('out', otbApplication.ImagePixelType_uint8)
bmx.ExecuteAndWriteOutput()
```

With pyotb, the following works with images of any number of bands :

```python
import pyotb

# transforming filepaths to pyotb objects
input1, input2, input3 = pyotb.Input('image1.tif'), pyotb.Input('image2.tif') , pyotb.Input('image3.tif')

res = input1 * input2 - 2 * input2
res.write('output.tif', pixel_type='uint8')
```

### Slicing

Using OTB, for selection bands or ROI, the code looks like:

```python
import otbApplication

# selecting first 3 bands
extracted = otbApplication.Registry.CreateApplication('ExtractROI')
extracted.SetParameterString('in', 'my_image.tif')
extracted.SetParameterStringList('cl', ['Channel1', 'Channel2', 'Channel3'])
extracted.Execute()

# selecting 1000x1000 subset
extracted = otbApplication.Registry.CreateApplication('ExtractROI')
extracted.SetParameterString('in', 'my_image.tif')
extracted.SetParameterString('mode', 'extent')
extracted.SetParameterString('mode.extent.unit', 'pxl')
extracted.SetParameterFloat('mode.extent.ulx', 0)
extracted.SetParameterFloat('mode.extent.uly', 0)
extracted.SetParameterFloat('mode.extent.lrx', 999)
extracted.SetParameterFloat('mode.extent.lry', 999)
extracted.Execute()
```

Instead, using pyotb:

```python
import pyotb

# transforming filepath to pyotb object
inp = pyotb.Input('my_image.tif')

extracted = inp[:, :, :3]  # selecting first 3 bands
extracted = inp[:1000, :1000]  # selecting 1000x1000 subset
```
