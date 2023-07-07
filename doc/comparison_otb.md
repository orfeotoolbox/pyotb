## Comparison between otbApplication and pyotb

### Single application execution

<table>
<tr>
<th> OTB </th>
<th> pyotb </th>
</tr>
<tr>
<td>

```python
import otbApplication as otb

input_path = 'my_image.tif'
app = otb.Registry.CreateApplication(
    'RigidTransformResample'
)
app.SetParameterString(
    'in', input_path
)
app.SetParameterString(
    'interpolator', 'linear'
)
app.SetParameterFloat(
    'transform.type.id.scalex', 0.5
)
app.SetParameterFloat(
    'transform.type.id.scaley', 0.5
)
app.SetParameterString(
    'out', 'output.tif'
)
app.SetParameterOutputImagePixelType(
    'out', otb.ImagePixelType_uint16
)

app.ExecuteAndWriteOutput()
```

</td>
<td>

```python
import pyotb

app = pyotb.RigidTransformResample({
    'in': 'my_image.tif', 
    'interpolator': 'linear',
    'transform.type.id.scaley': 0.5,
    'transform.type.id.scalex': 0.5
})

app.write(
    'output.tif', 
    pixel_type='uint16'
)
```

</td>
</tr>
</table>

### In-memory connections

<table>
<tr>
<th> OTB </th>
<th> pyotb </th>
</tr>
<tr>
<td>

```python
import otbApplication as otb

app1 = otb.Registry.CreateApplication(
    'RigidTransformResample'
)
app1.SetParameterString(
    'in', 'my_image.tif'
)
app1.SetParameterString(
    'interpolator', 'linear'
)
app1.SetParameterFloat(
    'transform.type.id.scalex', 0.5
)
app1.SetParameterFloat(
    'transform.type.id.scaley', 0.5
)
app1.Execute()

app2 = otb.Registry.CreateApplication(
    'OpticalCalibration'
)
app2.ConnectImage('in', app1, 'out')
app2.SetParameterString('level', 'toa')
app2.Execute()

app3 = otb.Registry.CreateApplication(
    'BinaryMorphologicalOperation'
)
app3.ConnectImage(
    'in', app2, 'out'
)
app3.SetParameterString(
    'filter', 'dilate'
)
app3.SetParameterString(
    'structype', 'ball'
)
app3.SetParameterInt(
    'xradius', 3
)
app3.SetParameterInt(
    'yradius', 3
)
app3.SetParameterString(
    'out', 'output.tif'
)
app3.SetParameterOutputImagePixelType(
    'out', otb.ImagePixelType_uint16
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

Every pyotb object supports arithmetic operations, such as addition, 
subtraction, comparison...
Consider an example where we want to perform the arithmetic operation 
`image1 * image2 - 2*image3`.

<table>
<tr>
<th> OTB </th>
<th> pyotb </th>
</tr>
<tr>
<td>

```python
import otbApplication as otb

bmx = otb.Registry.CreateApplication(
    'BandMathX'
)
bmx.SetParameterStringList(
    'il', 
    ['im1.tif', 'im2.tif', 'im3.tif']
)
exp = ('im1b1*im2b1-2*im3b1; '
       'im1b2*im2b2-2*im3b2; '
       'im1b3*im2b3-2*im3b3')
bmx.SetParameterString('exp', exp)
bmx.SetParameterString(
    'out', 
    'output.tif'
)
bmx.SetParameterOutputImagePixelType(
    'out', 
    otb.ImagePixelType_uint8
)
bmx.ExecuteAndWriteOutput()
```

Note: code limited to 3-bands images.

</td>
<td>

```python
import pyotb

# filepaths --> pyotb objects
in1 = pyotb.Input('im1.tif')
in2 = pyotb.Input('im2.tif')
in3 = pyotb.Input('im3.tif')

res = in1 * in2 - 2 * in3
res.write(
    'output.tif', 
    pixel_type='uint8'
)
```

Note: works with any number of bands.

</td>
</tr>
</table>

### Slicing

<table>
<tr>
<th> OTB </th>
<th> pyotb </th>
</tr>
<tr>
<td>


```python
import otbApplication as otb

# first 3 channels
app = otb.Registry.CreateApplication(
    'ExtractROI'
)
app.SetParameterString(
    'in', 'my_image.tif'
)
app.SetParameterStringList(
    'cl', 
    ['Channel1', 'Channel2', 'Channel3']
)
app.Execute()

# 1000x1000 roi
app = otb.Registry.CreateApplication(
    'ExtractROI'
)
app.SetParameterString(
    'in', 'my_image.tif'
)
app.SetParameterString(
    'mode', 'extent'
)
app.SetParameterString(
    'mode.extent.unit', 'pxl'
)
app.SetParameterFloat(
    'mode.extent.ulx', 0
)
app.SetParameterFloat(
    'mode.extent.uly', 0
)
app.SetParameterFloat(
    'mode.extent.lrx', 999
)
app.SetParameterFloat(
    'mode.extent.lry', 999
)
app.Execute()
```

</td>
<td>

```python
import pyotb

# filepath --> pyotb object
inp = pyotb.Input('my_image.tif')

extracted = inp[:, :, :3]  # Bands 1,2,3
extracted = inp[:1000, :1000]  # ROI
```

</td>
</tr>
</table>