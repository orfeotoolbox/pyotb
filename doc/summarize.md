## Summarize applications

pyotb enables to summarize applications as a dictionary with keys/values for 
parameters. This feature can be used to keep track of a process, composed of 
multiple applications chained together.

### Single application

Let's take the example of one single application.

```python
import pyotb

app = pyotb.RigidTransformResample({
    'in': 'my_image.tif', 
    'interpolator': 'linear',
    'transform.type.id.scaley': 0.5,
    'transform.type.id.scalex': 0.5
})
```

The application can be summarized using `pyotb.summarize()` or 
`app.summary()`, which are equivalent.

```python
print(app.summarize())
```

Results in the following (lines have been pretty printed for the sake of 
documentation):

```json lines
{
  'name': 'RigidTransformResample', 
  'parameters': {
    'transform.type': 'id', 
    'in': 'my_image.tif', 
    'interpolator': 'linear', 
    'transform.type.id.scaley': 0.5, 
    'transform.type.id.scalex': 0.5
  }
}
```

Note that we can also summarize an application after it has been executed:

```python
app.write('output.tif', pixel_type='uint16')
print(app.summarize())
```

Which results in the following:

```json lines
{
  'name': 'RigidTransformResample', 
  'parameters': {
    'transform.type': 'id',
    'in': 'my_image.tif', 
    'interpolator': 'linear', 
    'transform.type.id.scaley': 0.5, 
    'transform.type.id.scalex': 0.5, 
    'out': 'output.tif'
  }
}
```

Now `'output.tif'` has been added to the application parameters.

### Multiple applications chained together (pipeline)

When multiple applications are chained together, the summary of the last 
application will describe all upstream processes.

```python
import pyotb

app1 = pyotb.RigidTransformResample({
    'in': 'my_image.tif', 
    'interpolator': 'linear',
    'transform.type.id.scaley': 0.5,
    'transform.type.id.scalex': 0.5
})
app2 = pyotb.Smoothing(app1)
print(app2.summarize())
```

Results in:

```json lines
{
  'name': 'Smoothing', 
  'parameters': {
    'type': 'anidif', 
    'type.anidif.timestep': 0.125, 
    'type.anidif.nbiter': 10, 
    'type.anidif.conductance': 1.0, 
    'in': {
      'name': 'RigidTransformResample', 
      'parameters': {
        'transform.type': 'id', 
        'in': 'my_image.tif', 
        'interpolator': 'linear', 
        'transform.type.id.scaley': 0.5, 
        'transform.type.id.scalex': 0.5
      }
    }
  }
}
```

### Remote files URL stripping

Cloud-based raster URLs often include tokens or random strings resulting from 
the URL signing.
Those can be removed from the summarized paths, using the `strip_inpath` 
and/or `strip_outpath` arguments respectively for inputs and/or outputs.

Here is an example with Microsoft Planetary Computer:

```python
import planetary_computer
import pyotb

url = (
    "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/31/N/EA/2023/"
    "11/03/S2A_MSIL2A_20231103T095151_N0509_R079_T31NEA_20231103T161409.SAFE/"
    "GRANULE/L2A_T31NEA_A043691_20231103T100626/IMG_DATA/R10m/T31NEA_20231103"
    "T095151_B02_10m.tif"
)
signed_url = planetary_computer.sign_inplace(url)
app = pyotb.Smoothing(signed_url)
```

By default, the summary does not strip the URL.

```python
print(app.summarize()["parameters"]["in"])
```

This results in:

```
/vsicurl/https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/31/N/EA/...
2023/11/03/S2A_MSIL2A_20231103T095151_N0509_R079_T31NEA_20231103T161409.SAFE...
/GRANULE/L2A_T31NEA_A043691_20231103T100626/IMG_DATA/R10m/T31NEA_20231103T...
095151_B02_10m.tif?st=2023-11-07T15%3A52%3A47Z&se=2023-11-08T16%3A37%3A47Z&...
sp=rl&sv=2021-06-08&sr=c&skoid=c85c15d6-d1ae-42d4-af60-e2ca0f81359b&sktid=...
72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2023-11-08T11%3A41%3A41Z&ske=2023-...
11-15T11%3A41%3A41Z&sks=b&skv=2021-06-08&sig=xxxxxxxxxxx...xxxxx
```

Now we can strip the URL to keep only the resource identifier and get rid of 
the token:

```python
print(app.summarize(strip_inpath=True)["parameters"]["in"])
```

Which now results in:

```
/vsicurl/https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/31/N/EA/...
2023/11/03/S2A_MSIL2A_20231103T095151_N0509_R079_T31NEA_20231103T161409.SAFE...
/GRANULE/L2A_T31NEA_A043691_20231103T100626/IMG_DATA/R10m/T31NEA_20231103T...
095151_B02_10m.tif
```