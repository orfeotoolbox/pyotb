### Process raw Pleiades data
This is a common case of Pleiades data preprocessing : optical calibration -> orthorectification -> pansharpening

```python
import pyotb
srtm = '/media/data/raster/nasa/srtm_30m'
geoid = '/media/data/geoid/egm96.grd'

pan =  pyotb.OpticalCalibration(
    'IMG_PHR1A_P_001/DIM_PHR1A_P_201509011347379_SEN_1791374101-001.XML', 
    level='toa'
)
ms = pyotb.OpticalCalibration(
    'IMG_PHR1A_MS_002/DIM_PHR1A_MS_201509011347379_SEN_1791374101-002.XML', 
    level='toa'
)

pan_ortho = pyotb.OrthoRectification({
    'io.in': pan, 
    'elev.dem': srtm, 
    'elev.geoid': geoid
})
ms_ortho = pyotb.OrthoRectification({
    'io.in': ms, 
    'elev.dem': srtm, 
    'elev.geoid': geoid
})

pxs = pyotb.BundleToPerfectSensor(
    inp=pan_ortho, 
    inxs=ms_ortho, 
    method='bayes', 
    mode='default'
)

exfn = '?gdal:co:COMPRESS=DEFLATE&gdal:co:PREDICTOR=2&gdal:co:BIGTIFF=YES'
# Here we trigger every app in the pipeline and the process is blocked until result is written to disk
pxs.write('pxs_image.tif', pixel_type='uint16', ext_fname=exfn)
```
