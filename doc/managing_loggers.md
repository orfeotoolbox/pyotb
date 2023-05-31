## Managing loggers

Several environment variables are used in order to adjust logger level and behaviour. It should be set before importing pyotb.  

- `OTB_LOGGER_LEVEL` : used to set the default OTB logger level.
- `PYOTB_LOGGER_LEVEL` : used to set the pyotb logger level. if not set, `OTB_LOGGER_LEVEL` will be used.

If none of those two variables is set, the logger level will be set to 'INFO'.  
Available levels are : DEBUG, INFO, WARNING, ERROR, CRITICAL  

You may also change the logger level after import (for pyotb only) with the function `set_logger_level`.

```python
import pyotb
pyotb.set_logger_level('DEBUG')
```

Bonus : in some cases, yo may want to silence the GDAL driver logger (for example you will see a lot of errors when reading GML files with OGR).  
One useful trick is to redirect these logs to a file. This can be done using the variable `CPL_LOG`.

## Named applications in logs

It is possible to change an app name in order to track it easily in the logs :  

```python
import os
os.environ['PYOTB_LOGGER_LEVEL'] = 'DEBUG'
import pyotb

bm = pyotb.BandMath(['image.tif'], exp='im1b1 * 100')
bm.name = 'CustomBandMathApp'
bm.execute()
```

```text
2022-06-14 14:22:38 (DEBUG) [pyOTB] CustomBandMathApp: run execute() with parameters={'exp': 'im1b1 * 100', 'il': ['/home/vidlb/Téléchargements/test_4b.tif']}
2022-06-14 14:22:38 (INFO) BandMath: Image #1 has 4 components
2022-06-14 14:22:38 (DEBUG) [pyOTB] CustomBandMathApp: execution succeeded
```
