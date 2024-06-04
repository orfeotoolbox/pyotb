## Managing loggers

Several environment variables are used in order to adjust logger level and 
behaviour. It should be set before importing pyotb.  

- `OTB_LOGGER_LEVEL` : used to set the default OTB logger level.
- `PYOTB_LOGGER_LEVEL` : used to set the pyotb logger level.

If `PYOTB_LOGGER_LEVEL` isn't set, `OTB_LOGGER_LEVEL` will be used.  
If none of those two variables is set, the logger level will be set to 'INFO'.  
Available levels are : DEBUG, INFO, WARNING, ERROR, CRITICAL  

You may also change the logger level after import (for pyotb only) 
using pyotb.logger.setLevel(level).

```python
import pyotb
pyotb.logger.setLevel('DEBUG')
```

Bonus : in some cases, you may want to silence the GDAL driver logger 
(for example you will see a lot of errors when reading GML files with OGR).  
One useful trick is to redirect these logs to a file. This can be done using 
the variable `CPL_LOG`.

## Log to file
It is possible to change the behaviour of the default pyotb logger as follow

```py
import logging
import pyotb
# Optional : remove default stdout handler (but OTB will still print its own log)
pyotb.logger.handlers.pop()
# Add file handler
handler = logging.FileHandler("/my/log/file.log")
handler.setLevel("DEBUG")
pyotb.logger.addHandler(handler)
```

For more advanced configuration and to manage conflicts between several loggers, 
see the [logging module docs](https://docs.python.org/3/howto/logging-cookbook.html) 
and use the `dictConfig()` function to configure your own logger.  

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
