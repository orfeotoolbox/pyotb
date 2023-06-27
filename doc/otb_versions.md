## System with multiple OTB versions

If you want to quickly switch between OTB versions, or override the default 
system version, you may use the `OTB_ROOT` env variable :

```python
import os
# This is equivalent to "[set/export] OTB_ROOT=/opt/otb" before launching python
os.environ['OTB_ROOT'] = '/opt/otb'
import pyotb
```

```text
2022-06-14 13:59:03 (INFO) [pyOTB] Preparing environment for OTB in /opt/otb
2022-06-14 13:59:04 (INFO) [pyOTB] Successfully loaded 126 OTB applications
```

If you try to import pyotb without having set environment, it will try to find 
any OTB version installed on your system:

```python
import pyotb
```

```text
2022-06-14 13:55:41 (INFO) [pyOTB] Failed to import OTB. Searching for it...
2022-06-14 13:55:41 (INFO) [pyOTB] Found /opt/otb/lib/otb/
2022-06-14 13:55:41 (INFO) [pyOTB] Found /opt/otbtf/lib/otb
2022-06-14 13:55:42 (INFO) [pyOTB] Found /home/otbuser/Applications/OTB-8.0.1-Linux64
2022-06-14 13:55:43 (INFO) [pyOTB] Preparing environment for OTB in /home/otbuser/Applications/OTB-8.0.1-Linux64
2022-06-14 13:55:44 (INFO) [pyOTB] Successfully loaded 117 OTB applications
```

Here is the path precedence for this automatic env configuration :

```text
    OTB_ROOT env variable > python bindings directory
    OR search for releases installations    :    HOME
    OR (for linux)                          :    /opt/otbtf > /opt/otb > /usr/local > /usr
    OR (for windows)                        :    C:/Program Files
```

N.B. :  in case `otbApplication` is found in `PYTHONPATH` (and if `OTB_ROOT` 
was not set), the OTB which the python API is linked to will be used.  

## Fresh OTB installation

If you've just installed OTB binaries in a Linux environment, you may 
encounter an error at first import, pyotb will help you fix it :

```python
import pyotb
```

```text
2022-06-14 14:00:34 (INFO) [pyOTB] Preparing environment for OTB in /home/otbuser/Applications/OTB-8.0.1-Linux64
2022-07-07 16:56:04 (CRITICAL) [pyOTB] An error occurred while importing OTB Python API
2022-07-07 16:56:04 (CRITICAL) [pyOTB] OTB error message was 'libpython3.8.so.rh-python38-1.0: cannot open shared object file: No such file or directory'
2022-07-07 16:56:04 (CRITICAL) [pyOTB] It seems like you need to symlink or recompile python bindings
2022-07-07 16:56:04 (CRITICAL) [pyOTB] Use 'ln -s /usr/lib/x86_64-linux-gnu/libpython3.8.so /home/otbuser/Applications/OTB-8.0.1-Linux64/lib/libpython3.8.so.rh-python38-1.0'

# OR in case Python version is not 3.8 and cmake is installed :
2022-07-07 16:54:34 (CRITICAL) [pyOTB] Python library version mismatch (OTB was expecting 3.8) : a simple symlink may not work, depending on your python version
2022-07-07 16:54:34 (CRITICAL) [pyOTB] To recompile python bindings, use 'cd /home/otbuser/Applications/OTB-8.0.1-Linux64 ; source otbenv.profile ; ctest -S share/otb/swig/build_wrapping.cmake -VV'

Failed to import OTB. Exiting.
```
