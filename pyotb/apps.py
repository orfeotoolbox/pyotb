import sys
import subprocess
from pyotb.core import App, logger

"""
This is to enable aliases of Apps, i.e. using apps like `pyotb.AppName(...)` instead of `pyotb.App('AppName', ...)`
"""

AVAILABLE_APPLICATIONS = None
# Currently there is an incompatibility between OTBTF and Tensorflow that causes segfault when OTB is used in a script
# where tensorflow has been imported.
# Thus, we run this piece of code in a clean independent `subprocess` that doesn't interact with Tensorflow
if sys.executable:
    try:
        p = subprocess.run([sys.executable, '-c', 'import otbApplication; '
                                                  'print(otbApplication.Registry.GetAvailableApplications())'],
                           capture_output=True)
        AVAILABLE_APPLICATIONS = eval(p.stdout.decode().strip())
    except Exception as e:
        logger.warning('Failed to get the list of applications in an independent process. Trying to get it inside'
                       'the script scope')

# In case the previous has failed, we try the "normal" way to get the list of applications
if not AVAILABLE_APPLICATIONS:
    import otbApplication
    AVAILABLE_APPLICATIONS = otbApplication.Registry.GetAvailableApplications()

# This is to enable aliases of Apps, i.e. using apps like `pyotb.AppName(...)` instead of `pyotb.App('AppName', ...)`
if AVAILABLE_APPLICATIONS:
    for app_name in AVAILABLE_APPLICATIONS:
        exec(f"""def {app_name}(*args, **kwargs): return App('{app_name}', *args, **kwargs)""")
