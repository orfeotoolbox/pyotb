# -*- coding: utf-8 -*-
"""
Search for OTB (set env if necessary), subclass core.App for each available application
"""
import os
import sys
from pathlib import Path

from .tools import logger, find_otb, set_gdal_vars

# Will first call `import otbApplication`, trying to workaround any ImportError
OTB_ROOT, OTB_APPLICATION_PATH = find_otb()

if not OTB_ROOT:
    raise EnvironmentError("Can't run without OTB installed.")

set_gdal_vars(OTB_ROOT)

# Should not raise ImportError since it was tested in find_otb()
import otbApplication as otb


def get_available_applications(as_subprocess=False):
    """
    Find available OTB applications
    :param as_subprocess: indicate if function should list available applications using subprocess call
    :returns: tuple of available applications
    """
    app_list = ()
    if as_subprocess and sys.executable and hasattr(sys, 'ps1'):
        # Currently, there is an incompatibility between OTBTF and Tensorflow that causes segfault
        # when OTBTF apps are used in a script where tensorflow has already been imported.
        # See https://github.com/remicres/otbtf/issues/28
        # Thus, we run this piece of code in a clean independent `subprocess` that doesn't interact with Tensorflow
        env = os.environ.copy()
        if "PYTHONPATH" not in env:
            env["PYTHONPATH"] = ""
        env["PYTHONPATH"] = ":" + str(Path(otb.__file__).parent)
        env["OTB_LOGGER_LEVEL"] = "CRITICAL"  # in order to suppress warnings while listing applications
        pycmd = "import otbApplication; print(otbApplication.Registry.GetAvailableApplications())"
        cmd_args = [sys.executable, "-c", pycmd]
        try:
            import subprocess
            params = {"env": env, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
            with subprocess.Popen(cmd_args, **params) as p:
                logger.debug("%s %s", ' '.join(cmd_args[:-1]), pycmd)
                stdout, stderr = p.communicate()
                stdout, stderr = stdout.decode(), stderr.decode()
                # ast.literal_eval is secure and will raise more handy Exceptions than eval
                from ast import literal_eval
                app_list = literal_eval(stdout.strip())
                assert isinstance(app_list, (tuple, list))

        except subprocess.SubprocessError:
            logger.debug("Failed to call subprocess")
        except (ValueError, SyntaxError, AssertionError):
            logger.debug("Failed to decode output or convert to tuple:\nstdout=%s\nstderr=%s", stdout, stderr)
        if not app_list:
            logger.info("Failed to list applications in an independent process. Falling back to local otb import")

    if not app_list:
        app_list = otb.Registry.GetAvailableApplications()
    if not app_list:
        logger.warning("Unable to load applications. Set env variable OTB_APPLICATION_PATH then try again")
        return ()

    logger.info("Successfully loaded %s OTB applications", len(app_list))
    return app_list


if OTB_APPLICATION_PATH:
    otb.Registry.SetApplicationPath(OTB_APPLICATION_PATH)
    os.environ["OTB_APPLICATION_PATH"] = OTB_APPLICATION_PATH

AVAILABLE_APPLICATIONS = get_available_applications(as_subprocess=True)

# First core.py call (within __init__ scope)
from .core import App

# This is to enable aliases of Apps, i.e. using apps like `pyotb.AppName(...)` instead of `pyotb.App("AppName", ...)`
_code_template = """
class {name}(App):
    def __init__(self, *args, **kwargs):
        super().__init__('{name}', *args, **kwargs)
"""
for _app in AVAILABLE_APPLICATIONS:
    # Default behavior for any OTB application
    exec(_code_template.format(name=_app))  # pylint: disable=exec-used

    # Customize the behavior for TensorflowModelServe application. The user doesn't need to set the env variable
    # `OTB_TF_NSOURCES`, it is handled in pyotb
    if _app == 'TensorflowModelServe':
        class TensorflowModelServe(App):
            """
            Helper for OTBTF
            """
            @staticmethod
            def set_nb_sources(*args, n_sources=None):
                """
                Set the number of sources of TensorflowModelServe. Can be either user-defined or deduced from the args
                :param args: arguments
                :param n_sources: number of sources. Default is None (resolves the number of sources based on the
                content of the dict passed in args, where some 'source' str is found)
                """
                if n_sources:
                    os.environ['OTB_TF_NSOURCES'] = str(int(n_sources))
                else:
                    # Retrieving the number of `source#.il` parameters
                    params_dic = {k: v for arg in args if isinstance(arg, dict) for k, v in arg.items()}
                    n_sources = len([k for k in params_dic if k.startswith('source') and k.endswith('.il')])
                    os.environ['OTB_TF_NSOURCES'] = str(n_sources)

            def __init__(self, *args, n_sources=None, **kwargs):
                """
                :param args: args
                :param n_sources: number of sources. Default is None (resolves the number of sources based on the
                content of the dict passed in args, where some 'source' str is found)
                :param kwargs: kwargs
                """
                self.set_nb_sources(*args, n_sources=n_sources)
                super().__init__('TensorflowModelServe', *args, **kwargs)
