# -*- coding: utf-8 -*-
"""Search for OTB (set env if necessary), subclass core.App for each available application."""
import os
import sys
from pathlib import Path

import otbApplication as otb
from .helpers import logger
from .core import OTBObject


def get_available_applications(as_subprocess=False):
    """Find available OTB applications.

    Args:
        as_subprocess: indicate if function should list available applications using subprocess call

    Returns:
        tuple of available applications

    """
    app_list = ()
    if as_subprocess and sys.executable:
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
            import subprocess  # pylint: disable=import-outside-toplevel
            params = {"env": env, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
            with subprocess.Popen(cmd_args, **params) as p:
                logger.debug('Exec "%s \'%s\'"', ' '.join(cmd_args[:-1]), pycmd)
                stdout, stderr = p.communicate()
                stdout, stderr = stdout.decode(), stderr.decode()
                # ast.literal_eval is secure and will raise more handy Exceptions than eval
                from ast import literal_eval  # pylint: disable=import-outside-toplevel
                app_list = literal_eval(stdout.strip())
                assert isinstance(app_list, (tuple, list))
        except subprocess.SubprocessError:
            logger.debug("Failed to call subprocess")
        except (ValueError, SyntaxError, AssertionError):
            logger.debug("Failed to decode output or convert to tuple:\nstdout=%s\nstderr=%s", stdout, stderr)

        if not app_list:
            logger.info("Failed to list applications in an independent process. Falling back to local python import")
    # Find applications using the normal way
    if not app_list:
        app_list = otb.Registry.GetAvailableApplications()
    if not app_list:
        raise SystemExit("Unable to load applications. Set env variable OTB_APPLICATION_PATH and try again.")

    logger.info("Successfully loaded %s OTB applications", len(app_list))
    return app_list


class App(OTBObject):
    """Base class for UI related functions, will be subclassed using app name as class name, see CODE_TEMPLATE."""
    _name = ""

    def __init__(self, *args, **kwargs):
        """Default App constructor, adds UI specific attributes and functions."""
        super().__init__(*args, **kwargs)
        self.description = self.app.GetDocLongDescription()

    @property
    def name(self):
        """Application name that will be printed in logs.

        Returns:
            user's defined name or appname

        """
        return self._name or self.appname

    @name.setter
    def name(self, name):
        """Set custom name.

        Args:
          name: new name

        """
        if isinstance(name, str):
            self._name = name
        else:
            raise TypeError(f"{self.name}: bad type ({type(name)}) for application name, only str is allowed")

    @property
    def outputs(self):
        """List of application outputs."""
        return [getattr(self, key) for key in self.out_param_keys if key in self.parameters]

    def find_outputs(self):
        """Find output files on disk using path found in parameters.

        Returns:
            list of files found on disk

        """
        files = []
        missing = []
        for out in self.outputs:
            dest = files if out.exists() else missing
            dest.append(str(out.filepath.absolute()))
        for filename in missing:
            logger.error("%s: execution seems to have failed, %s does not exist", self.name, filename)

        return tuple(files)


class OTBTFApp(App):
    """Helper for OTBTF."""
    @staticmethod
    def set_nb_sources(*args, n_sources=None):
        """Set the number of sources of TensorflowModelServe. Can be either user-defined or deduced from the args.

        Args:
            *args: arguments (dict). NB: we don't need kwargs because it cannot contain source#.il
            n_sources: number of sources. Default is None (resolves the number of sources based on the
                       content of the dict passed in args, where some 'source' str is found)

        """
        if n_sources:
            os.environ['OTB_TF_NSOURCES'] = str(int(n_sources))
        else:
            # Retrieving the number of `source#.il` parameters
            params_dic = {k: v for arg in args if isinstance(arg, dict) for k, v in arg.items()}
            n_sources = len([k for k in params_dic if 'source' in k and k.endswith('.il')])
            if n_sources >= 1:
                os.environ['OTB_TF_NSOURCES'] = str(n_sources)

    def __init__(self, app_name, *args, n_sources=None, **kwargs):
        """Constructor for an OTBTFApp object.

        Args:
            app_name: name of the OTBTF app
            *args: arguments (dict). NB: we don't need kwargs because it cannot contain source#.il
            n_sources: number of sources. Default is None (resolves the number of sources based on the
                       content of the dict passed in args, where some 'source' str is found)
            **kwargs: kwargs

        """
        self.set_nb_sources(*args, n_sources=n_sources)
        super().__init__(app_name, *args, **kwargs)


AVAILABLE_APPLICATIONS = get_available_applications(as_subprocess=True)

# This is to enable aliases of Apps, i.e. using apps like `pyotb.AppName(...)` instead of `pyotb.App("AppName", ...)`
_CODE_TEMPLATE = """
class {name}(App):
    """ """
    def __init__(self, *args, **kwargs):
        super().__init__('{name}', *args, **kwargs)
"""

for _app in AVAILABLE_APPLICATIONS:
    # Customize the behavior for some OTBTF applications. `OTB_TF_NSOURCES` is now handled by pyotb
    if _app in ("PatchesExtraction", "TensorflowModelTrain", "TensorflowModelServe"):
        exec(_CODE_TEMPLATE.format(name=_app).replace("(App)", "(OTBTFApp)"))  # pylint: disable=exec-used
    # Default behavior for any OTB application
    else:
        exec(_CODE_TEMPLATE.format(name=_app))  # pylint: disable=exec-used
