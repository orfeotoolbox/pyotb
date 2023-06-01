# -*- coding: utf-8 -*-
"""Search for OTB (set env if necessary), subclass core.App for each available application."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import otbApplication as otb  # pylint: disable=import-error

from .core import App
from .helpers import logger


def get_available_applications(as_subprocess: bool = False) -> list[str]:
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
        env["PYTHONPATH"] += ":" + str(Path(otb.__file__).parent)
        env[
            "OTB_LOGGER_LEVEL"
        ] = "CRITICAL"  # in order to suppress warnings while listing applications
        pycmd = "import otbApplication; print(otbApplication.Registry.GetAvailableApplications())"
        cmd_args = [sys.executable, "-c", pycmd]
        try:
            params = {"env": env, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
            with subprocess.Popen(cmd_args, **params) as process:
                logger.debug("Exec \"%s '%s'\"", " ".join(cmd_args[:-1]), pycmd)
                stdout, stderr = process.communicate()
                stdout, stderr = stdout.decode(), stderr.decode()
                # ast.literal_eval is secure and will raise more handy Exceptions than eval
                from ast import literal_eval  # pylint: disable=import-outside-toplevel

                app_list = literal_eval(stdout.strip())
                assert isinstance(app_list, (tuple, list))
        except subprocess.SubprocessError:
            logger.debug("Failed to call subprocess")
        except (ValueError, SyntaxError, AssertionError):
            logger.debug(
                "Failed to decode output or convert to tuple:\nstdout=%s\nstderr=%s",
                stdout,
                stderr,
            )
        if not app_list:
            logger.debug(
                "Failed to list applications in an independent process. Falling back to local python import"
            )
    # Find applications using the normal way
    if not app_list:
        app_list = otb.Registry.GetAvailableApplications()
    if not app_list:
        raise SystemExit(
            "Unable to load applications. Set env variable OTB_APPLICATION_PATH and try again."
        )
    logger.info("Successfully loaded %s OTB applications", len(app_list))
    return app_list


class OTBTFApp(App):
    """Helper for OTBTF."""

    @staticmethod
    def set_nb_sources(*args, n_sources: int = None):
        """Set the number of sources of TensorflowModelServe. Can be either user-defined or deduced from the args.

        Args:
            *args: arguments (dict). NB: we don't need kwargs because it cannot contain source#.il
            n_sources: number of sources. Default is None (resolves the number of sources based on the
                       content of the dict passed in args, where some 'source' str is found)

        """
        if n_sources:
            os.environ["OTB_TF_NSOURCES"] = str(int(n_sources))
        else:
            # Retrieving the number of `source#.il` parameters
            params_dic = {
                k: v for arg in args if isinstance(arg, dict) for k, v in arg.items()
            }
            n_sources = len(
                [k for k in params_dic if "source" in k and k.endswith(".il")]
            )
            if n_sources >= 1:
                os.environ["OTB_TF_NSOURCES"] = str(n_sources)

    def __init__(self, name: str, *args, n_sources: int = None, **kwargs):
        """Constructor for an OTBTFApp object.

        Args:
            name: name of the OTBTF app
            *args: arguments (dict). NB: we don't need kwargs because it cannot contain source#.il
            n_sources: number of sources. Default is None (resolves the number of sources based on the
                       content of the dict passed in args, where some 'source' str is found)
            **kwargs: kwargs

        """
        self.set_nb_sources(*args, n_sources=n_sources)
        super().__init__(name, *args, **kwargs)


AVAILABLE_APPLICATIONS = get_available_applications(as_subprocess=True)

# This is to enable aliases of Apps, i.e. using apps like `pyotb.AppName(...)` instead of `pyotb.App("AppName", ...)`
_CODE_TEMPLATE = (
    """
class {name}(App):
    """
    """
    def __init__(self, *args, **kwargs):
        super().__init__('{name}', *args, **kwargs)
"""
)

for _app in AVAILABLE_APPLICATIONS:
    # Customize the behavior for some OTBTF applications. `OTB_TF_NSOURCES` is now handled by pyotb
    if _app in ("PatchesExtraction", "TensorflowModelTrain", "TensorflowModelServe"):
        exec(  # pylint: disable=exec-used
            _CODE_TEMPLATE.format(name=_app).replace("(App)", "(OTBTFApp)")
        )
    # Default behavior for any OTB application
    else:
        exec(_CODE_TEMPLATE.format(name=_app))  # pylint: disable=exec-used
