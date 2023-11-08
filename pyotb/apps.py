"""Search for OTB (set env if necessary), subclass core.App for each available application."""
from __future__ import annotations

import os

import otbApplication as otb  # pylint: disable=import-error

from .core import App
from .helpers import logger


def get_available_applications() -> tuple[str]:
    """Find available OTB applications.

    Returns:
        tuple of available applications

    Raises:
        SystemExit: if no application is found

    """
    app_list = otb.Registry.GetAvailableApplications()
    if app_list:
        logger.info("Successfully loaded %s OTB applications", len(app_list))
        return app_list
    raise SystemExit(
        "Unable to load applications. Set env variable OTB_APPLICATION_PATH and try again."
    )


class OTBTFApp(App):
    """Helper for OTBTF to ensure the nb_sources variable is set."""

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
            n_sources: number of sources. Default is None (resolves the number of sources based on the
                       content of the dict passed in args, where some 'source' str is found)

        """
        self.set_nb_sources(*args, n_sources=n_sources)
        super().__init__(name, *args, **kwargs)


AVAILABLE_APPLICATIONS = get_available_applications()

# This is to enable aliases of Apps, i.e. `pyotb.AppName(...)` instead of `pyotb.App("AppName", ...)`
_CODE_TEMPLATE = """
class {name}(App):
    def __init__(self, *args, **kwargs):
        super().__init__('{name}', *args, **kwargs)
"""

for _app in AVAILABLE_APPLICATIONS:
    # Customize the behavior for some OTBTF applications. `OTB_TF_NSOURCES` is now handled by pyotb
    if _app in ("PatchesExtraction", "TensorflowModelTrain", "TensorflowModelServe"):
        exec(  # pylint: disable=exec-used
            _CODE_TEMPLATE.format(name=_app).replace("(App)", "(OTBTFApp)")
        )
    # Default behavior for any OTB application
    else:
        exec(_CODE_TEMPLATE.format(name=_app))  # pylint: disable=exec-used
