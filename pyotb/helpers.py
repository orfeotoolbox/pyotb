"""This module ensure we properly initialize pyotb, or raise SystemExit in case of broken install."""

import logging
import logging.config
import os
import sys
import sysconfig
from pathlib import Path
from shutil import which

from .install import install_otb, interactive_config

# Allow user to switch between OTB directories without setting every env variable
OTB_ROOT = os.environ.get("OTB_ROOT")
DOCS_URL = "https://www.orfeo-toolbox.org/CookBook/Installation.html"

# Logging
# User can also get logger with `logging.getLogger("pyotb")`
# then use pyotb.set_logger_level() to adjust logger verbosity

# Search for PYOTB_LOGGER_LEVEL, else use OTB_LOGGER_LEVEL as pyotb level, or fallback to INFO
LOG_LEVEL = (
    os.environ.get("PYOTB_LOGGER_LEVEL") or os.environ.get("OTB_LOGGER_LEVEL") or "INFO"
)

logger = logging.getLogger("pyotb")

logging_cfg = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s (%(levelname)-4s) [pyotb] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        }
    },
    "loggers": {"pyotb": {"level": LOG_LEVEL, "handlers": ["stdout"]}},
}
logging.config.dictConfig(logging_cfg)


def find_otb(prefix: str = OTB_ROOT, scan: bool = True):
    """Try to load OTB bindings or scan system, help user in case of failure, set env.

    If in interactive prompt, user will be asked if he wants to install OTB.
    The OTB_ROOT variable allow one to override default OTB version, with auto env setting.
    Path precedence : $OTB_ROOT > location of python bindings location
    Then, if OTB is not found:
        search for releases installations: $HOME/Applications
        OR (for Linux): /opt/otbtf > /opt/otb > /usr/local > /usr
        OR (for Windows): C:/Program Files

    Args:
        prefix: prefix to search OTB in (Default value = OTB_ROOT)
        scan: find otb in system known locations (Default value = True)

    Returns:
        otbApplication module

    Raises:
        SystemError: is OTB is not found (when using interactive mode)
        SystemExit: if OTB is not found, since pyotb won't be usable

    """
    otb = None
    # Try OTB_ROOT env variable first (allow override default OTB version)
    if prefix:
        try:
            set_environment(prefix)
            import otbApplication as otb  # pylint: disable=import-outside-toplevel

            return otb
        except SystemError as e:
            raise SystemExit(f"Failed to import OTB with prefix={prefix}") from e
        except ImportError as e:
            __suggest_fix_import(str(e), prefix)
            raise SystemExit("Failed to import OTB. Exiting.") from e
    # Else try import from actual Python path
    try:
        # Here, we can't properly set env variables before OTB import.
        # We assume user did this before running python
        # For LD_LIBRARY_PATH problems, use OTB_ROOT instead of PYTHONPATH
        import otbApplication as otb  # pylint: disable=import-outside-toplevel

        if "OTB_APPLICATION_PATH" not in os.environ:
            lib_dir = __find_lib(otb_module=otb)
            apps_path = __find_apps_path(lib_dir)
            otb.Registry.SetApplicationPath(apps_path)
        return otb
    except ImportError as e:
        pythonpath = os.environ.get("PYTHONPATH")
        if not scan:
            raise SystemExit(
                f"Failed to import OTB with env PYTHONPATH={pythonpath}"
            ) from e
    # Else search system
    logger.info("Failed to import OTB. Searching for it...")
    prefix = __find_otb_root()
    # Try auto install if shell is interactive
    if not prefix and hasattr(sys, "ps1"):
        if input("OTB is missing. Do you want to install it ? (y/n): ") == "y":
            return find_otb(install_otb(*interactive_config()))
        raise SystemError("OTB libraries not found on disk. ")
    if not prefix:
        raise SystemExit(
            "OTB libraries not found on disk. "
            "To install it, open an interactive python shell and 'import pyotb'"
        )
    # If OTB was found on disk, set env and try to import one last time
    try:
        set_environment(prefix)
        import otbApplication as otb  # pylint: disable=import-outside-toplevel

        return otb
    except SystemError as e:
        raise SystemExit("Auto setup for OTB env failed. Exiting.") from e
    # Help user to fix this
    except ImportError as e:
        __suggest_fix_import(str(e), prefix)
        raise SystemExit("Failed to import OTB. Exiting.") from e


def set_environment(prefix: str):
    """Set environment variables (before OTB import), raise error if anything is wrong.

    Args:
        prefix: path to OTB root directory

    Raises:
        SystemError: if OTB or GDAL is not found

    """
    logger.info("Preparing environment for OTB in %s", prefix)
    # OTB root directory
    prefix = Path(prefix)
    if not prefix.exists():
        raise FileNotFoundError(str(prefix))
    built_from_source = False
    if not (prefix / "README").exists():
        built_from_source = True
    # External libraries
    lib_dir = __find_lib(prefix)
    if not lib_dir:
        raise SystemError("Can't find OTB external libraries")
    # LD library path : this does not seems to work
    if sys.platform == "linux" and built_from_source:
        new_ld_path = f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH') or ''}"
        os.environ["LD_LIBRARY_PATH"] = new_ld_path

    # Add python bindings directory first in PYTHONPATH
    otb_api = __find_python_api(lib_dir)
    if not otb_api:
        raise SystemError("Can't find OTB Python API")
    if otb_api not in sys.path:
        sys.path.insert(0, otb_api)

    # Add /bin first in PATH, in order to avoid conflicts with another GDAL install
    os.environ["PATH"] = f"{prefix / 'bin'}{os.pathsep}{os.environ['PATH']}"
    # Ensure APPLICATION_PATH is set
    apps_path = __find_apps_path(lib_dir)
    if Path(apps_path).exists():
        os.environ["OTB_APPLICATION_PATH"] = apps_path
    else:
        raise SystemError("Can't find OTB applications directory")
    os.environ["LC_NUMERIC"] = "C"
    os.environ["GDAL_DRIVER_PATH"] = "disable"

    # Find GDAL libs
    if (prefix / "share/gdal").exists():
        # Local GDAL (OTB Superbuild, .run, .exe)
        gdal_data = str(prefix / "share/gdal")
        proj_lib = str(prefix / "share/proj")
    elif sys.platform == "linux":
        # If installed using apt or built from source with system deps
        gdal_data = "/usr/share/gdal"
        proj_lib = "/usr/share/proj"
    elif sys.platform == "win32":
        gdal_data = str(prefix / "share/data")
        proj_lib = str(prefix / "share/proj")
    else:
        raise SystemError(
            f"Can't find GDAL location with current OTB prefix '{prefix}' or in /usr"
        )
    os.environ["GDAL_DATA"] = gdal_data
    os.environ["PROJ_LIB"] = proj_lib


def __find_lib(prefix: str = None, otb_module=None):
    """Try to find OTB external libraries directory.

    Args:
        prefix: try with OTB root directory
        otb_module: try with otbApplication library path if found, else None

    Returns:
        lib path, or None if not found

    """
    if prefix is not None:
        lib_dir = prefix / "lib"
        if lib_dir.exists():
            return lib_dir.absolute()
    if otb_module is not None:
        lib_dir = Path(otb_module.__file__).parent.parent
        # Case OTB .run file
        if lib_dir.name == "lib":
            return lib_dir.absolute()
        # Case /usr
        lib_dir = lib_dir.parent
        if lib_dir.name in ("lib", "x86_64-linux-gnu"):
            return lib_dir.absolute()
        # Case built from source (/usr/local, /opt/otb, ...)
        lib_dir = lib_dir.parent
        if lib_dir.name == "lib":
            return lib_dir.absolute()
    return None


def __find_python_api(lib_dir: Path):
    """Try to find the python path.

    Args:
        prefix: prefix

    Returns:
        OTB python API path, or None if not found

    """
    otb_api = lib_dir / "python"
    if not otb_api.exists():
        otb_api = lib_dir / "otb/python"
    if otb_api.exists():
        return str(otb_api.absolute())
    logger.debug("Failed to find OTB python bindings directory")
    return None


def __find_apps_path(lib_dir: Path):
    """Try to find the OTB applications path.

    Args:
        lib_dir: library path

    Returns:
        application path, or empty string if not found

    """
    if lib_dir.exists():
        otb_application_path = lib_dir / "otb/applications"
        if otb_application_path.exists():
            return str(otb_application_path.absolute())
        # This should not happen, may be with failed builds ?
        logger.error("Library directory found but 'applications' is missing")
    return ""


def __find_otb_root():
    """Search for OTB root directory in well known locations.

    Returns:
        str path of the OTB directory, or None if not found

    """
    prefix = None
    # Search possible known locations (system scpecific)
    if sys.platform == "linux":
        possible_locations = (
            "/usr/lib/x86_64-linux-gnu/otb",
            "/usr/local/lib/otb/",
            "/opt/otb/lib/otb/",
            "/opt/otbtf/lib/otb",
        )
        for str_path in possible_locations:
            path = Path(str_path)
            if not path.exists():
                continue
            logger.info("Found %s", str_path)
            if path.parent.name == "x86_64-linux-gnu":
                prefix = path.parent.parent.parent
            else:
                prefix = path.parent.parent
    elif sys.platform == "win32":
        for path in sorted(Path("c:/Program Files").glob("**/OTB-*/lib")):
            logger.info("Found %s", path.parent)
            prefix = path.parent
    # Search for pyotb OTB install, or default on macOS
    apps = Path.home() / "Applications"
    for path in sorted(apps.glob("OTB-*/lib/")):
        logger.info("Found %s", path.parent)
        prefix = path.parent
    # Return latest found prefix (and version), see precedence in find_otb() docstrings
    if isinstance(prefix, Path):
        return prefix.absolute()
    return None


def __suggest_fix_import(error_message: str, prefix: str):
    """Help user to fix the OTB installation with appropriate log messages."""
    logger.critical("An error occurred while importing OTB Python API")
    logger.critical("OTB error message was '%s'", error_message)
    if sys.platform == "win32":
        if error_message.startswith("DLL load failed"):
            if sys.version_info.minor != 7:
                logger.critical(
                    "You need Python 3.5 (OTB 6.4 to 7.4) or Python 3.7 (since OTB 8)"
                )
            else:
                logger.critical(
                    "It seems that your env variables aren't properly set,"
                    " first use 'call otbenv.bat' then try to import pyotb once again"
                )
    elif error_message.startswith("libpython3."):
        logger.critical(
            "It seems like you need to symlink or recompile python bindings"
        )
        if (
            sys.executable.startswith("/usr/bin")
            and which("ctest")
            and which("python3-config")
        ):
            logger.critical(
                "To compile, use 'cd %s ; source otbenv.profile ; "
                "ctest -S share/otb/swig/build_wrapping.cmake -VV'",
                prefix,
            )
            return
        logger.critical(
            "You may need to install cmake, python3-dev and mesa's libgl"
            " in order to recompile python bindings"
        )
        expected = int(error_message[11])
        if expected != sys.version_info.minor:
            logger.critical(
                "Python library version mismatch (OTB expected 3.%s) : "
                "a symlink may not work, depending on your python version",
                expected,
            )
        lib_dir = sysconfig.get_config_var("LIBDIR")
        lib = f"{lib_dir}/libpython3.{sys.version_info.minor}.so"
        if Path(lib).exists():
            target = f"{prefix}/lib/libpython3.{expected}.so.1.0"
            logger.critical("If using OTB>=8.0, try 'ln -sf %s %s'", lib, target)
    logger.critical(
        "You can verify installation requirements for your OS at %s", DOCS_URL
    )


# This part of pyotb is the first imported during __init__ and checks if OTB is found
# If OTB isn't found, a SystemExit is raised to prevent execution of the core module
find_otb()
