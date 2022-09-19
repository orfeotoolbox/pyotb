# -*- coding: utf-8 -*-
"""This module provides some helpers to properly initialize pyotb."""
import os
import sys
import logging
from pathlib import Path
from shutil import which


# Allow user to switch between OTB directories without setting every env variable
OTB_ROOT = os.environ.get("OTB_ROOT")

# Logging
# User can also get logger with `logging.getLogger("pyOTB")`
# then use pyotb.set_logger_level() to adjust logger verbosity
logger = logging.getLogger("pyOTB")
logger_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(fmt="%(asctime)s (%(levelname)-4s) [pyOTB] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger_handler.setFormatter(formatter)
# Search for PYOTB_LOGGER_LEVEL, else use OTB_LOGGER_LEVEL as pyOTB level, or fallback to INFO
LOG_LEVEL = os.environ.get("PYOTB_LOGGER_LEVEL") or os.environ.get("OTB_LOGGER_LEVEL") or "INFO"
logger.setLevel(getattr(logging, LOG_LEVEL))
# Here it would be possible to use a different level for a specific handler
# A more verbose one can go to text file while print only errors to stdout
logger_handler.setLevel(getattr(logging, LOG_LEVEL))
logger.addHandler(logger_handler)


def set_logger_level(level):
    """Allow user to change the current logging level.

    Args:
        level: logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """
    logger.setLevel(getattr(logging, level))
    logger_handler.setLevel(getattr(logging, level))


def find_otb(prefix=OTB_ROOT, scan=True, scan_userdir=True):
    """Try to load OTB bindings or scan system, help user in case of failure, set env variables.

    Path precedence :                                OTB_ROOT > python bindings directory
        OR search for releases installations    :    HOME
        OR (for linux)                          :    /opt/otbtf > /opt/otb > /usr/local > /usr
        OR (for windows)                        :    C:/Program Files

    Args:
        prefix: prefix to search OTB in (Default value = OTB_ROOT)
        scan: find otb in system known locations (Default value = True)
        scan_userdir: search for OTB release in user's home directory (Default value = True)

    Returns:
        otbApplication module

    """
    otb = None
    # Try OTB_ROOT env variable first (allow override default OTB version)
    if prefix:
        try:
            set_environment(prefix)
            import otbApplication as otb  # pylint: disable=import-outside-toplevel
            return otb
        except EnvironmentError as e:
            raise SystemExit(f"Failed to import OTB with prefix={prefix}") from e
        except ImportError as e:
            __suggest_fix_import(str(e), prefix)
            raise SystemExit("Failed to import OTB. Exiting.") from e
    # Else try import from actual Python path
    try:
        # Here, we can't properly set env variables before OTB import. We assume user did this before running python
        # For LD_LIBRARY_PATH problems, use OTB_ROOT instead of PYTHONPATH
        import otbApplication as otb  # pylint: disable=import-outside-toplevel
        if "OTB_APPLICATION_PATH" not in os.environ:
            lib_dir = __find_lib(otb_module=otb)
            apps_path = __find_apps_path(lib_dir)
            otb.Registry.SetApplicationPath(apps_path)
        return otb
    except ImportError as e:
        PYTHONPATH = os.environ.get("PYTHONPATH")
        if not scan:
            raise SystemExit(f"Failed to import OTB with env PYTHONPATH={PYTHONPATH}") from e
    # Else search system
    logger.info("Failed to import OTB. Searching for it...")
    prefix = __find_otb_root(scan_userdir)
    # Try to import one last time before raising error
    try:
        set_environment(prefix)
        import otbApplication as otb  # pylint: disable=import-outside-toplevel
        return otb
    except EnvironmentError as e:
        raise SystemExit("Auto setup for OTB env failed. Exiting.") from e
    # Unknown error
    except ModuleNotFoundError as e:
        raise SystemExit("Can't run without OTB installed. Exiting.") from e
    # Help user to fix this
    except ImportError as e:
        __suggest_fix_import(str(e), prefix)
        raise SystemExit("Failed to import OTB. Exiting.") from e


def set_environment(prefix):
    """Set environment variables (before OTB import), raise error if anything is wrong.

    Args:
        prefix: path to OTB root directory

    """
    logger.info("Preparing environment for OTB in %s", prefix)
    # OTB root directory
    prefix = Path(prefix)
    if not prefix.exists():
        raise FileNotFoundError(str(prefix))
    built_from_source = False
    if not (prefix / 'README').exists():
        built_from_source = True
    # External libraries
    lib_dir = __find_lib(prefix)
    if not lib_dir:
        raise EnvironmentError("Can't find OTB external libraries")
    # This does not seems to work
    if sys.platform == "linux" and built_from_source:
        new_ld_path = f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH') or ''}"
        os.environ["LD_LIBRARY_PATH"] = new_ld_path
    # Add python bindings directory first in PYTHONPATH
    otb_api = __find_python_api(lib_dir)
    if not otb_api:
        raise EnvironmentError("Can't find OTB Python API")
    if otb_api not in sys.path:
        sys.path.insert(0, otb_api)
    # Add /bin first in PATH, in order to avoid conflicts with another GDAL install when using os.system()
    os.environ["PATH"] = f"{prefix / 'bin'}{os.pathsep}{os.environ['PATH']}"
    # Applications path  (this can be tricky since OTB import will succeed even without apps)
    apps_path = __find_apps_path(lib_dir)
    if Path(apps_path).exists():
        os.environ["OTB_APPLICATION_PATH"] = apps_path
    else:
        raise EnvironmentError("Can't find OTB applications directory")

    os.environ["LC_NUMERIC"] = "C"
    os.environ["GDAL_DRIVER_PATH"] = "disable"
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
        raise EnvironmentError(f"Can't find GDAL location with current OTB prefix '{prefix}' or in /usr")

    os.environ["GDAL_DATA"] = gdal_data
    os.environ["PROJ_LIB"] = proj_lib


def __find_lib(prefix=None, otb_module=None):
    """Try to find OTB external libraries directory.

    Args:
        prefix: try with OTB root directory
        otb_module: try with OTB python module (otbApplication) library path if found, else None

    Returns:
        lib path

    """
    if prefix is not None:
        lib_dir = prefix / 'lib'
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


def __find_python_api(lib_dir):
    """Try to find the python path.

    Args:
        prefix: prefix

    Returns:
        python API path if found, else None

    """
    otb_api = lib_dir / "python"
    if not otb_api.exists():
        otb_api = lib_dir / "otb/python"
    if otb_api.exists():
        return str(otb_api.absolute())
    logger.debug("Failed to find OTB python bindings directory")
    return None


def __find_apps_path(lib_dir):
    """Try to find the OTB applications path.

    Args:
        lib_dir: library path

    Returns:
        application path if found, else empty string

    """
    if lib_dir.exists():
        otb_application_path = lib_dir / "otb/applications"
        if otb_application_path.exists():
            return str(otb_application_path.absolute())
        # This should not happen, may be with failed builds ?
        logger.error("Library directory found but 'applications' is missing")
    return ""


def __find_otb_root(scan_userdir=False):
    """Search for OTB root directory in well known locations.

    Args:
        scan_userdir: search with glob in $HOME directory

    Returns:
        str path of the OTB directory

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
            prefix = prefix.absolute()
    elif sys.platform == "win32":
        for path in Path("c:/Program Files").glob("**/OTB-*/lib"):
            logger.info("Found %s", path.parent)
            prefix = path.parent.absolute()
    elif sys.platform == "darwin":
        # TODO: find OTB in macOS
        pass

    # If possible, use OTB found in user's HOME tree (this may take some time)
    if scan_userdir:
        for path in Path().home().glob("**/OTB-*/lib"):
            logger.info("Found %s", path.parent)
            prefix = path.parent.absolute()

    return prefix


def __suggest_fix_import(error_message, prefix):
    """Help user to fix the OTB installation with appropriate log messages."""
    logger.critical("An error occurred while importing OTB Python API")
    logger.critical("OTB error message was '%s'", error_message)
    if sys.platform == "linux":
        if error_message.startswith('libpython3.'):
            logger.critical("It seems like you need to symlink or recompile python bindings")
            if sys.executable.startswith('/usr/bin'):
                lib = f"/usr/lib/x86_64-linux-gnu/libpython3.{sys.version_info.minor}.so"
                if which('ctest'):
                    logger.critical("To recompile python bindings, use 'cd %s ; source otbenv.profile ; "
                                    "ctest -S share/otb/swig/build_wrapping.cmake -VV'", prefix)
                elif Path(lib).exists():
                    expect_minor = int(error_message[11])
                    if expect_minor != sys.version_info.minor:
                        logger.critical("Python library version mismatch (OTB was expecting 3.%s) : "
                                        "a simple symlink may not work, depending on your python version", expect_minor)
                    target_lib = f"{prefix}/lib/libpython3.{expect_minor}.so.rh-python3{expect_minor}-1.0"
                    logger.critical("Use 'ln -s %s %s'", lib, target_lib)
                else:
                    logger.critical("You may need to install cmake in order to recompile python bindings")
            else:
                logger.critical("Unable to automatically locate python dynamic library of %s", sys.executable)
            return
    elif sys.platform == "win32":
        if error_message.startswith("DLL load failed"):
            if sys.version_info.minor != 7:
                logger.critical("You need Python 3.5 (OTB releases 6.4 to 7.4) or Python 3.7 (since OTB 8)")
                issue_link = "https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/issues/2010"
                logger.critical("Another workaround is to recompile Python bindings with cmake, see %s", issue_link)
            else:
                logger.critical("It seems that your env variables aren't properly set,"
                                " first use 'call otbenv.bat' then try to import pyotb once again")
            return
    docs_link = "https://www.orfeo-toolbox.org/CookBook/Installation.html"
    logger.critical("You can verify installation requirements for your OS at %s", docs_link)
