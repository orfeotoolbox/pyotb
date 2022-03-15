# -*- coding: utf-8 -*-
"""
This module provides some helpers to properly initialize pyotb
"""
import os
import sys
import logging
from pathlib import Path

# This will prevent erasing user config
if "logger" not in globals():
    logging.basicConfig(
        format="%(asctime)s (%(levelname)-4s) [pyOTB] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
logger = logging.getLogger()


def find_otb(root="", scan=True, scan_userdir=True):
    """
    Try to load OTB bindings or scan system, help user in case of failure, return env variables
    Precedence :                                    OTB_ROOT > python bindings directory
        OR search for releases installations    :    current directory > home directory
        OR (for linux)                          :    /opt/otbtf > /opt/otb > /usr/local > /usr
    :param root: prefix to search OTB in
    :param scan: find otb in system known locations
    :param scan_userdir: search for OTB release in user's home directory
    :return:   (OTB_ROOT, OTB_APPLICATIONS_PATH) environment variables
    """

    # OTB_ROOT env variable is first (allow override default OTB version)
    prefix = root or os.environ.get("OTB_ROOT") or os.environ.get("OTB_DIR")
    if prefix:
        logger.info("Found OTB Python API in %s", prefix)
        # Add path first in PYTHONPATH
        otb_api = get_python_api(prefix)
        sys.path.insert(0, str(otb_api))
        if "otb" not in globals():
            try:
                import otbApplication
                lib_dir = get_lib(otbApplication)
                apps_path = get_apps_path(lib_dir)
                prefix = lib_dir.parent
                if prefix.name == 'lib':
                    prefix = prefix.parent
                return str(prefix), str(apps_path)

            except ImportError:
                logging.critical("Can't import OTB with OTB_ROOT=%s", prefix)
                return None, None
        else:
            logger.warning("Can't reset otb module with new path. You need to restart Python")

    # Else try with python file location
    try:
        import otbApplication
        lib_dir = get_lib(otbApplication)
        apps_path = get_apps_path(lib_dir)
        logger.debug("Found OTB directory using Python lib location")
        return str(lib_dir.parent), str(apps_path)

    # Else search system
    except ImportError:
        PYTHONPATH = os.environ.get("PYTHONPATH")
        logger.info("Failed to import otbApplication with PYTHONPATH=%s", PYTHONPATH)
        if not scan:
            return None, None
        logger.info("Searching for it...")
        lib_dir = None
        # Scan user's HOME directory tree (this may take some time)
        if scan_userdir:
            for path in Path().home().glob("**/OTB-*/lib"):
                logger.info("Found %s", path.parent)
                lib_dir = path
                prefix = str(path.parent.absolute())
        # Or search possible known locations (system scpecific)
        if sys.platform == "linux":
            possible_locations = (
                "/usr/lib/x86_64-linux-gnu/otb", "/usr/local/lib/otb/",
                "/opt/otb/lib/otb/", "/opt/otbtf/lib/otb",
            )
            for str_path in possible_locations:
                path = Path(str_path)
                if not path.exists():
                    continue
                logger.info("Found %s", str_path)
                if not prefix:
                    if path.parent.name == "x86_64-linux-gnu":
                        prefix = path.parent.parent.parent
                    else:
                        prefix = path.parent.parent
                    prefix = str(prefix.absolute())
                    lib_dir = path.parent.absolute()
        else:
            # TODO: find OTB path in other OS ? pathlib should help
            pass

        # Found OTB
        if prefix and lib_dir is not None:
            otb_api = get_python_api(prefix)
            if otb_api is None or not otb_api.exists():
                logger.error("Can't find OTB python API")
                return None, None

            # Try to import one last time before sys.exit (in apps.py)
            try:
                sys.path.insert(0, str(otb_api))
                import otbApplication
                logger.info("Using OTB in %s", prefix)
                return prefix, get_apps_path(lib_dir)
            except ModuleNotFoundError:
                logger.critical("Unable to find OTB Python bindings", exc_info=1)
                return None, None
            except ImportError as e:
                logger.critical("An error occurred while importing Python API")
                if str(e).startswith('libpython3.'):
                    logger.critical("It seems like you need to symlink or recompile OTB SWIG bindings")
                    if sys.platform == "linux":
                        logger.critical("Use 'cd %s ; source otbenv.profile ; "
                                        "ctest -S share/otb/swig/build_wrapping.cmake -VV'", prefix)
                        return None, None
                logger.critical("full traceback", exc_info=1)

    return None, None


def get_python_api(prefix):
    """
    Try to find the python path
    :param prefix: prefix
    :return: the path if found, else None
    """
    root = Path(prefix)
    if root.exists():
        otb_api = root / "lib/python"
        if not otb_api.exists():
            otb_api = root / "lib/otb/python"
        if otb_api.exists():
            return otb_api.absolute()
    return None


def get_lib(otb_module):
    """
    Try to find the otb library path
    :param otb_module: otb module (otbApplication)
    :return: library path
    """
    lib_dir = Path(otb_module.__file__).parent.parent
    # OTB .run file
    if lib_dir.name == "lib":
        return lib_dir.absolute()
    # Case /usr
    lib_dir = lib_dir.parent
    if lib_dir.name in ("lib", "x86_64-linux-gnu"):
        return lib_dir.absolute()
    # Case built from source (/usr/local, /opt/otb, ...)
    lib_dir = lib_dir.parent
    return lib_dir.absolute()


def get_apps_path(lib_dir):
    """
    Try to get the otb applications path
    :param lib_dir: library path
    :return: library path if found, else ""
    """
    if isinstance(lib_dir, Path) and lib_dir.exists():
        logger.debug("Using library from %s", lib_dir)
        otb_application_path = lib_dir / "otb/applications"
        if otb_application_path.exists():
            return str(otb_application_path.absolute())
        # This should not happen, may be with failed builds ?
        logger.debug("Library directory found but no 'applications' directory inside")
    return ""


def set_gdal_vars(root):
    """
    Set GDAL environment variables
    :param root: root directory
    :return: True is succeed, False else
    """
    if (Path(root) / "share/gdal").exists():
        # Local GDAL (OTB Superbuild, .run, .exe)
        gdal_data = str(Path(root + "/share/gdal"))
        proj_lib = str(Path(root + "/share/proj"))
    elif sys.platform == "linux":
        # If installed using apt or built from source with system deps
        gdal_data = "/usr/share/gdal"
        proj_lib = "/usr/share/proj"
    else:
        logger.warning("Can't find GDAL directory with prefix %s", root)
        return False
    # Not sure if SWIG will see these
    os.environ["LC_NUMERIC"] = "C"
    os.environ["GDAL_DATA"] = gdal_data
    os.environ["PROJ_LIB"] = proj_lib
    return True
