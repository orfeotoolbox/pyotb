"""This module contains functions for interactive auto installation of OTB."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import sysconfig
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from shutil import which


def interactive_config():
    """Prompt user to configure installation variables."""
    version = input("Choose a version to download (default is latest): ")
    default_dir = Path.home() / "Applications"
    path = input(f"Parent directory for installation (default is {default_dir}): ")
    env = input("Permanently change user's environment variables ? (y/n): ") == "y"
    return version, path, env


def otb_latest_release_tag():
    """Use gitlab API to find latest release tag name, but skip pre-releases."""
    api_endpoint = "https://gitlab.orfeo-toolbox.org/api/v4/projects/53/repository/tags"
    vers_regex = re.compile(r"^\d\.\d\.\d$")  # we ignore rc-* or alpha-*
    with urllib.request.urlopen(api_endpoint) as stream:
        data = json.loads(stream.read())
    releases = sorted(
        [tag["name"] for tag in data if vers_regex.match(tag["name"])],
    )
    return releases[-1]


def check_versions(sysname: str, python_minor: int, otb_major: int) -> tuple[bool, int]:
    """Verify if python version is compatible with major OTB version.

    Args:
        sysname: OTB's system name convention (Linux64, Darwin64, Win64)
        python_minor: minor version of python
        otb_major: major version of OTB to be installed

    Returns:
        (True, 0) if compatible or (False, expected_version) in case of conflicts

    """
    if sysname == "Win64":
        expected = 5 if otb_major in (6, 7) else 7
        if python_minor == expected:
            return True, 0
    elif sysname == "Darwin64":
        expected = 7, 0
        if python_minor == expected:
            return True, 0
    elif sysname == "Linux64":
        expected = 5 if otb_major in (6, 7) else 8
        if python_minor == expected:
            return True, 0
    return False, expected


def env_config_unix(otb_path: Path):
    """Update env profile for current user with new otb_env.profile call.

    Args:
        otb_path: the path of the new OTB installation

    """
    profile = Path.home() / ".profile"
    with profile.open("a", encoding="utf-8") as buf:
        buf.write(f'\n. "{otb_path}/otbenv.profile"\n')
        print(f"##### Added new environment variables to {profile}")


def env_config_windows(otb_path: Path):
    """Update user's registry hive with new OTB_ROOT env variable.

    Args:
        otb_path: path of the new OTB installation

    """
    import winreg  # pylint: disable=import-error,import-outside-toplevel

    with winreg.OpenKeyEx(
        winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE
    ) as reg_key:
        winreg.SetValueEx(reg_key, "OTB_ROOT", 0, winreg.REG_EXPAND_SZ, str(otb_path))
        print(
            "##### Environment variable 'OTB_ROOT' added to user's registry. "
            "You'll need to login / logout to apply this change."
        )
        reg_cmd = "reg.exe delete HKEY_CURRENT_USER\\Environment /v OTB_ROOT /f"
        print(f"To undo this, you may use '{reg_cmd}'")


def install_otb(version: str = "latest", path: str = "", edit_env: bool = True):
    """Install pre-compiled OTB binaries in path, use latest release by default.

    Args:
        version: OTB version tag, e.g. '8.1.2'
        path: installation directory, default is $HOME/Applications
        edit_env: whether or not to permanently modify user's environment variables

    Returns:
        full path of the new installation

    Raises:
        SystemExit: if python version is not compatible with major OTB version
        SystemError: if automatic env config failed

    """
    # Read env config
    if sys.version_info.major == 2:
        raise SystemExit("Python 3 is required for OTB bindings.")
    python_minor = sys.version_info.minor
    if not version or version == "latest":
        version = otb_latest_release_tag()
    name_corresp = {"linux": "Linux64", "darwin": "Darwin64", "win32": "Win64"}
    sysname = name_corresp[sys.platform]
    ext = "zip" if sysname == "Win64" else "run"
    cmd = which("zsh") or which("bash") or which("sh")
    otb_major = int(version[0])
    check, expected = check_versions(sysname, python_minor, otb_major)
    if sysname == "Win64" and not check:
        raise SystemExit(
            f"Python 3.{expected} is required to import bindings on Windows."
        )

    # Fetch archive and run installer
    filename = f"OTB-{version}-{sysname}.{ext}"
    url = f"https://www.orfeo-toolbox.org/packages/archives/OTB/{filename}"
    tmpdir = tempfile.gettempdir()
    tmpfile = Path(tmpdir) / filename
    print(f"##### Downloading {url}")
    urllib.request.urlretrieve(url, tmpfile)
    if path:
        default_path = False
        path = Path(path)
    else:
        default_path = True
        path = Path.home() / "Applications" / tmpfile.stem
    if sysname == "Win64":
        with zipfile.ZipFile(tmpfile) as zipf:
            print("##### Extracting zip file")
            # Unzip will always create a dir with OTB-version name
            zipf.extractall(path.parent if default_path else path)
    else:
        install_cmd = f"{cmd} {tmpfile} --target {path} --accept"
        print(f"##### Executing '{install_cmd}'\n")
        subprocess.run(install_cmd, shell=True, check=True)
    tmpfile.unlink()  # cleaning

    # Add env variable to profile
    if edit_env:
        if sysname == "Win64":
            env_config_windows(path)
        else:
            env_config_unix(path)
    elif not default_path:
        ext = "bat" if sysname == "Win64" else "profile"
        print(
            f"Remember to call '{path}{os.sep}otbenv.{ext}' before importing pyotb, "
            f"or add 'OTB_ROOT=\"{path}\"' to your env variables."
        )
    # Requirements are met, no recompilation or symlink required
    if check:
        return str(path)

    # Else try recompile bindings : can fail because of OpenGL
    # Here we check for /usr/bin because CMake's will find_package() only there
    if (
        sys.executable.startswith("/usr/bin")
        and which("ctest")
        and which("python3-config")
    ):
        try:
            print("\n!!!!! Python version mismatch, trying to recompile bindings")
            ctest_cmd = (
                ". ./otbenv.profile && ctest -S share/otb/swig/build_wrapping.cmake -V"
            )
            print(f"##### Executing '{ctest_cmd}'")
            subprocess.run(ctest_cmd, cwd=path, check=True, shell=True)
            return str(path)
        except subprocess.CalledProcessError:
            print("\nCompilation failed.")
    # TODO: support for sudo auto build deps install using apt, pacman/yay, brew...
    print(
        "You need cmake, python3-dev and libgl1-mesa-dev installed."
        "\nTrying to symlink libraries instead - this may fail with newest versions."
    )

    # Finally try with cross version python symlink (only tested on Ubuntu)
    suffix = "so.1.0" if otb_major >= 8 else f"so.rh-python3{expected}-1.0"
    target_lib = f"{path}/lib/libpython3.{expected}.{suffix}"
    lib_dir = sysconfig.get_config_var("LIBDIR")
    lib = f"{lib_dir}/libpython3.{sys.version_info.minor}.so"
    if Path(lib).exists():
        print(f"##### Creating symbolic link: {lib} -> {target_lib}")
        ln_cmd = f'ln -sf "{lib}" "{target_lib}"'
        subprocess.run(ln_cmd, executable=cmd, shell=True, check=True)
        return str(path)
    raise SystemError(
        f"Unable to automatically locate library for executable '{sys.executable}', "
        f"you could manually create a symlink from that file to {target_lib}"
    )
