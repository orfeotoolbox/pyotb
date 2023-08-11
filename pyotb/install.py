"""This module contains functions for interactive auto installation of OTB."""
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from shutil import which


def interactive_config():
    """Prompt user to configure installation variables."""
    version = input("Choose a version to download (default is latest): ")
    default_dir = f"<user_dir>{os.path.sep}Applications{os.path.sep}"
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


def check_versions(sysname: str, python_minor: int, otb_major: int):
    """Verify if python version is compatible with OTB version.

    Args:
        sysname: OTB's system name convention (Linux64, Darwin64, Win64)
        python_minor: minor version of python
        otb_major: major version of OTB to be installed
    Returns:
        True if requirements are satisfied
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


def update_unix_env(otb_path: Path):
    """Update env profile for current user with new otb_env.profile call.

    Args:
        otb_path: the path of the new OTB installation

    """
    profile = Path.home() / ".profile"
    with open(profile, "a", encoding="utf-8") as buf:
        buf.write(f'\n. "{otb_path}/otbenv.profile"\n')
        print(f"##### Added new environment variables to {profile}")


def update_windows_env(otb_path: Path):
    """Update registry hive for current user with new OTB_ROOT env variable.

    Args:
        otb_path: path of the new OTB installation

    """
    import winreg  # pylint: disable=import-error,import-outside-toplevel

    with winreg.OpenKeyEx(
        winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE
    ) as reg_key:
        winreg.SetValueEx(reg_key, "OTB_ROOT", 0, winreg.REG_EXPAND_SZ, str(otb_path))
        print(
            "##### Environment variable 'OTB_ROOT' added to user's registry."
            "You'll need to login / logout to apply this change."
        )
        reg_cmd = "reg.exe delete HKEY_CURRENT_USER\\Environment /v OTB_ROOT /f"
        print(f"To undo this, you may use '{reg_cmd}'")


def recompile_python_bindings(path: Path, cmd: str):
    """Run subprocess command to recompile python bindings.

    Args:
        path: path of the new OTB installation
        cmd: path of the default system shell command

    """
    print("\n##### Recompiling python bindings...")
    ctest_cmd = ". ./otbenv.profile && ctest -S share/otb/swig/build_wrapping.cmake -VV"
    try:
        subprocess.run(ctest_cmd, executable=cmd, cwd=str(path), shell=True, check=True)
    except subprocess.CalledProcessError as err:
        raise SystemExit(
            "Unable to recompile python bindings, "
            "some dependencies (libgl1) may require manual installation."
        ) from err


def symlink_python_library(target_lib: str, cmd: str):
    """Run subprocess command to recompile python bindings.

    Args:
        target_lib: path of the missing python library
        cmd: path of the default system shell command

    """
    lib = f"/usr/lib/x86_64-linux-gnu/libpython3.{sys.version_info.minor}.so"
    if Path(lib).exists():
        print(f"##### Creating symbolic links: {lib} -> {target_lib}")
        ln_cmd = f'ln -sf "{lib}" "{target_lib}"'
        subprocess.run(ln_cmd, executable=cmd, shell=True, check=True)


def install_otb(version: str = "latest", path: str = "", edit_env: bool = False):
    """Install pre-compiled OTB binaries in path, use latest release by default.

    Args:
        version: OTB version tag, e.g. '8.1.2'
        path: installation directory, default is $HOME/Applications
        edit_env: whether or not to permanently modify user's environment variables

    Returns:
        full path of the new installation

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
            print("##### Extracting zip file...")
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
            update_windows_env(path)
        else:
            update_unix_env(path)
    elif not default_path:
        ext = "bat" if sysname == "Win64" else "profile"
        print(
            f"Remember to call '{path}{os.sep}otbenv.{ext}' before importing pyotb, "
            f"or add 'OTB_ROOT=\"{path}\"' to your env variables."
        )
    # No recompilation or symlink required
    if check:
        return str(path)

    # Else recompile bindings : this may fail because of OpenGL
    suffix = f"so.rh-python3{expected}-1.0" if otb_major < 8 else "so.1.0"
    target_lib = f"{path}/lib/libpython3.{expected}.{suffix}"
    can_compile = which("ctest") and which("python3-config")
    # Google Colab ships with cmake and python3-dev, but not libgl1-mesa-dev
    if can_compile and "COLAB_RELEASE_TAG" not in os.environ:
        recompile_python_bindings(path, cmd)
        return str(path)
    # Or use dirty cross version python symlink (only tested on Ubuntu)
    if sys.executable.startswith("/usr/bin"):
        symlink_python_library(target_lib, cmd)
        return str(path)
    print(
        f"Unable to automatically locate library for executable {sys.executable}"
        f"You could manually create a symlink from that file to {target_lib}"
    )
    # TODO: support for auto build deps install using brew, apt, pacman/yay, yum...
    msg = (
        "\nYou need to install 'cmake', 'python3-dev' and 'libgl1-mesa-dev'"
        " in order to recompile python bindings. "
    )
    raise SystemExit(msg)
