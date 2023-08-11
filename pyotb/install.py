"""This module contains functions for interactive auto installation of OTB."""
import json
import sys
import re
import subprocess
import tempfile
import zipfile
import urllib.request
from pathlib import Path
from shutil import which


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


def install_otb(version: str = "latest", path: str = "", edit_env: bool = False):
    """Install pre-compiled OTB binaries in path, use latest release by default.

    Args:
        version: OTB version tag, e.g. '8.1.2'
        path: installation directory, default is $HOME/Applications

    Returns:
        full path of the new installation
    """
    # Read env config
    if sys.version_info.major == 2:
        raise SystemExit("Python 3 is required for OTB bindings.")
    python_minor = sys.version_info.minor
    if not version or version == "latest":
        version = otb_latest_release_tag()
    name_corresp = {"linux": "Linux64", "darwnin": "Darwin64", "win32": "Win64"}
    sysname = name_corresp[sys.platform]
    ext = "zip" if sysname == "Win64" else "run"
    cmd = which("zsh") or which("bash") or which("sh")
    otb_major = int(version[0])
    check, expected = check_versions(sysname, python_minor, otb_major)
    if sysname == "Win64" and not check:
        print(f"Python 3.{expected} is required to import bindings on Windows.")
        return

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
            zipf.extractall(path.parent)
    else:
        install_cmd = f"{cmd} {tmpfile} --target {path} --accept"
        print(f"##### Executing '{install_cmd}'\n")
        subprocess.run(install_cmd, shell=True, check=True)
    tmpfile.unlink()  # cleaning

    # Add env variable to profile
    if edit_env:
        if sysname == "Win64":
            # TODO: import winreg
            return str(path)
        else:
            profile = Path.home() / ".profile"
            print(f"##### Adding new env variables to {profile}")
            with open(profile, "a", encoding="utf-8") as buf:
                buf.write(f'\n. "{path}/otbenv.profile"\n')
    elif not default_path:
        ext = "bat" if sysname == "Win64" else "profile"
        print(
            f"Remember to call 'otbenv.{ext}' before importing pyotb, "
            f"or add 'OTB_ROOT=\"{path}\"' to your env variables."
        )
    # No recompilation or symlink required
    if check:
        return str(path)

    # Else recompile bindings : this may fail because of OpenGL
    if which("ctest") and which("python3-config"):
        print("\n##### Recompiling python bindings...")
        ctest_cmd = (
            ". ./otbenv.profile && ctest -S share/otb/swig/build_wrapping.cmake -VV"
        )
        try:
            subprocess.run(ctest_cmd, executable=cmd, cwd=str(path), shell=True, check=True)
            return str(path)
        except subprocess.CalledProcessError as err:
            raise SystemExit(
                "Unable to recompile python bindings, "
                "some dependencies (libgl1) may require manual installation."
            ) from err
    # Use dirty cross python version symlink (only tested on Ubuntu)
    elif sys.executable.startswith("/usr/bin"):
        suffix = f"so.rh-python3{expected}-1.0" if otb_major < 8 else ".so.1.0"
        target_lib = f"{path}/lib/libpython3.{expected}.{suffix}"
        lib = f"/usr/lib/x86_64-linux-gnu/libpython3.{sys.version_info.minor}.so"
        if Path(lib).exists():
            print(f"Creating symbolic links: {lib} -> {target_lib}")
            ln_cmd = f'ln -sf "{lib}" "{target_lib}"'
            subprocess.run(ln_cmd, executable=cmd, shell=True, check=True)
            return str(path)
    else:
        print(
            f"Unable to automatically locate library for executable {sys.executable}"
            f"You'll need to manually symlink that one file to {target_lib}"
        )
    # TODO: support for auto build deps install using brew, apt, pacman/yay, yum...
    msg = (
        "\nYou need to install 'cmake', 'python3-dev' and 'libgl1-mesa-dev'"
        " in order to recompile python bindings. "
    )
    raise SystemExit(msg)


def interactive_config():
    """Prompt user to configure installation variables."""
    version = input("Choose a version number to install (default is latest): ")
    path = input(
        "Provide a path for installation "
        "(default is <user_dir>/Applications/OTB-<version>): "
    )
    edit_env = (
        input("Modify user environment variables for this installation ? (y/n): ")
        == "y"
    )
    return version, path, edit_env
