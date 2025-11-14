"""
!! --- IMPORTANT NOTES --- !!
This installer detects (optionally) a provided virtual environment and installs the required libraries.
These libraries (see requirements.txt) are needed for Prophitbet-v2 to run.

Recommended Python is 3.11.*. Other versions (3.9–3.13) may work but are not fully tested.

USAGE
-----
# System Python:
python install.py (or double click on windows).

# Install into an existing virtual environment:
python install.py --venv "C:\Users\You\python\envs\myenv"      (Windows)
python install.py --venv "/home/you/python/envs/myenv"         (Linux/macOS)

If installation succeeds, launcher scripts are created:
- run_app.bat (Windows)
- run_app.sh  (Linux/macOS)
These will activate the venv (if provided) and run app.py.
"""

import argparse
import os
import sys
import subprocess
import platform
from shutil import which
from typing import List, Optional

RECOMMENDED_PY_VERSION = (3, 11)


def in_virtualenv() -> bool:
    return getattr(sys, 'base_prefix', sys.prefix) != sys.prefix or hasattr(sys, 'real_prefix')


def run(cmd, check=True):
    print(f"--> Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def guess_venv_python(venv_path: str) -> Optional[str]:
    """Return the Python executable inside the given venv, or None if not found."""

    if platform.system() == 'Windows':
        # For windows, simply join the provided path with Scripts/python.exe.
        exec_path = os.path.join(venv_path, 'Scripts', 'python.exe')
    else:
        # For linux, try bin/python3 or bin/python.
        exec_path = os.path.join(venv_path, 'bin', 'python3')

        if not os.path.exists(exec_path):  # fallback to 'python'
            exec_path = os.path.join(venv_path, 'bin', 'python')

    return exec_path if os.path.exists(exec_path) else None


def get_pip_cmd(python_exe: str) -> List[str]:
    """Get pip command bound to the specified python interpreter."""

    return [python_exe, '-m', 'pip']


def pause_if_double_clicked():
    """ Keeps the window open when double-clicked (mostly useful on Windows) """

    try:
        input('\nDone. Press ENTER to close this window...')
    except EOFError:
        pass


def get_requirements(file_path: str = "requirements.txt") -> List[str]:
    """ Gets a list with all requirements. """

    if not os.path.exists(file_path):
        print(f'ERROR: {file_path} not found.')
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]


def make_run_bat(project_dir: str, venv_path: Optional[str]) -> None:
    """ Create run_app.bat that runs app.py, optionally activating venv. """

    lines = [
        "@echo off",
        "REM Navigate to the directory of the batch file",
        'cd /d "%~dp0"',
        "",
    ]
    if venv_path:
        act = os.path.join(venv_path, "Scripts", "activate.bat")
        lines += [
            "REM Auto-activate virtual environment",
            f'call "{act}"',
            "",
        ]
    lines += [
        "REM Open cmd in the current project directory and run app.py",
        'cmd /k "python app.py"',
        "",
    ]
    path = os.path.join(project_dir, "run_app.bat")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Created: {path}")


def make_run_sh(project_dir: str, venv_path: Optional[str]) -> None:
    """Create run_app.sh that runs app.py, optionally activating venv."""
    lines = [
        "#!/bin/bash",
        '# Navigate to the directory of the script',
        'cd "$(dirname "$0")"',
        "",
    ]
    if venv_path:
        act = os.path.join(venv_path, "bin", "activate")
        lines += [
            "# Auto-activate virtual environment",
            f'source "{act}"',
            "",
        ]
    lines += [
        "# Run app.py",
        "python3 app.py",
        "",
    ]
    path = os.path.join(project_dir, "run_app.sh")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Make executable (best effort)
    try:
        os.chmod(path, 0o755)
    except Exception:
        pass

    print(f'Created: {path}')


def main():
    parser = argparse.ArgumentParser(description="Install project requirements.")
    parser.add_argument(
        "--venv",
        type=str,
        default=None,
        help="Path to an existing virtual environment to install into (optional).",
    )
    args = parser.parse_args()

    py_ver = sys.version_info[:3]
    interp = sys.executable
    sysname = platform.system()
    arch = platform.machine()

    print('-' * 70)
    print('Python package installer')
    print('-' * 70)
    print(f'Interpreter : {interp}')
    print(f'Platform    : {sysname} ({arch})')
    print(f'Python      : {py_ver[0]}.{py_ver[1]}.{py_ver[2]}')
    print(f"Environment : {'Virtualenv' if in_virtualenv() else 'System (no venv detected)'}")
    if args.venv:
        print(f'--venv      : {args.venv}')
    print('-' * 70)

    if (py_ver[0], py_ver[1]) != RECOMMENDED_PY_VERSION:
        print(
            f'WARNING: Python {RECOMMENDED_PY_VERSION[0]}.{RECOMMENDED_PY_VERSION[1]} is recommended for this app. '
            f'Continuing on {py_ver[0]}.{py_ver[1]}.{py_ver[2]}.\n'
        )

    # Choose which Python to use for installation:
    if args.venv:
        venv_python = guess_venv_python(args.venv)
        if not venv_python:
            print("ERROR: Could not locate Python inside the provided venv.\n"
                  "Ensure the venv exists and is initialized (python -m venv <path>).")
            pause_if_double_clicked()
            sys.exit(1)
        python_for_install = venv_python
    else:
        python_for_install = interp

    # Sanity check: chosen Python must be resolvable
    if which(python_for_install) is None and not os.path.exists(python_for_install):
        print('ERROR: Could not locate the chosen Python interpreter.\n'
              'Try manually installing the libraries or provide a valid --venv.')
        pause_if_double_clicked()
        sys.exit(1)

    # Read requirements
    packages = get_requirements("requirements.txt")
    if not packages:
        print("No packages to install (requirements.txt empty or missing).")

    pip = get_pip_cmd(python_for_install)

    try:
        print('\n--> Upgrading pip, setuptools, wheel...\n')
        run(pip + ['install', '--upgrade', 'pip', 'setuptools', 'wheel'])

        if packages:
            print('\n--> Installing required packages. This may take a while...\n')
            run(pip + ['install', '--no-input'] + packages)

        print("\n--> Verifying environment with 'pip check'...\n")

        # Don't hard-fail if some packages only warn
        run(pip + ['check'], check=False)

        print("\nInstallation complete!")

        # Generate launchers
        project_dir = os.path.abspath(os.path.dirname(__file__))
        make_run_bat(project_dir, args.venv)
        make_run_sh(project_dir, args.venv)

        print("\nLaunchers created:")
        print(" - run_app.bat (Windows)")
        print(" - run_app.sh  (Linux/macOS)")
        print("\nTip:")
        print(" - On Linux/macOS: if double-click closes Terminal immediately, run from an open terminal or use:")
        print('     gnome-terminal -- bash -c "./run_app.sh; exec bash"')
    except subprocess.CalledProcessError as e:
        print("\nAn installation step failed.")
        print("Common fixes:")
        print(" - Check your internet connection.")
        if sysname in ("Linux", "Darwin"):
            print(" - You may need system build tools for packages like lxml/numba.")
            print("   Debian/Ubuntu:  sudo apt-get install build-essential libxml2-dev libxslt1-dev")
            print("   macOS:         xcode-select --install")
        if sysname == "Windows":
            print(" - If you see permission errors, try an Administrator terminal.")
        print(" - Prefer installing inside a virtual environment to avoid conflicts.")
        print(f"\nDetails: {e}\n")
    finally:
        pause_if_double_clicked()


if __name__ == "__main__":
    main()
