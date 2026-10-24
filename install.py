"""
Install ProphitBet dependencies and create a launcher for this OS.
This version supports 64-bit Python 3.11 only.

Usage:
    python install.py
    python install.py --venv "C:\\Users\\You\\python\\envs\\myenv"
    python install.py --venv "/home/you/python/envs/myenv"
    python install.py --skip-install   # Regenerate launchers only

Outputs (in the directory containing this file):
    Windows: app.bat, ProphitBet.exe (if the C# compiler is available)
    Linux:   prophitbet.sh
    macOS:   prophitbet.command

The Windows EXE is only a small launcher for app.bat. Python, the dependencies,
main.py, and the BAT file remain separate and are required to run the app.
"""

import argparse
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional


SUPPORTED_PY_VERSIONS = {(3, 11)}
APP_NAME = "ProphitBet"

# Optional: set this to a folder containing an existing virtual environment,
# or a new folder where an environment should be created automatically.
# Leave None to use the Python interpreter running install.py.
# Examples: r"C:\Users\You\python\envs\prophitbet", "Downloads/myEnv", "Home/prophitbetenv", etc.
VENV_PATH = None


def resolve_venv_path(raw_path: str, project: Path) -> Path:
    path = Path(raw_path).expanduser()

    if not path.is_absolute():
        path = project / path

    return path.resolve()


def venv_python(venv: Path) -> Path:
    if platform.system() == "Windows":
        return venv / "Scripts" / "python.exe"

    python3 = venv / "bin" / "python3"
    return python3 if python3.is_file() else venv / "bin" / "python"


def venv_activation(venv: Path) -> Path:
    """ Finds the Python interpreter executable based on the operating system. """

    if platform.system() == "Windows":
        return venv / "Scripts" / "activate.bat"

    return venv / "bin" / "activate"


def chosen_python_info(python: Path):
    """ Return the interpreter version, implementation and pointer width. """

    result = subprocess.run(
        [
            str(python), "-c",
            "import struct, sys; print('{}.{}.{}|{}|{}'.format(*sys.version_info[:3], "
            "sys.implementation.name, struct.calcsize('P') * 8))",
        ],
        check=True,
        capture_output=True,
        text=True
    )
    version, implementation, bits = result.stdout.strip().split("|")
    return version, implementation, int(bits)


def supported_python(python: Path) -> bool:
    try:
        version, implementation, bits = chosen_python_info(python)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print("ERROR: Could not inspect Python interpreter {}: {}".format(python, exc))
        return False

    print("Using Python: {} ({}; {}-bit {})".format(python, version, bits, implementation))
    if tuple(map(int, version.split(".")[:2])) not in SUPPORTED_PY_VERSIONS or implementation != "cpython" or bits != 64:
        print("ERROR: These exact package pins require 64-bit CPython 3.11.")
        print("       Python 3.8, 3.9, 3.10 and 3.12 are not supported by this requirements.txt.")
        return False
    return True


def ensure_venv(venv: Path, skip_install: bool) -> bool:
    """ Reuse an existing venv or create one at the explicitly chosen path. """

    if (venv / "pyvenv.cfg").is_file():
        print("Reusing virtual environment: {}".format(venv))
        return True
    if venv.exists() and (not venv.is_dir() or any(venv.iterdir())):
        print("ERROR: {} exists but is not a virtual environment or an empty folder.".format(venv))
        return False
    if skip_install:
        print("ERROR: {} does not contain a venv. Remove --skip-install to create it.".format(venv))
        return False
    if not supported_python(Path(sys.executable)):
        print("Create new virtual environments with a 64-bit Python 3.11 interpreter.")
        return False
    print("Creating new virtual environment: {}".format(venv))

    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print("ERROR: Could not create virtual environment: {}".format(exc))
        return False
    return True


def batch_path(path: Path) -> str:
    """ Escape percent signs before a path is inserted into a Windows BAT. """

    return str(path).replace("%", "%%")


def make_windows_bat(project: Path, venv: Optional[Path], python: Path) -> Path:
    lines = [
        "@echo off",
        "REM Navigate to the directory containing this batch file",
        'cd /d "%~dp0" || exit /b 1',
        "",
        "REM Prefer the Python installation selected when running install.py",
        'set "PATH={};%PATH%"'.format(batch_path(python.parent)),
        "",
        "REM (Optional) Activate your virtual environment first:",
        'REM call "%USERPROFILE%\\python\\envs\\myenv\\Scripts\\activate.bat"',
    ]
    if venv:
        lines.extend([
            "REM Activate the environment selected during installation",
            'call "{}" || exit /b 1'.format(batch_path(venv_activation(venv))),
        ])
    lines.extend([
        "",
        "REM Open cmd in the current project directory and run main.py",
        'cmd /k "python main.py"',
        "",
    ])

    path = project / "app.bat"
    with path.open("w", encoding="utf-8", newline="\r\n") as file:
        file.write("\n".join(lines))
    return path


def make_unix_launcher(project: Path, venv: Optional[Path], python: Path, macos: bool) -> Path:
    lines = [
        "#!/bin/bash",
        "# Navigate to the directory containing this launcher",
        'cd "$(dirname "$0")" || exit 1',
        "",
        "# Prefer the Python installation selected when running install.py",
        'export PATH={}:"$PATH"'.format(shlex.quote(str(python.parent))),
        "",
        "# (Optional) Activate your virtual environment first:",
        '# source "$HOME/python/envs/myenv/bin/activate"',
    ]
    if venv:
        lines.extend([
            "# Activate the environment selected during installation",
            "source {} || exit 1".format(shlex.quote(str(venv_activation(venv)))),
        ])
    lines.extend([
        "",
        "# Run main.py",
        "python3 main.py",
    ])
    if macos:
        lines.extend([
            "status=$?",
            '# Keep Terminal open when launched by double-clicking prophitbet.command',
            'if [[ -t 0 ]]; then read -r -p "Press Enter to close..." _; fi',
            'exit "$status"',
        ])
    lines.append("")

    path = project / ("prophitbet.command" if macos else "prophitbet.sh")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)
    return path


def compile_windows_exe(project: Path) -> bool:
    """ Compile the supplied Launcher.cs using the .NET Framework C# compiler. """
    
    source = project / "Launcher.cs"
    if not source.is_file():
        print("WARNING: Launcher.cs is missing; ProphitBet.exe was not generated.")
        return False

    windows_dir = os.environ.get("WINDIR") or os.environ.get("SystemRoot", r"C:\Windows")
    compiler_paths = [
        Path(windows_dir) / "Microsoft.NET" / "Framework64" / "v4.0.30319" / "csc.exe",
        Path(windows_dir) / "Microsoft.NET" / "Framework" / "v4.0.30319" / "csc.exe",
        ]
    compiler = next((candidate for candidate in compiler_paths if candidate.is_file()), None)
    if compiler is None:
        print("WARNING: .NET Framework C# compiler (csc.exe) was not found.")
        print("         Install a compatible compiler and compile Launcher.cs manually.")
        print("         app.bat is ready to use without the EXE.")
        return False

    output = project / (APP_NAME + ".exe")
    command = [
        str(compiler),
        "/nologo",
        "/target:winexe",
        "/out:" + str(output),
        "/reference:System.Windows.Forms.dll",
        str(source),
        ]
    print("Compiling Windows launcher: {}".format(output))
    try:
        subprocess.run(command, cwd=str(project), check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print("WARNING: Could not compile ProphitBet.exe: {}".format(exc))
        return False

    print("Created: {}".format(output))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Install ProphitBet requirements and generate launchers.")
    parser.add_argument("--venv", help="Override VENV_PATH: create or reuse a virtual environment at this path.")
    parser.add_argument("--skip-install", action="store_true", help="Only regenerate launchers; do not install packages.")
    parser.add_argument("--pause", action="store_true", help="Wait for Enter before exiting (useful on Windows).")
    args = parser.parse_args()

    project = Path(__file__).resolve().parent
    os_name = platform.system()
    requirements = project / "requirements.txt"
    entrypoint = project / "main.py"

    print("{} installer | {} | Python {}".format(APP_NAME, os_name, sys.version.split()[0]))
    if not entrypoint.is_file():
        old_name = project / "app.py"
        print("ERROR: main.py not found in {}.".format(project))
        if old_name.is_file():
            print("Rename app.py to main.py in your repository before installing.")
        return 1
    if not args.skip_install and not requirements.is_file():
        print("ERROR: requirements.txt not found in {}.".format(project))
        return 1

    configured_venv = args.venv if args.venv is not None else VENV_PATH
    venv = resolve_venv_path(configured_venv, project) if configured_venv else None
    if venv and not ensure_venv(venv, args.skip_install):
        return 1

    python = venv_python(venv) if venv else Path(sys.executable)
    if not python.is_file():
        print("ERROR: Python interpreter not found: {}".format(python))
        return 1
    if venv and not venv_activation(venv).is_file():
        print("ERROR: Virtual environment activation script not found: {}".format(venv_activation(venv)))
        return 1

    if not supported_python(python):
        return 1

    if not args.skip_install:
        print("Installing packages from {}".format(requirements))
        try:
            subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)], check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            print("ERROR: Dependency installation failed: {}".format(exc))
            print("Consider using a virtual environment: python -m venv .venv")
            return 1
        check_result = subprocess.run([str(python), "-m", "pip", "check"], check=False)
        if check_result.returncode:
            print("WARNING: pip check reported dependency conflicts; review the output above.")

    if os_name == "Windows":
        print("Created: {}".format(make_windows_bat(project, venv, python)))
        if not compile_windows_exe(project):
            print("Windows EXE not generated. You can still launch using app.bat.")
    elif os_name == "Linux":
        print("Created: {}".format(make_unix_launcher(project, venv, python, macos=False)))
    elif os_name == "Darwin":
        print("Created: {}".format(make_unix_launcher(project, venv, python, macos=True)))
    else:
        print("ERROR: Unsupported OS: {}".format(os_name))
        return 1

    print("Installation and launcher setup completed.")
    return 0


if __name__ == "__main__":
    status = main()
    if "--pause" in sys.argv:
        try:
            input("\nPress Enter to close...")
        except EOFError:
            pass
    sys.exit(status)
