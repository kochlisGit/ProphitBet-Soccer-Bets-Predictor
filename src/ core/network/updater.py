"""Download and apply application updates from the official GitHub repository."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable, Optional
from urllib.request import Request, urlopen
from zipfile import ZipFile


UPDATE_ARCHIVE_URL = (
    'https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/'
    'archive/refs/heads/main.zip'
)


def project_root() -> Path:
    """Return the application root containing ``src`` and the launch scripts."""

    return Path(__file__).resolve().parents[3]


def stage_update(progress: Optional[Callable[[int], None]] = None) -> Path:
    """Download and safely extract the latest source tree to a temporary folder."""

    update_dir = Path(tempfile.mkdtemp(prefix='prophitbet-update-'))
    archive_path = update_dir / 'update.zip'
    request = Request(UPDATE_ARCHIVE_URL, headers={'User-Agent': 'ProphitBet-Updater'})

    try:
        with urlopen(request, timeout=60) as response, archive_path.open('wb') as archive:
            total = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                archive.write(chunk)
                downloaded += len(chunk)
                if progress is not None and total:
                    progress(min(90, int(downloaded * 90 / total)))

        extract_dir = update_dir / 'extracted'
        extract_dir.mkdir()
        with ZipFile(archive_path) as zip_file:
            members = zip_file.infolist()
            if not members:
                raise RuntimeError('The downloaded update archive is empty.')

            extract_root = extract_dir.resolve()
            for member in members:
                destination = (extract_dir / member.filename).resolve()
                if extract_root != destination and extract_root not in destination.parents:
                    raise RuntimeError('The update archive contains an unsafe path.')
                # GitHub archives should not contain symbolic links. Reject them so
                # an archive can never write through a link during extraction.
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RuntimeError('The update archive contains an unsupported link.')
            zip_file.extractall(extract_dir)

        roots = [path for path in extract_dir.iterdir() if path.is_dir()]
        if len(roots) != 1 or not (roots[0] / 'src').is_dir():
            raise RuntimeError('The downloaded archive is not a valid ProphitBet update.')

        archive_path.unlink(missing_ok=True)
        if progress is not None:
            progress(100)
        return roots[0]
    except Exception:
        shutil.rmtree(update_dir, ignore_errors=True)
        raise


def launch_staged_update(source: Path, target: Path) -> None:
    """Start a detached helper which overlays the update after this process exits."""

    update_dir = source.parent.parent
    helper = update_dir / 'apply_update.py'
    shutil.copy2(Path(__file__), helper)

    if getattr(sys, 'frozen', False):
        restart_command = [sys.executable, *sys.argv[1:]]
    else:
        current_entrypoint = Path(sys.argv[0]).resolve()
        downloaded_entrypoint = source / current_entrypoint.name
        if not downloaded_entrypoint.is_file() and (source / 'app.py').is_file():
            downloaded_entrypoint = source / 'app.py'
        entrypoint = str(target / downloaded_entrypoint.name)
        restart_command = [sys.executable, entrypoint, *sys.argv[1:]]

    specification = update_dir / 'update.json'
    specification.write_text(
        json.dumps({
            'pid': os.getpid(),
            'source': str(source.resolve()),
            'target': str(target.resolve()),
            'restart': restart_command,
        }),
        encoding='utf-8',
    )

    kwargs = {
        'cwd': str(update_dir),
        'stdin': subprocess.DEVNULL,
        'stdout': subprocess.DEVNULL,
        'stderr': subprocess.DEVNULL,
        'close_fds': True,
    }
    if os.name == 'nt':
        kwargs['creationflags'] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        kwargs['start_new_session'] = True

    subprocess.Popen(
        [sys.executable, str(helper), '--apply-update', str(specification)],
        **kwargs,
    )


def _process_exists(pid: int) -> bool:
    if os.name == 'nt':
        import ctypes

        synchronize = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def apply_update(specification_path: Path) -> int:
    """Helper-process entry point: wait, overlay all files, and restart the app."""

    specification = json.loads(specification_path.read_text(encoding='utf-8'))
    pid = int(specification['pid'])
    source = Path(specification['source'])
    target = Path(specification['target'])

    for _ in range(300):
        if not _process_exists(pid):
            break
        time.sleep(0.1)
    else:
        return 1

    try:
        shutil.copytree(source, target, dirs_exist_ok=True, copy_function=shutil.copy2)
        subprocess.Popen(
            specification['restart'],
            cwd=str(target),
            close_fds=True,
        )
        return 0
    finally:
        # The helper lives here too. Cleanup is best-effort, especially on Windows.
        shutil.rmtree(source.parent.parent, ignore_errors=True)


if __name__ == '__main__' and len(sys.argv) == 3 and sys.argv[1] == '--apply-update':
    raise SystemExit(apply_update(Path(sys.argv[2])))
