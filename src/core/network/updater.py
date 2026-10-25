"""Download, install, verify, and restart ProphitBet updates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Callable, Optional
from urllib.request import Request, urlopen
from zipfile import ZipFile


UPDATE_ARCHIVE_URL = (
    'https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/'
    'archive/refs/heads/main.zip'
)


@dataclass(frozen=True)
class UpdateResult:
    """Details needed by the GUI after a successful in-place update."""

    root: Path
    entrypoint: Path
    file_count: int


def project_root() -> Path:
    """Find the project root without depending on a fixed package depth."""

    module_path = Path(__file__).resolve()
    for candidate in (module_path.parent, *module_path.parents):
        has_entrypoint = (candidate / 'app.py').is_file() or (candidate / 'main.py').is_file()
        if has_entrypoint and (candidate / 'requirements.txt').is_file():
            return candidate
    raise RuntimeError('Could not locate the ProphitBet application root.')


def _report(progress: Optional[Callable[[int], None]], value: int) -> None:
    if progress is not None:
        progress(max(0, min(100, value)))


def stage_update(progress: Optional[Callable[[int], None]] = None) -> Path:
    """Download and safely extract the complete GitHub ``main`` archive."""

    update_dir = Path(tempfile.mkdtemp(prefix='prophitbet-update-'))
    archive_path = update_dir / 'main.zip'
    request = Request(UPDATE_ARCHIVE_URL, headers={'User-Agent': 'ProphitBet-Updater'})
    _report(progress, 0)

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
                if total:
                    _report(progress, int(downloaded * 40 / total))
        _report(progress, 40)

        extract_dir = update_dir / 'extracted'
        extract_dir.mkdir()
        with ZipFile(archive_path) as zip_file:
            members = zip_file.infolist()
            if not members:
                raise RuntimeError('The downloaded update archive is empty.')
            corrupt_member = zip_file.testzip()
            if corrupt_member is not None:
                raise RuntimeError(f'The downloaded archive is corrupt at {corrupt_member}.')

            extract_root = extract_dir.resolve()
            for member in members:
                destination = (extract_dir / member.filename).resolve()
                if extract_root != destination and extract_root not in destination.parents:
                    raise RuntimeError('The update archive contains an unsafe path.')
                file_type = (member.external_attr >> 16) & 0o170000
                if file_type == 0o120000:
                    raise RuntimeError('The update archive contains an unsupported link.')
            zip_file.extractall(extract_dir)

        roots = [path for path in extract_dir.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError('The update archive has an unexpected directory structure.')

        source = roots[0]
        has_entrypoint = (source / 'app.py').is_file() or (source / 'main.py').is_file()
        if not has_entrypoint or not (source / 'requirements.txt').is_file():
            raise RuntimeError('The archive is not a complete ProphitBet main branch.')

        archive_path.unlink(missing_ok=True)
        _report(progress, 50)
        return source
    except Exception:
        shutil.rmtree(update_dir, ignore_errors=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_file(source: Path, destination: Path) -> None:
    """Copy one file beside its target and atomically replace the target."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{destination.name}.update-',
        dir=str(destination.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        if destination.exists():
            destination.chmod(destination.stat().st_mode | stat.S_IWRITE)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _select_entrypoint(source: Path, target: Path) -> Path:
    current_name = Path(sys.argv[0]).name
    for name in (current_name, 'app.py', 'main.py'):
        if name and (source / name).is_file():
            return target / name
    raise RuntimeError('The update does not contain a runnable application entry point.')


def _normalized_relative_path(path: Path, source: Path) -> Path:
    """Normalize accidental edge whitespace in GitHub path components."""

    parts = tuple(part.strip() for part in path.relative_to(source).parts)
    if not parts or any(not part or part in {'.', '..'} for part in parts):
        raise RuntimeError(f'The update contains an invalid path: {path}')
    return Path(*parts)


def apply_staged_update(
        source: Path,
        target: Path,
        progress: Optional[Callable[[int], None]] = None,
) -> UpdateResult:
    """Overlay and verify every directory and file contained in the archive."""

    source = source.resolve()
    target = target.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise RuntimeError('The staged update and application folders overlap.')

    directories = sorted(path for path in source.rglob('*') if path.is_dir())
    files = sorted(path for path in source.rglob('*') if path.is_file())
    if not files:
        raise RuntimeError('The staged update contains no files.')

    entrypoint = _select_entrypoint(source, target)
    destinations = [_normalized_relative_path(path, source) for path in files]
    if len(destinations) != len(set(destinations)):
        raise RuntimeError('The update contains file paths that normalize to the same destination.')

    for directory in directories:
        (target / _normalized_relative_path(directory, source)).mkdir(parents=True, exist_ok=True)

    for index, (source_file, relative) in enumerate(zip(files, destinations), start=1):
        destination = target / relative
        _replace_file(source_file, destination)
        _report(progress, 50 + int(index * 40 / len(files)))

    # Verify every downloaded file, rather than checking only a few root files.
    for index, (source_file, relative) in enumerate(zip(files, destinations), start=1):
        destination = target / relative
        if not destination.is_file() or _sha256(source_file) != _sha256(destination):
            raise RuntimeError(f'Update verification failed for {relative}.')
        _report(progress, 90 + int(index * 10 / len(files)))

    _report(progress, 100)
    return UpdateResult(root=target, entrypoint=entrypoint, file_count=len(files))


def update_from_github(progress: Optional[Callable[[int], None]] = None) -> UpdateResult:
    """Install all files from GitHub ``main`` into the application root."""

    source = stage_update(progress)
    update_dir = source.parent.parent
    try:
        return apply_staged_update(source, project_root(), progress)
    finally:
        shutil.rmtree(update_dir, ignore_errors=True)


def restart_application(result: UpdateResult) -> None:
    """Start the updated application in a new detached process."""

    if not result.entrypoint.is_file():
        raise RuntimeError(f'Updated entry point not found: {result.entrypoint}')

    if getattr(sys, 'frozen', False):
        command = [sys.executable, *sys.argv[1:]]
    else:
        command = [sys.executable, str(result.entrypoint), *sys.argv[1:]]

    options = {
        'cwd': str(result.root),
        'stdin': subprocess.DEVNULL,
        'stdout': subprocess.DEVNULL,
        'stderr': subprocess.DEVNULL,
        'close_fds': True,
    }
    if os.name == 'nt':
        options['creationflags'] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        options['start_new_session'] = True

    subprocess.Popen(command, **options)
