"""Fail CI on credential-shaped values; print filenames only, never secrets."""
import pathlib
import re
import subprocess
import sys

patterns = [re.compile(rb'vck_[A-Za-z0-9]{20,}'),
            re.compile(rb'gh[pousr]_[A-Za-z0-9]{20,}'),
            re.compile(rb'sk-(?:proj-)?[A-Za-z0-9_-]{24,}'),
            re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]
files = subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0')
failed = []
for name in filter(None, files):
    path = pathlib.Path(name)
    if path.name == '.env' or (path.name.startswith('.env.') and path.name != '.env.example'):
        failed.append(name)
        continue
    if path.is_file() and path.stat().st_size < 2_000_000:
        data = path.read_bytes()
        if any(pattern.search(data) for pattern in patterns):
            failed.append(name)
if failed:
    print('Potential credentials found; review without printing values:')
    print('\n'.join(failed))
    sys.exit(1)
print('Credential-pattern checks passed (not a complete security audit).')
