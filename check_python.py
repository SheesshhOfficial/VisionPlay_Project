import struct
import sys
if not ((3, 10) <= sys.version_info[:2] <= (3, 12) and struct.calcsize('P') == 8):
    print('Use 64-bit Python 3.10, 3.11 or 3.12 (3.11 recommended).')
    raise SystemExit(1)
if '--venv' in sys.argv and sys.prefix == sys.base_prefix:
    print('This interpreter is not an isolated virtual environment. Rerun setup.bat.')
    raise SystemExit(1)
print('Compatible Python:', sys.version.split()[0])
