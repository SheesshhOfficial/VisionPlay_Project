import struct
import sys
if not ((3, 10) <= sys.version_info[:2] <= (3, 12) and struct.calcsize('P') == 8):
    print('Use 64-bit Python 3.10, 3.11 or 3.12 (3.11 recommended).')
    raise SystemExit(1)
print('Compatible Python:', sys.version.split()[0])
