"""Download only the official Google-hosted task bundle; no shell installers."""
from pathlib import Path
import hashlib
import urllib.request
import zipfile
URL = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'
SHA256 = '97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482'
TARGET = Path(__file__).resolve().parent / 'models' / 'gesture_recognizer.task'
def valid(path):
    return (path.exists() and zipfile.is_zipfile(path)
            and hashlib.sha256(path.read_bytes()).hexdigest() == SHA256)
def main():
    TARGET.parent.mkdir(exist_ok=True)
    if not valid(TARGET):
        temp = TARGET.with_suffix('.download')
        try:
            print('Downloading official Google model (~8 MB)...')
            with urllib.request.urlopen(URL, timeout=60) as source, temp.open('wb') as dest:
                while block := source.read(1024 * 1024): dest.write(block)
            if not valid(temp): raise RuntimeError('Downloaded file is not a valid model bundle.')
            temp.replace(TARGET)
        finally:
            temp.unlink(missing_ok=True)
    print('Model ready. SHA256 (local record):', hashlib.sha256(TARGET.read_bytes()).hexdigest())
if __name__ == '__main__':
    try: main()
    except Exception as exc:
        print('Model download failed:', exc)
        print('Check internet access then rerun setup.bat. No model means no camera demo.')
        raise SystemExit(1)
