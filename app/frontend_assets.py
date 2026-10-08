"""Install pinned, integrity-checked original-viewer browser dependencies locally."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'vendor' / 'lightonocr' / 'static' / 'lib'
PACKAGES = {
    'marked': ('14.1.3', 'sha512-ZibJqTULGlt9g5k4VMARAktMAjXoVnnr+Y3aCqW1oDftcV4BA3UmrBifzXoZyenHRk75csiPu9iwsTj4VNBT0g=='),
    'katex': ('0.16.11', 'sha512-RQrI8rlHY92OLf3rho/Ts8i/XvjgguEjOkO1BEXcU3N8BqPpSzBNwV/G0Ukr+P/l3ivvJUE/Fa/CwbS6HesGNQ=='),
    'dompurify': ('3.4.16', 'sha512-sqo+pNp3qRhCIpbgRi1y8Tgk27Bo2Ry7w0dC1NBeNTdZChWjz9Xb/KOoZbRP/R6pQZ80Qw8YhXw13hWWBbMRnQ=='),
}

def install_assets():
    DEST.mkdir(parents=True, exist_ok=True)
    for name, (version, integrity) in PACKAGES.items():
        marker = DEST / f'.{name}.json'
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved.get('integrity') == integrity and all((DEST / p).is_file() for p in saved.get('files', [])):
                print(f'Viewer library {name} {version} already installed.', flush=True)
                continue
        print(f'Downloading viewer library {name} {version}...', flush=True)
        url = f'https://registry.npmjs.org/{name}/-/{name}-{version}.tgz'
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read(10 * 1024 * 1024 + 1)
        if len(data) > 10 * 1024 * 1024:
            raise RuntimeError(f'Oversized archive: {name}')
        digest = 'sha512-' + base64.b64encode(hashlib.sha512(data).digest()).decode()
        if digest != integrity:
            raise RuntimeError(f'Integrity check failed: {name}')
        written = []
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                source = member.name
                target = None
                if name == 'katex' and source.startswith('package/dist/'):
                    target = Path('katex') / source.removeprefix('package/dist/')
                elif name == 'marked' and source == 'package/marked.min.js':
                    target = Path('marked.min.js')
                elif name == 'dompurify' and source == 'package/dist/purify.min.js':
                    target = Path('purify.min.js')
                elif Path(source).name.upper().startswith('LICENSE') and len(Path(source).parts) == 2:
                    target = Path(f'{name}-LICENSE.txt')
                if target is None:
                    continue
                if target.is_absolute() or '..' in target.parts:
                    raise RuntimeError('Unsafe asset path in archive')
                destination = DEST / target
                destination.parent.mkdir(parents=True, exist_ok=True)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError('Missing asset archive member')
                destination.write_bytes(extracted.read())
                written.append(str(target))
        expected = {'marked': 'marked.min.js', 'katex': 'katex/katex.min.js', 'dompurify': 'purify.min.js'}[name]
        if expected not in written:
            raise RuntimeError(f'Missing required viewer asset: {expected}')
        marker.write_text(json.dumps({'integrity': integrity, 'files': written}), encoding='utf-8')

if __name__ == '__main__':
    install_assets()
