"""Offline browser-asset installation regression tests; no OCR or model downloads."""
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    'frontend_assets', Path(__file__).resolve().parents[1] / 'app' / 'frontend_assets.py')
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


def archive_bytes(files):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return output.getvalue()


class AssetInstallation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dest = Path(self.temp.name) / 'lib'
        self.archives = {
            'marked': archive_bytes({'package/marked.min.js': b'marked', 'package/LICENSE.md': b'license'}),
            'katex': archive_bytes({
                'package/dist/katex.min.js': b'katex',
                'package/dist/katex.min.css': b'css',
                'package/dist/fonts/KaTeX_Main-Regular.woff2': b'font',
                'package/LICENSE': b'license'}),
            'dompurify': archive_bytes({'package/dist/purify.min.js': b'purify', 'package/LICENSE': b'license'}),
        }
        self.packages = {name: ('test', 'sha512-' + base64.b64encode(
            hashlib.sha512(data).digest()).decode()) for name, data in self.archives.items()}
        self.addCleanup(patch.stopall)
        patch.object(assets, 'DEST', self.dest).start()
        patch.object(assets, 'PACKAGES', self.packages).start()
        self.download = patch.object(assets.urllib.request, 'urlopen', side_effect=self.urlopen).start()

    def urlopen(self, url, timeout):
        self.assertEqual(timeout, 90)
        name = url.split('/')[3]
        return io.BytesIO(self.archives[name])

    def test_install_and_cached_reuse(self):
        # Executes real extraction and validation on each runner's native filesystem.
        # str(Path(...)) on Windows fails before the required KaTeX check completes.
        assets.install_assets()
        self.assertEqual(self.download.call_count, 3)
        self.assertEqual((self.dest / 'katex/katex.min.js').read_bytes(), b'katex')
        self.assertEqual((self.dest / 'katex/fonts/KaTeX_Main-Regular.woff2').read_bytes(), b'font')
        self.assertEqual((self.dest / 'purify.min.js').read_bytes(), b'purify')
        for name in self.packages:
            self.assertTrue((self.dest / f'{name}-LICENSE.txt').is_file())
            saved = json.loads((self.dest / f'.{name}.json').read_text())
            self.assertTrue(saved['files'])
            self.assertTrue(all('\\' not in entry for entry in saved['files']))
        self.download.reset_mock()
        assets.install_assets()
        self.download.assert_not_called()

    def test_missing_cached_font_is_reinstalled(self):
        assets.install_assets()
        (self.dest / 'katex/fonts/KaTeX_Main-Regular.woff2').unlink()
        self.download.reset_mock()
        assets.install_assets()
        self.assertEqual(self.download.call_count, 1)
        self.assertEqual((self.dest / 'katex/fonts/KaTeX_Main-Regular.woff2').read_bytes(), b'font')

    def test_integrity_failure_does_not_install(self):
        self.archives['marked'] += b'tampered'
        with self.assertRaisesRegex(RuntimeError, 'Integrity check failed: marked'):
            assets.install_assets()
        self.assertFalse((self.dest / 'marked.min.js').exists())
        self.assertFalse((self.dest / '.marked.json').exists())

    def test_missing_required_asset_does_not_mark_complete(self):
        self.archives['katex'] = archive_bytes({'package/dist/katex.min.css': b'css'})
        self.packages['katex'] = ('test', 'sha512-' + base64.b64encode(
            hashlib.sha512(self.archives['katex']).digest()).decode())
        with self.assertRaisesRegex(RuntimeError, 'Missing required viewer asset: katex/katex.min.js'):
            assets.install_assets()
        self.assertFalse((self.dest / '.katex.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
