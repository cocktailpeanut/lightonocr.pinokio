"""Real-model HTTP integration check; run against the Pinokio-started app."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import httpx

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('url', help='http://127.0.0.1:PORT from the launcher')
parser.add_argument('--timeout', type=float, default=1800)
args = parser.parse_args()
url = args.url.rstrip('/')
if not url.startswith('http://127.0.0.1:'):
    parser.error('Use the loopback URL reported by Pinokio.')
subprocess.run([sys.executable, str(ROOT / 'make_fixtures.py')], check=True)
results = {}
with httpx.Client(timeout=args.timeout, trust_env=False) as client:
    health = client.get(url + '/healthz')
    health.raise_for_status()
    results['health'] = health.json()
    print(json.dumps(results['health'], indent=2), flush=True)
    response = client.get(url + '/')
    assert response.status_code == 200 and 'Extract text' in response.text
    for name, mime in [('fixture.png', 'image/png'), ('fixture.pdf', 'application/pdf')]:
        started = time.perf_counter()
        with (ROOT / name).open('rb') as document:
            response = client.post(url + '/api/ocr', files={'file': (name, document, mime)},
                                   data={'mode': 'plain', 'max_new_tokens': '256'})
        response.raise_for_status()
        result = response.json()
        result['http_elapsed_seconds'] = round(time.perf_counter() - started, 3)
        text = result['markdown'].lower()
        assert result['page_count'] == 1, result
        assert not result['truncated'], result
        for expected in ['12345', 'apples', 'total', '3.00']:
            assert expected in text, (name, expected, result)
        results[name] = result
        print(name, json.dumps(result, indent=2), flush=True)
output = ROOT / 'results'
output.mkdir(exist_ok=True)
(output / 'inference.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print('REAL_IMAGE_AND_PDF_OCR_PASS', flush=True)
