"""Real-model tests through the ORIGINAL upstream viewer run/job API (not browser UI tests)."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
import httpx

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('url', help='http://127.0.0.1:PORT reported by Pinokio')
parser.add_argument('--timeout', type=float, default=1800)
args = parser.parse_args()
url = args.url.rstrip('/')
if not url.startswith('http://127.0.0.1:'):
    parser.error('Use the loopback URL from Pinokio.')
subprocess.run([sys.executable, str(ROOT / 'make_fixtures.py')], check=True)
results = {}
with httpx.Client(timeout=30, trust_env=False) as client:
    health = client.get(url + '/healthz'); health.raise_for_status()
    results['health'] = health.json()
    print(json.dumps(results['health'], indent=2), flush=True)
    response = client.get(url + '/')
    assert response.status_code == 200 and 'new-run-dialog' in response.text
    for asset in ['/static/app.js', '/static/app.css', '/static/lib/marked.min.js', '/static/lib/purify.min.js', '/static/lib/katex/katex.min.js']:
        assert client.get(url + asset).status_code == 200, asset
    settings = client.get(url + '/api/settings'); settings.raise_for_status()
    assert settings.json()['reachable'] and settings.json()['grounding'], settings.text
    for filename, mode in [('fixture.png', 'grounding'), ('fixture.pdf', 'plain')]:
        started = time.perf_counter()
        name = f'integration-{mode}-{uuid.uuid4().hex[:8]}'
        response = client.post(url + '/api/runs', params={'name':name, 'mode':mode},
                               headers={'X-Filename':filename, 'Content-Type':'application/octet-stream'},
                               content=(ROOT / filename).read_bytes())
        response.raise_for_status(); job = response.json()
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            jobs = client.get(url + '/api/jobs'); jobs.raise_for_status()
            job = next(item for item in jobs.json() if item['id'] == job['id'])
            if job['status'] in ['done', 'error']:
                break
            time.sleep(1)
        assert job['status'] == 'done', job
        response = client.get(url + '/api/runs/' + name); response.raise_for_status()
        result = response.json(); assert result['page_count'] == 1, result
        page = result['pages'][0]; text = page['raw'].lower()
        for expected in ['12345','apples','total','3.00']:
            assert expected in text, (filename,expected,result)
        assert client.get(url + page['image']).status_code == 200
        if mode == 'grounding':
            assert page['blocks'], result
            assert all(len(block['bbox']) == 4 for block in page['blocks']), result
        result['http_elapsed_seconds'] = round(time.perf_counter() - started, 3)
        results[filename] = result
        print(filename, json.dumps(result, indent=2), flush=True)
output = ROOT / 'results'; output.mkdir(exist_ok=True)
(output / 'inference.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print('REAL_ORIGINAL_VIEWER_IMAGE_PDF_GROUNDING_PASS',flush=True)
