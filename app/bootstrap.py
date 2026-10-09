"""Install dependencies and viewer assets; download models only on explicit selection."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import venv

from model_catalog import MODELS, download_model, get_model

ROOT = Path(__file__).resolve().parent
TORCH_VERSION = '2.10.0'

def run(args):
    print('+ ' + ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True, cwd=ROOT)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--model', choices=list(MODELS))
    args = parser.parse_args()
    if args.download_only:
        if not args.model:
            parser.error('--download-only requires an explicitly selected --model.')
        download_model(get_model(args.model))
        return
    if sys.version_info < (3, 10) or sys.version_info >= (3, 14):
        raise SystemExit('Python 3.10–3.13 is required. Update Pinokio and its managed Python.')
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise SystemExit('This pinned PyTorch release supports Apple Silicon Macs, not Intel Macs.')
    marker = ROOT / '.installed'
    marker.unlink(missing_ok=True)
    env = ROOT / 'env'
    python = env / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        print('Creating isolated Python environment...', flush=True)
        venv.EnvBuilder(with_pip=True).create(env)
    run([python, '-m', 'pip', 'install', '--upgrade', 'pip==26.0.1', '--index-url', 'https://pypi.org/simple'])
    command = [python, '-m', 'pip', 'install', f'torch=={TORCH_VERSION}', 'torchvision==0.25.0']
    if platform.system() != 'Darwin':
        command += ['--index-url', 'https://download.pytorch.org/whl/' + ('cu128' if args.device == 'cuda' else 'cpu')]
    else:
        command += ['--index-url', 'https://pypi.org/simple']
    run(command)
    run([python, '-m', 'pip', 'install', '-r', ROOT / 'requirements.txt', '--index-url', 'https://pypi.org/simple'])
    run([python, '-m', 'pip', 'check'])
    run([python, ROOT / 'frontend_assets.py'])
    marker.write_text(json.dumps({'models': list(MODELS), 'device_wheels': args.device,
                                  'python': sys.version, 'torch': TORCH_VERSION}, indent=2), encoding='utf-8')
    print('LIGHTONOCR_INSTALL_COMPLETE', flush=True)

if __name__ == '__main__':
    main()
