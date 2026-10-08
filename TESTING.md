# Validation record

Private build tested on 2026-10-08. Windows and CUDA are not tested. macOS validation is being coordinated separately.

## Linux CPU

Actual model inference uses LightOnOCR-3-1B revision `b9a2b4c17f1eee9f29058d716b66b5f8e7d8db86`; no substitute OCR engine or mock model is used by `tests/test_inference.py`.

- Pinokiod 8.2.2 / pterm 0.0.25 / Node 22.23.3.
- Native `install.js` successfully runs bootstrap through Pinokio's managed Conda base; `start.js` invokes the isolated environment’s Python directly and reports its dynamic loopback URL after loading.
- Main environment: Python 3.12.14, torch 2.10.0+cpu, torchvision 0.25.0, Transformers 5.16.1, CPU bfloat16.
- Fresh install in an application path containing spaces: passed, using managed Python 3.13.13. Model weights were reused via local hard links in this isolated test; the initial real install downloaded the pinned 2.02 GB model.
- All 532 checkpoint tensors load with zero missing, unexpected or mismatched keys.
- PNG OCR: passed in 18.3 seconds. PDF OCR: passed in 29.9 seconds. Both returned the four fixture lines, 30 tokens, no truncation.
- Assertions check `12345`, `Apples`, `Total`, and `3.00`, plus one page and non-truncated output.
- Stop: passed; the app's HTTP listener disappeared.
- Repeated native install: passed, with existing dependencies and model reused (0 downloaded model bytes).
- Stop/restart: passed, followed by another real PNG/PDF assertion pass.
- Spaced-path startup and real PNG/PDF inference: passed. PNG 17.6 seconds; PDF 31.1 seconds. This fresh environment used Python 3.13.13.
- Both test apps were stopped after validation; environments and weights are retained for reproduction.
- Observed main-server process peak resident memory during restart testing: approximately 2.60 GiB (2,723,900 KiB). This is a fixture measurement, not a maximum for all documents.

### Compatibility issue found and fixed

The pinned model card's bare `from_pretrained` example is insufficient with Transformers 5.16.1: its checkpoint uses legacy `language_model.model.*` keys. The first integration run loaded randomly initialized text-model parameters and failed OCR assertions. The app now explicitly maps this prefix to `model.language_model.*` using Transformers' supported `key_mapping` option. It checks loading information and refuses readiness if any weights are missing, unexpected or mismatched. No model weight files are modified.

### Paths containing spaces

Pinokiod 8.2.2 emitted an unquoted virtual-environment activation command for a spaced application path. The launcher now invokes the quoted relative `env/bin/python` (or `env\Scripts\python.exe` on Windows) directly through native `shell.run`. It also clears stale readiness URLs before starting and only saves a captured readiness URL.

### Cloud test environment caveats

The isolated cloud daemon needs a separate loopback-only preload because sandbox network-interface enumeration is unavailable. Pinokiod package source is unmodified. Its default native Miniforge download path could not resolve GitHub through the cloud proxy, so the same official Miniforge 26.3.2-3 installer was fetched with curl and installed to Pinokio's managed directory. The environment's existing trusted system CA bundle was supplied via an untracked ENVIRONMENT file for HTTPS proxy trust; certificate verification was never disabled. These cloud-specific settings are not shipped in the launcher.

## Reproduce

Start this app through Pinokio, then run from its repository directory:

```
app/env/bin/python tests/test_inference.py http://127.0.0.1:PORT
```

Windows:

```
app\env\Scripts\python.exe tests\test_inference.py http://127.0.0.1:PORT
```

Use the actual URL from Pinokio. The test generates its own PNG/PDF, performs real HTTP OCR, and saves results to ignored `tests/results/inference.json`. It allows up to 30 minutes per request for slower CPUs.

API-only unit tests:

```
app/env/bin/python tests/test_api.py
```

20 tests pass, covering bounded uploads, PDF pages, errors, concurrency, origin/host protection, CSP, transparent-image preprocessing and raw-text output. These tests explicitly use a fake engine and provide no model-inference evidence.

Static launcher-contract checks: `node tests/test_launcher.js` (backend wheel selection, Windows/POSIX executable quoting, readiness capture, menu states and exact reset scope). These are not Windows/CUDA runtime tests.
