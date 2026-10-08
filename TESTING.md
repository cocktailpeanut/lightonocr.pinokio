# Original-viewer validation record

This integration replaces the earlier custom UI with upstream LightOnOCR's original viewer, pinned at `36755d461be079737860a5f03ae0c803501269e9`, plus a local Transformers OpenAI-compatible adapter. No vLLM is installed or used.

## Current results (2026-10-08)

- Real startup passed on Linux CPU bfloat16: all 532 checkpoint tensors loaded, no missing/unexpected/mismatched keys; internal `/v1/models` returned the pinned model before viewer readiness.
- Real PNG upload through the original `/api/runs` API in grounding mode passed in 25.175 seconds. All four fixture strings were present, and the original parser produced labeled bounding-box blocks. Source/page images/Markdown/grounding JSON were persisted by the original pipeline.
- The subsequent PDF test was interrupted to release model RAM for the user's browser demonstration. This run is not counted as a PDF pass.
- Actual browser upload → run → document/layout/rendered result validation is pending. Serving HTML or passing an API test is not called a browser pass.
- 14 fake-engine adapter tests pass, including the unchanged official OpenAI client, plain/grounding, sampling, usage/truncation, strict checkpoint loading guard, JSON-safe health, request bounds, cancellation and serialized inference.
- 63 fake-engine original-viewer HTTP assertions pass, covering original routes/assets, local KaTeX fonts, upload/jobs/history, plain/grounding/selected PDF pages, source/results storage, deletion, host/origin protection and size/page limits. These do not prove model accuracy.
- Python compilation and JavaScript syntax pass.
- New-viewer Mac, Windows and CUDA runtime tests are pending. Previous custom-UI Mac/Linux tests do not validate this changed integration.

## Reproduce

After installation and Pinokio startup, run from the repository directory:

```
app/env/bin/python tests/test_inference.py http://127.0.0.1:PORT
```

Windows uses `app\env\Scripts\python.exe`. The real-model test generates a non-sensitive PNG/PDF, submits original viewer jobs, polls them, asserts fixture text and grounding boxes, and writes ignored `tests/results/inference.json`. Its success marker is `REAL_ORIGINAL_VIEWER_IMAGE_PDF_GROUNDING_PASS`.

Also test the actual original browser UI: New run, choose a file, Run, wait, open the completed run, verify document image/rendered text/layout boxes, hover/click linking, Raw, filters and zoom.

Unit/contract tests (explicit fake engines, no model-inference evidence):

```
app/env/bin/python tests/test_adapter.py
app/env/bin/python tests/test_viewer.py
node tests/test_launcher.js
```

## Preserved safeguards and known boundaries

- Legacy checkpoint prefix conversion is preserved, with an exact weight-loading guard.
- Quoted relative Python startup preserves the earlier spaced-path launcher fix.
- Original upstream rendering uses local Marked/KaTeX plus DOMPurify sanitization. CSS/layout/interaction design are retained.
- The original viewer persists documents and output in app/out; it is not an in-memory-only service.
- Uploads are capped at 25 MB, PDFs at 20 pages. Model inference and viewer jobs are serial.
- Direct adapter callers receive OpenAI `finish_reason=length` for truncated output. The original viewer does not expose that field; review dense documents for completeness at its 2048-token default.

## Linux cloud environment caveats

Earlier native Pinokiod 8.2.2 tests used a separate loopback-only preload because sandbox interface enumeration was unavailable. Pinokiod source was unmodified. Its Miniforge downloader could not resolve GitHub through the cloud proxy, so the same official Miniforge26.3.2-3 installer was fetched using curl into Pinokio's managed directory. The existing trusted system CA bundle was supplied through an ignored ENVIRONMENT file; certificate verification was never disabled. These cloud-specific settings are not shipped.

The original-viewer candidate reuses the installed pinned model and CPU environment. Its new OpenAI client and integrity-checked local browser libraries were installed separately. A complete new native launcher lifecycle retest remains pending.
