# Original-viewer validation record

This integration replaces the earlier custom UI with upstream LightOnOCR's original viewer, pinned at `36755d461be079737860a5f03ae0c803501269e9`, plus a local Transformers OpenAI-compatible adapter. No vLLM is installed or used.

Runtime-tested commit: `6f521f837a6a9c27a130924df32d7a9f1d2f27b3`. Subsequent documentation-only changes do not alter that runtime.

## Linux CPU and cloud-browser results (2026-10-08)

- Real startup passed on Linux CPU bfloat16: all 532 checkpoint tensors loaded, no missing/unexpected/mismatched keys; internal `/v1/models` returned the pinned model before viewer readiness.
- Real PNG upload through the original `/api/runs` API in grounding mode passed in 25.175 seconds. All four fixture strings were present, and the original parser produced labeled bounding-box blocks. Source/page images/Markdown/grounding JSON were persisted by the original pipeline.
- The initial standalone PDF API test was interrupted to release model RAM for the browser demonstration; it is not counted as a PDF API pass.
- Actual original-browser PNG upload → Run OCR → completed result passed, with all four expected text strings and four visible grounding boxes. This PNG selection used automation file input, not the native chooser.
- A separate single-page PDF browser test passed through the native desktop chooser: New run → choose fixture.pdf in the native dialog → Open → filename remained selected → Run OCR → completed result with all four expected strings and grounding boxes. This completed at 21:37:26 UTC.
- The user-controlled remote picker still fails in a flow where agent-operated native input succeeds. Its cause is unconfirmed. These tests do not establish that the user's picker issue is fixed or that the handed-off remote-control experience is fully verified.
- A separate three-page, 49,672-byte sample PDF completed end-to-end on cloud CPU bfloat16 after native chooser selection → Open → Run OCR. It ran from 21:40:11 to 21:59:31 UTC (approximately 19 minutes 20 seconds).
- Its source PDF and all three page PNG, Markdown and grounding JSON files were verified nonempty. Metadata records grounding mode and pages `[1, 2, 3]`; the parsed page results contain 10, 10 and 7 blocks. The first page identifies the sample PDF; later pages contain plausible sample text.
- This establishes completion for that cloud CPU three-page document, not a transcription-accuracy benchmark or a general performance guarantee. Mac multipage processing remains untested. The separate user-controlled remote-picker issue remains unresolved.
- 14 fake-engine adapter tests pass, including the unchanged official OpenAI client, plain/grounding, sampling, usage/truncation, strict checkpoint loading guard, JSON-safe health, request bounds, cancellation and serialized inference.
- 63 fake-engine original-viewer HTTP assertions pass, covering original routes/assets, local KaTeX fonts, upload/jobs/history, plain/grounding/selected PDF pages, source/results storage, deletion, host/origin protection and size/page limits. These do not prove model accuracy.
- Python compilation and JavaScript syntax pass.
- Windows and CUDA runtime remain untested. Previous custom-UI results are not used to claim original-viewer coverage.

## Apple Silicon macOS MPS results (2026-10-08)

Validated the exact original-viewer runtime commit `6f521f837a6a9c27a130924df32d7a9f1d2f27b3`:

- Native Chrome file-chooser selection passed for PNG and PDF, without file injection; selected filenames remained attached before submission.
- Both browser grounding runs returned the expected fixture text and four bounding boxes in the original viewer.
- Real API OCR passed before and after stop/restart; saved runs persisted.
- Update/repair passed. Inference used Transformers on MPS float32, with no vLLM.
- The app was stopped after validation and both viewer/adapter ports closed. Installed environment, weights and launcher icon were retained.
- Pinokio embedded/popup file picking, Mac CPU fallback and multipage PDFs were not tested.

These results are specific to native Chrome on the tested Mac; they do not resolve the separate remote-control picker issue described above.

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

The original-viewer candidate reuses the installed pinned model and CPU environment. Its new OpenAI client and integrity-checked local browser libraries were installed separately. A complete new native Linux launcher lifecycle retest remains pending; the separate Mac original-viewer lifecycle results are recorded above.
