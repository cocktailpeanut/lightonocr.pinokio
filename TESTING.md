# Original-viewer validation record

This integration replaces the earlier custom UI with upstream LightOnOCR's original viewer, pinned at `36755d461be079737860a5f03ae0c803501269e9`, plus a local Transformers OpenAI-compatible adapter. No vLLM is installed or used.

The earlier platform sections record the previous 1B-only runtime. The initial three-model Mac section records model/runtime coverage; the native menu regression section records the corrected Electron button path.

Linux/Mac runtime-tested commit: `6f521f837a6a9c27a130924df32d7a9f1d2f27b3`. Windows CUDA tested that same runtime with the browser-asset path normalization included in this revision. Those reported platform checks predate the three-model changes.

## Three-model Apple Silicon macOS validation (2026-10-09 UTC)

Validated the local three-model candidate based on `6a090830fa2b1cbd930903e1613cde778f9b3870`, using macOS 26.3.1(a), Apple M1 Max (10 CPU/32 GPU cores), 64 GiB unified memory and native Pinokio 8.2.0. Runtime: Python 3.10.20, PyTorch 2.10.0, Transformers 5.16.1 and MPS float32. Initial starts and stops used the actual Pinokio `start.js` through a browser-served menu or managed pterm. Those checks did not establish the native Electron menu dispatch path for all variants.

- Actual `install.js` completed with `pip check` and local browser assets. The app remained offline with no default Start target. Only the preexisting 1B cache was present; neither 0.8B nor 4B was downloaded by installation.
- Opening the live launcher displayed **Start 0.8B**, **Start 1B** and **Start 4B** without automatically starting any of them. The browser-served 1B menu entry passed; 0.8B and 4B passed through the same script with pterm model arguments. This missed a native menu/tab collision subsequently reported by the user.
- Selecting 1B reused its original `app/models/lightonocr` cache with 0.00B downloaded. Selecting 0.8B downloaded only its independent cache; 4B remained absent until selected. The three caches retain their pinned revisions.
- Strict loading passed for all three models: 532 tensors for 1B, 473 for 0.8B and 723 for 4B, with no missing, unexpected or mismatched keys. The official Qwen3.5 architecture loaded for 0.8B/4B; the original LightOnOCR architecture and legacy key mapping loaded for 1B.

| Selected model | PNG grounding API | PDF plain API | Native Chrome PNG grounding | Fresh PNG plain OCR after cached restart |
|---|---|---|---|---|
| 1B | Pass, 7.046 s | Pass, 6.513 s | Pass | Pass, 4.029 s |
| 0.8B | Pass, 88.470 s | Pass, 250.781 s | Pass | Pass, 22.076 s |
| 4B | Pass, 15.096 s | Pass, 22.337 s | Pass | Pass, 8.048 s |

These are elapsed synthetic fixture job checks, not speed or accuracy benchmarks. The initial 0.8B PDF job queued behind its native PNG run, so that duration includes queue wait. All API fixture outputs contained `12345`, `Apples`, `Total` and `3.00`; each grounding PNG had four parsed boxes.

For each model, native Chrome testing used the visible macOS file chooser, without file injection: **New run → choose fixture.png → Open → filename remained selected → Run OCR → completed result**. The document image, four boxes and corresponding text were visibly verified and captured. Native PDF chooser testing in this three-model candidate was not run; earlier 1B PDF chooser evidence is recorded separately below.

Cached stop/start passed for all three variants. Each restart downloaded 0.00B, retained saved history including all three native test runs, reported the expected active model and completed new real OCR. All checkpoint sizes and modification times were unchanged. A live runtime lock prevented a second process from acquiring the app lock; model-selection contracts also verify rejection before any second download or load. The live 0.8B endpoint listed only 0.8B and rejected an attempt to switch to 4B through viewer settings.

Every tested stop closed both viewer and adapter ports. The final app state was offline, with no ready URL, no default target and both final listeners closed. The environment, all model caches and saved runs were retained. The upstream viewer files, **LightOnOCR 3** title and official avatar were unchanged.

Six fake-loader selection tests, 14 adapter tests, 63 viewer HTTP assertions, launcher contracts, Python compilation and JavaScript syntax passed. These contract tests are separate from the real-model results above. Three-model Windows/Linux operation, Intel Mac, Mac CPU fallback, multipage PDFs and Pinokio embedded/popup picking were not run in this validation.

## Native Pinokio model-button regression (2026-10-09 UTC)

The user reported that clicking 4B launched 0.8B. Pinokio 8.2 strips query parameters from the default script-tab target, resolves the first link with that target for callbacks, and reuses native hard-tab frames by script pathname. The former three choices all used `start.js?model=...`, giving them the same script/tab identity. Earlier pterm inference checks bypassed this native dispatch behavior and did not prove it worked.

Each choice now uses its own explicit async entry point: `start-08b.js`, `start-1b.js`, or `start-4b.js`. Each binds a fixed variant through the shared `start.js` launch logic. Running/readiness menus follow the actual entry script. Model caches, strict loaders, runtime lock, upstream viewer, title and icon are retained. No custom native tab ID or query argument is needed to select a model.

An intermediate wrapper failed visibly with a disconnected terminal and no launch steps: Pinokio only awaits runners declared `async`, and supplies `kernel.info` as its second runner argument. Both contracts are corrected and covered by the launcher tests. That failed attempt is not an inference pass.

The final native Electron window passed the sequence below without page refreshes between selections. Each entry was clicked visibly; where a retained stopped terminal showed **Run**, that native Run control was also clicked. Every start reported source `ui`, the correct script/process arguments, pinned revision, architecture, MPS float32 and an exact strict weight load. `/v1/models` contained only the selected checkpoint. The original viewer opened automatically with the matching ready URL and active model.

| Native selection order | Entry script | Checkpoint tensors | Fresh fixture PNG plain OCR |
|---|---|---|---|
| 4B | start-4b.js | 723 | Pass, 6.028 s |
| 0.8B | start-08b.js | 473 | Pass, 24.077 s |
| 1B | start-1b.js | 532 | Pass, 4.018 s |
| 4B, retained terminal | start-4b.js | 723 | Pass, 8.039 s |

Fresh OCR was submitted through the original viewer API after each native-button launch. Each output contained all four expected fixture strings. Every native Stop closed both viewer/adapter ports, cleared readiness/local state and released the runtime lock. All four starts reused cached checkpoints with 0.00B downloaded; previous documents and runs remained present. Screenshots, process/API/checkpoint records, startup logs, OCR outputs and stop records are stored locally under `artifacts/mac-three-models/menu-regression` in the validation workspace.

Actual Pinokio `start.js` calls with no model and with an invalid model also passed rejection checks. Each new pterm stream recorded Python rejecting `--model invalid` before any download/load or readiness marker. Both tests ended offline; no silent fallback model started and checkpoint sizes/modification times were unchanged. Final launcher contracts and all six model-selection tests passed.

These final native dispatch results do not establish Windows/Linux three-model operation or native embedded file-picker compatibility.

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
- Windows CUDA results are recorded separately below. Linux CUDA remains untested. Previous custom-UI results are not used to claim original-viewer coverage.

## Apple Silicon macOS MPS results (2026-10-08)

Validated the exact original-viewer runtime commit `6f521f837a6a9c27a130924df32d7a9f1d2f27b3`:

- Native Chrome file-chooser selection passed for PNG and PDF, without file injection; selected filenames remained attached before submission.
- Both browser grounding runs returned the expected fixture text and four bounding boxes in the original viewer.
- Real API OCR passed before and after stop/restart; saved runs persisted.
- Update/repair passed. Inference used Transformers on MPS float32, with no vLLM.
- The app was stopped after validation and both viewer/adapter ports closed. Installed environment, weights and launcher icon were retained.
- Pinokio embedded/popup file picking, Mac CPU fallback and multipage PDFs were not tested.

These results are specific to native Chrome on the tested Mac; they do not resolve the separate remote-control picker issue described above.

## Windows CUDA results (2026-10-08)

- Hardware: NVIDIA RTX A4500, 19190 MiB VRAM, driver 596.86; Windows 10.0.26200.9457, Pinokio 8.2.2.
- The unmodified viewer installer failed twice because `str(Path(...))` recorded `katex\\katex.min.js` on Windows while required-asset validation expected `katex/katex.min.js`. The file had been extracted correctly. Recording `target.as_posix()` fixes validation and makes new cache manifests portable; integrity verification is unchanged.
- Corrected native installation through Pinokio/pterm completed, including pinned browser libraries, model download and `pip check`. No manual asset bypass was used.
- Native startup passed: Transformers on CUDA bfloat16, Python 3.10.20, torch 2.10.0+cu128, Transformers 5.16.1, OpenAI client 3.26.1. All 532 checkpoint tensors loaded with no missing/unexpected/mismatched keys.
- Real original-viewer API PNG grounding passed in 7.232 seconds and PDF transcription in 3.154 seconds. Both contained the four fixture strings; PNG produced four parsed layout boxes.
- Browser New run → choose generated PNG → Run OCR passed. Document image, rendered text, boxes, linked text selection, Raw, label hide/show and zoom/Fit were checked. File selection used the browser automation chooser, so this does not establish Pinokio embedded/popup picker compatibility.
- Native stop removed the listener. Repeated native installation reused Marked, KaTeX and DOMPurify, downloaded 0.00B model data, and retained model sizes/modification times.
- Restart/history persistence passed. Real OCR passed again: PNG 7.232 seconds, PDF 3.199 seconds. Both runs reported `REAL_ORIGINAL_VIEWER_IMAGE_PDF_GROUNDING_PASS`.
- All 63 fake-engine viewer assertions, 14 adapter tests and launcher contracts passed. The corrected app was left online for the user; saved runs were retained.
- Windows paths containing spaces and Linux CUDA remain outside these CUDA results. The separate Windows CPU results below do not imply testing a machine without an NVIDIA GPU.

## Windows CPU results (2026-10-08)

- A separate Pinokio app with its own environment and model cache exercised the published runtime on Windows 10.0.26200.9457. The host exposes four cores/eight logical processors of an AMD EPYC 7543P and also has an NVIDIA GPU.
- A test-only copy of `install.js` forced its existing CPU branch instead of NVIDIA detection; `LIGHTONOCR_DEVICE=cpu` selected runtime CPU. The test harness is not shipped. This validates CPU-only installation/inference on this host, not automatic selection on GPU-free hardware.
- Native Pinokio/pterm installation completed with torch 2.10.0+cpu, Python 3.10.20, Transformers 5.16.1 and OpenAI client 3.26.1. No CUDA wheels or shared environment/model junctions were used.
- Unmodified `start.js` loaded all 532 tensors on CPU bfloat16, with no missing/unexpected/mismatched keys. The actual local Transformers adapter responded before viewer readiness.
- Included real original-viewer API tests passed: PNG grounding in 107.111 seconds, PDF transcription in 249.925 seconds, all four expected fixture strings and four PNG bounding boxes; success marker `REAL_ORIGINAL_VIEWER_IMAGE_PDF_GROUNDING_PASS`.
- The API-created PNG result was opened in the browser; document image, four boxes and rendered text were visibly verified. CPU browser upload/picker execution was not tested.
- Native stop removed the listener. Repeated native CPU installation passed `pip check`, reused all three browser libraries and downloaded 0.00B model data; model sizes and modification times were unchanged. Restart readiness, strict CPU-only backend health and persistence of both saved results passed. Real CPU OCR was not repeated after restart; the separate CUDA lifecycle test repeated both fixtures.

## Browser-asset regression checks

`python tests/test_frontend_assets.py` uses small in-memory archives and temporary directories; it downloads no packages or models. It checks native extraction and portable cache manifests, download-free repeated installation, recovery of a missing cached font, rejection of tampered bytes and rejection of missing required assets. On Windows the original implementation fails the installation/cache tests; the fixed implementation passes all four tests.

The `Browser asset installation` GitHub Actions workflow runs these checks on Windows, Linux and macOS with Python 3.10 and 3.13. This is asset-installer coverage, not model inference, GPU or complete launcher lifecycle coverage. CI outcomes must be read from the workflow run rather than inferred from this matrix definition.

All six jobs passed for fix commit `753b9bccafa22d1b7855291b7c7d2ff3413b4c52`: [workflow run 37857584552](https://github.com/cocktailpeanut/lightonocr.pinokio/actions/runs/37857584552).

## Reproduce

After installation and Pinokio startup, run from the repository directory:

```
app/env/bin/python tests/test_inference.py http://127.0.0.1:PORT
```

Windows uses `app\env\Scripts\python.exe`. The real-model test generates a non-sensitive PNG/PDF, submits original viewer jobs, polls them, asserts fixture text and grounding boxes, and writes ignored `tests/results/inference.json`. Its success marker is `REAL_ORIGINAL_VIEWER_IMAGE_PDF_GROUNDING_PASS`.

Also test the actual original browser UI: New run, choose a file, Run, wait, open the completed run, verify document image/rendered text/layout boxes, hover/click linking, Raw, filters and zoom.

Unit/contract tests (explicit fake engines, no model-inference evidence):

```
app/env/bin/python tests/test_models.py
app/env/bin/python tests/test_adapter.py
app/env/bin/python tests/test_viewer.py
app/env/bin/python tests/test_frontend_assets.py
node tests/test_launcher.js
```

## Preserved safeguards and known boundaries

- 1B legacy checkpoint prefix conversion is preserved. The 0.8B/4B models use their official Qwen3.5 loader, without that conversion. Every model has an exact weight-loading guard.
- Installation downloads no checkpoint and selects no default Start entry. Explicit starts use independent caches; the legacy 1B cache and saved runs are retained. A process lock prevents a second runtime from preparing or loading another model.
- Quoted relative Python startup preserves the earlier spaced-path launcher fix.
- Original upstream rendering uses local Marked/KaTeX plus DOMPurify sanitization. CSS/layout/interaction design are retained.
- The original viewer persists documents and output in app/out; it is not an in-memory-only service.
- Uploads are capped at 25 MB, PDFs at 20 pages. Model inference and viewer jobs are serial.
- Direct adapter callers receive OpenAI `finish_reason=length` for truncated output. The original viewer does not expose that field; review dense documents for completeness at its 2048-token default.

## Linux cloud environment caveats

Earlier native Pinokiod 8.2.2 tests used a separate loopback-only preload because sandbox interface enumeration was unavailable. Pinokiod source was unmodified. Its Miniforge downloader could not resolve GitHub through the cloud proxy, so the same official Miniforge26.3.2-3 installer was fetched using curl into Pinokio's managed directory. The existing trusted system CA bundle was supplied through an ignored ENVIRONMENT file; certificate verification was never disabled. These cloud-specific settings are not shipped.

The original-viewer candidate reuses the installed pinned model and CPU environment. Its new OpenAI client and integrity-checked local browser libraries were installed separately. A complete new native Linux launcher lifecycle retest remains pending; the separate Mac original-viewer lifecycle results are recorded above.
