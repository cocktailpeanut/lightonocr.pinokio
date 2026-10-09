# LightOnOCR for Pinokio

The **original LightOnOCR web viewer**, connected to a local **Transformers** backend. **No vLLM is installed or used.** This is a community launcher, not an official LightOn product.

The original viewer includes persistent runs, document images with linked layout boxes, rendered Markdown/HTML tables/math, label filters, page navigation, zoom and raw output. Its CSS, layout and interaction design come from the pinned official upstream source; this is not a replacement frontend.

## Install and run

1. Clone this repository into Pinokio and click **Install**. Existing installations must run **Update / repair** after upgrading to this viewer.
2. Install sets up pinned PyTorch/dependencies and small local browser libraries, then finishes without starting or downloading a model. Existing model caches and saved runs are retained. No npm install scripts or vLLM dependencies are used.
3. Choose **Start 0.8B**, **Start 1B** or **Start 4B**. The first launch downloads only that selected checkpoint; later launches reuse its cache. Wait for model readiness, then **Open LightOnOCR**.
4. Click **New run**, choose a PDF/image and select grounding (text and layout) or plain transcription. Click **Run**. Progress and completed runs appear in the sidebar.
5. Open a completed run to inspect the document and rendered output. Use **Raw** for the model's original text.
6. Stop through Pinokio when finished to release model memory. To switch models, stop the running app, then choose another Start entry. One model runs at a time.

The viewer and internal compatibility API bind only to `127.0.0.1`, on dynamic ports. Readiness requires correctly loaded weights, a responsive adapter and a bound viewer. The original viewer shows the active model. Its settings accept only the running checkpoint; select a different model through the Pinokio Start choices after stopping. External OCR endpoints are disabled.

## Local storage and limits

Unlike the earlier custom UI, the original viewer **saves uploads and OCR results locally** under `app/out/<run>/`: source document, page images, Markdown, grounding JSON and metadata. Runs persist across restarts and are excluded from Git. The viewer can delete runs. Resetting dependencies retains these documents and model weights.

Documents are not sent to a remote OCR provider. Installation contacts Hugging Face and package registries. The viewer's rendering libraries are served locally; no CDN connection is needed to display results. Do not expose this unauthenticated local app through a public tunnel.

Uploads: 25 MB maximum, PDFs with 1–20 pages, rendered longest edge 1540 px for 1B and 2048 px for 0.8B/4B. Jobs and pages are processed serially for the selected local model. The compatibility API accepts 128–4096 output tokens; the original viewer uses the adapter's 2048-token default. Dense output can reach that limit, so check important documents against the source. OCR can misread text, figures, numbers and layout.

## Backends and validation

The three-model runtime passed Apple Silicon Mac validation (M1 Max, 64 GiB; Pinokio 8.2.0, MPS float32): dependency-only install with no autostart, strict loading, real PNG grounding/PDF transcription, native Chrome PNG chooser/rendering, and cached restarts. A reported native menu collision was then fixed with separate async model entry scripts and verified through the actual Electron buttons in the order 4B → 0.8B → 1B → 4B, with fresh OCR and clean stops each time. A retained stopped terminal may show **Run** after selecting its model; click Run to start that selected checkpoint. The app was left offline with all caches and documents retained. Three-model Windows/Linux checks remain unrun. See [TESTING.md](TESTING.md) for exact scope and fixture timings.

| Platform | Runtime | Earlier 1B original-viewer verification |
|---|---|---|
| Linux x86-64 CPU | CPU bfloat16 | Real PNG API/browser and single-/three-page PDF native-chooser flows passed; remote-control picker issue remains |
| Apple Silicon macOS | MPS float32 | Native Chrome PNG/PDF chooser and rendering, API OCR, restart/history and Update/repair passed |
| Windows x86-64 CPU | CPU bfloat16 | CPU-only wheels: install/reinstall, PNG/PDF API OCR, rendered grounding and restart/history passed; forced CPU selection on an NVIDIA host |
| Windows NVIDIA | CUDA 12.8 bfloat16 | Native install, PNG/PDF API OCR, browser grounding, restart/history and repeated install passed |
| Linux NVIDIA | CUDA 12.8, bfloat16 or float16 | Not tested |
| Intel Mac | Unsupported by pinned PyTorch wheel | Installer rejects |
| AMD GPU | CPU fallback | Not tested |

The platform results above cover the earlier 1B-only runtime. The new three-model selection and 0.8B/4B checkpoints have not been validated on Windows or Linux. Linux/Mac original-viewer checks used runtime commit `6f521f837a6a9c27a130924df32d7a9f1d2f27b3`; Windows CUDA checks used that runtime with the Windows browser-asset path fix now included here. Older custom-UI results are separate. The user-controlled remote file picker still has an unresolved issue even though agent-operated native chooser tests passed. A cloud CPU three-page sample completed in approximately 19 minutes 20 seconds; this is one completion test, not an accuracy benchmark or general speed claim. Pinokio embedded/popup picking, Mac CPU fallback and Mac multipage-PDF processing remain unverified. See [TESTING.md](TESTING.md) for the exact test scope.

Memory and storage requirements grow with the selected model, dtype and document size. Weight downloads are approximately 1.7 GB (0.8B), 2 GB (1B) and 9 GB (4B); MPS uses float32, which needs more model memory than the downloaded weights. Keep additional room for environments, cached variants and saved documents. NVIDIA needs a driver compatible with CUDA 12.8; the launcher does not install drivers. `LIGHTONOCR_DEVICE=cpu|cuda|mps|auto` selects the device; `LIGHTONOCR_DTYPE=float32` is available for CPU compatibility. `/healthz` reports the actual backend and versions.

## Windows validation and reproduction

Windows CUDA validation passed on an NVIDIA RTX A4500 after normalizing cached browser-asset paths. A separate CPU-only installation on the same Windows host passed PNG grounding and PDF transcription in 107.111 and 249.925 seconds. The following steps reproduce the real-model checks. The separate CI matrix passed offline asset extraction and cache reuse on Windows, Linux and macOS with Python 3.10 and 3.13; it does not run OCR or validate accelerators.

1. Run Install, then choose one of the explicit Start model entries through Windows Pinokio.
2. In the repository directory run, replacing `PORT` with the viewer port:

   ```
   app\env\Scripts\python.exe tests\test_inference.py http://127.0.0.1:PORT
   ```

3. Require `REAL_ORIGINAL_VIEWER_IMAGE_PDF_GROUNDING_PASS`. This checks actual PNG grounding and PDF transcription through the original viewer API and saves ignored results under `tests/results`.
4. Also test **New run → choose file → Run** in the browser, then verify the document image, text, layout boxes and raw view. API tests alone do not establish browser usability.
5. Stop/restart and repeat OCR; repeat Install to verify cached reuse. Record `/healthz`, then stop the app. CPU success does not validate CUDA.

## Maintenance

- Install/repair installs pinned dependencies and local browser assets without starting a model. Selected-model startup reuses its existing weights.
- Update fast-forwards the checkout, then reruns installation.
- Reset removes only `app/env` and `app/.installed`. It retains `app/models`, `app/out`, and source. Stop first, then reinstall.

## Pinned source and licenses

- Original viewer/client: [lightonai/LightOnOCR](https://github.com/lightonai/LightOnOCR/tree/36755d461be079737860a5f03ae0c803501269e9), commit `36755d461be079737860a5f03ae0c803501269e9`, Apache-2.0. Vendored source and license are under `app/vendor`.
- Models (Apache-2.0): [0.8B](https://huggingface.co/lightonai/LightOnOCR-3-0.8B) at `4a953edfc77f0e435532c503dd69dd74663d44c3`; [1B](https://huggingface.co/lightonai/LightOnOCR-3-1B) at `b9a2b4c17f1eee9f29058d716b66b5f8e7d8db86`; [4B](https://huggingface.co/lightonai/LightOnOCR-3-4B) at `a06e5c5459551c9d1696468aece70ffbcf01ae62`. Weights are not redistributed in Git. Each has an independent cache under `app/models`; 1B retains the existing `lightonocr` directory.
- PyTorch 2.10.0, torchvision 0.25.0, Transformers 5.16.1, OpenAI client 3.26.1; direct dependencies in `app/requirements.txt`. Transitive dependencies are not fully locked across platforms.
- 1B uses `LightOnOcrProcessor`/`LightOnOcrForConditionalGeneration` with its required legacy weight-name mapping. The official 0.8B/4B checkpoints use `AutoProcessor`/`Qwen3_5ForConditionalGeneration`. Every model must load without missing, unexpected or mismatched tensors.
- Marked 14.1.3 and KaTeX 0.16.11 match upstream. DOMPurify 3.4.16 sanitizes model-generated HTML. SHA-512-verified npm archives are downloaded by `app/frontend_assets.py`; bundled licenses are preserved with the installed libraries.
- Launcher/adapter code: Apache-2.0 (LICENSE). Dependencies retain their own licenses.

Narrow upstream frontend changes replace CDN URLs with local assets, sanitize rendered HTML and escape model-provided labels. Original viewer styling/features are retained. See `app/vendor/UPSTREAM.md`.
