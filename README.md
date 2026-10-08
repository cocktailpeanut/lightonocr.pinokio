# LightOnOCR for Pinokio

The **original LightOnOCR web viewer**, connected to a local **Transformers** backend. **No vLLM is installed or used.** This is a community launcher, not an official LightOn product.

The original viewer includes persistent runs, document images with linked layout boxes, rendered Markdown/HTML tables/math, label filters, page navigation, zoom and raw output. Its CSS, layout and interaction design come from the pinned official upstream source; this is not a replacement frontend.

## Install and run

1. Clone this private repository into Pinokio and click **Install**. Existing installations must run **Update / repair** after upgrading to this viewer.
2. Installation downloads pinned PyTorch/dependencies, approximately 2 GB of model weights, and small local browser libraries. No npm install scripts or vLLM dependencies are used.
3. Click **Start**, wait for model readiness, then **Open LightOnOCR**.
4. Click **New run**, choose a PDF/image and select grounding (text and layout) or plain transcription. Click **Run**. Progress and completed runs appear in the sidebar.
5. Open a completed run to inspect the document and rendered output. Use **Raw** for the model's original text.
6. Stop through Pinokio when finished to release model memory.

The viewer and internal compatibility API bind only to `127.0.0.1`, on dynamic ports. Readiness requires correctly loaded weights, a responsive adapter and a bound viewer. Model settings are restricted to this launcher's local pinned model; external OCR endpoints are disabled.

## Local storage and limits

Unlike the earlier custom UI, the original viewer **saves uploads and OCR results locally** under `app/out/<run>/`: source document, page images, Markdown, grounding JSON and metadata. Runs persist across restarts and are excluded from Git. The viewer can delete runs. Resetting dependencies retains these documents and model weights.

Documents are not sent to a remote OCR provider. Installation contacts Hugging Face and package registries. The viewer's rendering libraries are served locally; no CDN connection is needed to display results. Do not expose this unauthenticated local app through a public tunnel.

Uploads: 25 MB maximum, PDFs with 1–20 pages, rendered longest edge 1540 px. Jobs and pages are processed serially for this single local model. The compatibility API accepts 128–4096 output tokens; the original viewer uses the adapter's 2048-token default. Dense output can reach that limit, so check important documents against the source. OCR can misread text, figures, numbers and layout.

## Backends and validation

| Platform | Runtime | Current original-viewer verification |
|---|---|---|
| Linux x86-64 CPU | CPU bfloat16 | Real PNG grounding through original run/job API passed; browser/PDF checks pending |
| Apple Silicon macOS | MPS float32 | Retest pending for original viewer |
| Windows x86-64 CPU | CPU bfloat16 | Not tested |
| Linux/Windows NVIDIA | CUDA 12.8, bfloat16 or float16 | Not tested |
| Intel Mac | Unsupported by pinned PyTorch wheel | Installer rejects |
| AMD GPU | CPU fallback | Not tested |

Previous custom-UI Linux/Mac results do not validate this new viewer integration. See [TESTING.md](TESTING.md) for current evidence and limitations.

Plan on at least 8 GB RAM and roughly 8 GB disk for CPU/Mac, with more room for CUDA wheels and saved documents. NVIDIA needs a driver compatible with CUDA 12.8; the launcher does not install drivers. `LIGHTONOCR_DEVICE=cpu|cuda|mps|auto` selects the device; `LIGHTONOCR_DTYPE=float32` is available for CPU compatibility. `/healthz` reports the actual backend and versions.

## Windows manual validation (not yet tested)

These are manual instructions, not automated CI:

1. Run Install and Start through Windows Pinokio.
2. In the repository directory run, replacing `PORT` with the viewer port:

   ```
   app\env\Scripts\python.exe tests\test_inference.py http://127.0.0.1:PORT
   ```

3. Require `REAL_ORIGINAL_VIEWER_IMAGE_PDF_GROUNDING_PASS`. This checks actual PNG grounding and PDF transcription through the original viewer API and saves ignored results under `tests/results`.
4. Also test **New run → choose file → Run** in the browser, then verify the document image, text, layout boxes and raw view. API tests alone do not establish browser usability.
5. Stop/restart and repeat OCR; repeat Install to verify cached reuse. Record `/healthz`, then stop the app. CPU success does not validate CUDA.

## Maintenance

- Install/repair reuses weights and installs pinned dependencies and local browser assets.
- Update fast-forwards the checkout, then reruns installation.
- Reset removes only `app/env` and `app/.installed`. It retains `app/models`, `app/out`, and source. Stop first, then reinstall.

## Pinned source and licenses

- Original viewer/client: [lightonai/LightOnOCR](https://github.com/lightonai/LightOnOCR/tree/36755d461be079737860a5f03ae0c803501269e9), commit `36755d461be079737860a5f03ae0c803501269e9`, Apache-2.0. Vendored source and license are under `app/vendor`.
- Model: [lightonai/LightOnOCR-3-1B](https://huggingface.co/lightonai/LightOnOCR-3-1B), revision `b9a2b4c17f1eee9f29058d716b66b5f8e7d8db86`, Apache-2.0. Weights are not redistributed in Git.
- PyTorch 2.10.0, torchvision 0.25.0, Transformers 5.16.1, OpenAI client 3.26.1; direct dependencies in `app/requirements.txt`. Transitive dependencies are not fully locked across platforms.
- The pinned checkpoint needs explicit legacy weight-name mapping. Startup rejects any missing, unexpected or mismatched tensors.
- Marked 14.1.3 and KaTeX 0.16.11 match upstream. DOMPurify 3.4.16 sanitizes model-generated HTML. SHA-512-verified npm archives are downloaded by `app/frontend_assets.py`; bundled licenses are preserved with the installed libraries.
- Launcher/adapter code: Apache-2.0 (LICENSE). Dependencies retain their own licenses.

Narrow upstream frontend changes replace CDN URLs with local assets, sanitize rendered HTML and escape model-provided labels. Original viewer styling/features are retained. See `app/vendor/UPSTREAM.md`.
