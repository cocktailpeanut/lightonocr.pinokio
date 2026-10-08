# LightOnOCR for Pinokio

Private tested candidate: Linux CPU image/PDF OCR and launcher lifecycle pass. macOS verification is coordinated separately. Windows and CUDA have not been tested. This is a community launcher, not an official LightOn product.

A small localhost-only image/PDF OCR app for **LightOnOCR-3-1B**, using its official Transformers inference classes. Upload PNG, JPEG, WebP or PDF; receive Markdown, copy it or download a `.md` file. Plain transcription and grounding output are supported. No vLLM, Docker, WSL or hosted OCR service is required.

## Install and run

1. Clone this repository into Pinokio (private repository access is required).
2. Click **Install**. The first install downloads PyTorch, dependencies and approximately 2 GB of model weights. Allow roughly 8 GB of free disk space for CPU/Mac, more for CUDA wheels.
3. Click **Start**, wait for the model-ready message, then **Open Web UI**.
4. Upload a document and click Extract. CPU inference can take several minutes. Large or dense pages take longer.
5. Use Pinokio's Stop control to release model memory.

The server binds to `127.0.0.1` on an available port. Model load completes before readiness is announced. Uploads remain in memory, are not logged or uploaded to a third-party OCR API, and are not deliberately saved. Model/dependency downloads contact Hugging Face and package registries. Do not expose this local app through a tunnel or public proxy.

## Backends

| Platform | Installation target | Runtime | Verification |
|---|---|---|---|
| Linux x86-64, no supported GPU | CPU wheels | CPU bfloat16 | Image/PDF and lifecycle pass |
| Apple Silicon macOS | macOS wheels | MPS float32 | In progress |
| Windows x86-64, no NVIDIA GPU | CPU wheels | CPU bfloat16 | Not tested |
| Linux/Windows NVIDIA | CUDA 12.8 wheels | CUDA bfloat16, or float16 on older hardware | Not tested |
| Intel Mac | Unsupported by pinned PyTorch wheel | — | Installer rejects |
| AMD GPU | CPU fallback | CPU | Not tested |

See [TESTING.md](TESTING.md) for reproducible commands and measured results.

A recent NVIDIA driver compatible with CUDA 12.8 is required for CUDA. A GPU is optional. Plan on at least 8 GB RAM for CPU/Mac, with more free memory for larger pages; these are initial estimates pending measurements. This launcher does not install drivers. `LIGHTONOCR_DEVICE=cpu|cuda|mps|auto` can select a backend; `LIGHTONOCR_DTYPE=float32` provides a CPU fallback if needed. See the server's `/healthz` for actual backend, dtype and versions.

## Maintenance

- **Install / repair** is repeatable and reuses downloaded weights.
- **Update** fast-forwards the Git checkout, then reinstalls pinned dependencies/model. It does not follow a floating model revision.
- **Reset environment** removes only `app/env` and `app/.installed`; it retains weights and source. Stop the server first. Reinstall afterward.
- Weights live in `app/models/lightonocr`. Environments, weights and runtime files are excluded from Git.

Limits: one OCR job at a time, 25 MB upload, at most 20 PDF pages, maximum 4096 generated tokens per page. A token-limit warning means output may be incomplete. AI OCR can make mistakes: check important text, figures, tables and numbers against the original.

## Reproducibility and provenance

- Model: [lightonai/LightOnOCR-3-1B](https://huggingface.co/lightonai/LightOnOCR-3-1B), revision `b9a2b4c17f1eee9f29058d716b66b5f8e7d8db86`.
- Official upstream client reference: [lightonai/LightOnOCR](https://github.com/lightonai/LightOnOCR/tree/36755d461be079737860a5f03ae0c803501269e9), commit `36755d461be079737860a5f03ae0c803501269e9`. This launcher does not install that vLLM-based client.
- PyTorch `2.10.0`, torchvision `0.25.0`, Transformers `5.16.1`; direct app dependencies pinned in `app/requirements.txt`. Transitive dependencies are resolved by pip and are not fully locked across platforms.
- The pinned legacy checkpoint uses an explicit Transformers weight-name conversion; startup refuses incomplete or mismatched weights.
- Python 3.10–3.13. Pinokio's managed Python is used to create the app virtual environment.
- PDF rendering uses pypdfium2, with longest edge 1540 pixels, preserving aspect ratio.

## License

Launcher/app code is Apache-2.0 (see LICENSE). LightOnOCR upstream code and the pinned model weights are Apache-2.0 per their respective upstream repository/model card. Downloaded dependencies retain their own licenses; PyTorch/torchvision are BSD-style, Transformers Apache-2.0, FastAPI MIT, Uvicorn BSD-3-Clause, Pillow HPND, and pypdfium2 Apache-2.0/BSD-3-Clause with PDFium's third-party notices. No model weights are redistributed in this repository.
