"""Loopback-only, in-memory LightOnOCR 3 server. Run bootstrap.py before use."""
from __future__ import annotations

import argparse
import io
import logging
import math
import os
import secrets
import sys
import threading
import time
import warnings
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.middleware.trustedhost import TrustedHostMiddleware

MODEL_ID = "lightonai/LightOnOCR-3-1B"
MODEL_REVISION = "b9a2b4c17f1eee9f29058d716b66b5f8e7d8db86"
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models" / "lightonocr"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 64 * 1024
MAX_PAGES = 20
LONGEST_EDGE = 1540
Image.MAX_IMAGE_PIXELS = 40_000_000
logger = logging.getLogger("lightonocr")
inference_lock = threading.Lock()


class MemoryMultipartParser(MultiPartParser):
    # Both attribute names cover Starlette's old and new spool-size naming.
    # The entire request is bounded before parsing, so uploads never spill to disk.
    spool_max_size = MAX_REQUEST_BYTES + 1
    max_file_size = MAX_REQUEST_BYTES + 1


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


class Engine:
    def __init__(self) -> None:
        import torch
        import transformers
        from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

        self.torch = torch
        requested = os.getenv("LIGHTONOCR_DEVICE", "auto").strip().lower()
        if requested not in {"auto", "cpu", "cuda", "mps"}:
            raise RuntimeError("LIGHTONOCR_DEVICE must be auto, cpu, cuda, or mps.")
        cuda = torch.cuda.is_available()
        mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        self.device = ("cuda" if cuda else "mps" if mps else "cpu") if requested == "auto" else requested
        if self.device == "cuda" and not cuda:
            raise RuntimeError("CUDA was requested but is unavailable. Use LIGHTONOCR_DEVICE=cpu or auto.")
        if self.device == "mps" and not mps:
            raise RuntimeError("Apple MPS was requested but is unavailable. Use LIGHTONOCR_DEVICE=cpu or auto.")
        dtype_name = os.getenv("LIGHTONOCR_DTYPE", "auto").strip().lower()
        if dtype_name == "auto":
            if self.device == "mps":
                dtype_name = "float32"
            elif self.device == "cuda":
                dtype_name = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            else:
                dtype_name = "bfloat16"
        if dtype_name not in {"float32", "float16", "bfloat16"}:
            raise RuntimeError("LIGHTONOCR_DTYPE must be auto, float32, float16, or bfloat16.")
        if self.device == "mps" and dtype_name != "float32":
            raise RuntimeError("MPS requires LIGHTONOCR_DTYPE=float32 for this model.")
        if self.device == "cpu" and dtype_name == "float16":
            raise RuntimeError("Use bfloat16 or float32 on CPU.")
        self.dtype_name = dtype_name
        self.dtype = getattr(torch, dtype_name)
        try:
            threads = int(os.getenv("LIGHTONOCR_CPU_THREADS", "4"))
        except ValueError as exc:
            raise RuntimeError("LIGHTONOCR_CPU_THREADS must be an integer from 1 to 32.") from exc
        if not 1 <= threads <= 32:
            raise RuntimeError("LIGHTONOCR_CPU_THREADS must be an integer from 1 to 32.")
        torch.set_num_threads(min(threads, os.cpu_count() or 1))
        if not (MODEL_DIR / "config.json").is_file():
            raise RuntimeError("Model is not installed. Run the launcher Install action first.")
        logger.info("Loading %s on %s (%s) from local files", MODEL_ID, self.device, dtype_name)
        self.processor = LightOnOcrProcessor.from_pretrained(str(MODEL_DIR), local_files_only=True)
        # This pinned checkpoint retains the legacy nested CausalLM key prefix.
        # Transformers' native LightOnOcr class embeds the base language model.
        self.model, loading = LightOnOcrForConditionalGeneration.from_pretrained(
            str(MODEL_DIR), dtype=self.dtype, local_files_only=True,
            key_mapping={r"^language_model\.model\.": "model.language_model."},
            output_loading_info=True,
        )
        failures = {key: values for key, values in loading.items() if values}
        if failures:
            raise RuntimeError(f"Checkpoint weights did not load exactly; refusing to start: {failures}")
        logger.info("All checkpoint weights loaded; no missing, unexpected, or mismatched keys")
        self.model = self.model.to(self.device).eval()
        self.versions = {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers.__version__, "fastapi": package_version("fastapi"),
            "pypdfium2": package_version("pypdfium2"),
        }

    def health(self) -> dict[str, Any]:
        return {"status": "ready", "backend": self.device, "device": self.device,
                "dtype": self.dtype_name, "model": MODEL_ID, "revision": MODEL_REVISION,
                "versions": self.versions, "busy": inference_lock.locked(),
                "limits": {"upload_mb": 25, "pdf_pages": MAX_PAGES, "longest_edge": LONGEST_EDGE}}

    def transcribe(self, image: Image.Image, mode: str, max_new_tokens: int) -> tuple[str, int, bool]:
        content: list[dict[str, Any]] = [{"type": "image", "image": image}]
        if mode == "grounding":
            content.append({"type": "text", "text": "grounding"})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        )
        inputs = {key: value.to(device=self.device, dtype=self.dtype)
                  if value.is_floating_point() else value.to(self.device)
                  for key, value in inputs.items()}
        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.1)
        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        token_count = int(generated.shape[0])
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        eos_ids = set(eos if isinstance(eos, (tuple, list)) else [eos])
        ended = token_count > 0 and int(generated[-1]) in eos_ids
        text = self.processor.decode(generated, skip_special_tokens=True)
        return text, token_count, token_count >= max_new_tokens and not ended


def normalize_image(image: Image.Image) -> Image.Image:
    transposed = ImageOps.exif_transpose(image)
    try:
        if transposed.mode in {"RGBA", "LA"} or "transparency" in transposed.info:
            rgba = transposed.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            try:
                result = Image.alpha_composite(background, rgba).convert("RGB")
            finally:
                rgba.close()
                background.close()
        else:
            result = transposed.convert("RGB")
    finally:
        transposed.close()
    result.thumbnail((LONGEST_EDGE, LONGEST_EDGE), Image.Resampling.LANCZOS)
    return result


def perform_ocr(engine: Engine, data: bytes, mode: str, max_new_tokens: int) -> dict[str, Any]:
    started = time.perf_counter()
    output: list[str] = []
    pages: list[dict[str, Any]] = []

    def page_ocr(image: Image.Image, number: int, total: int) -> None:
        logger.info("OCR page %d/%d started", number, total)
        page_start = time.perf_counter()
        markdown, tokens, truncated = engine.transcribe(image, mode, max_new_tokens)
        elapsed = time.perf_counter() - page_start
        output.append(markdown)
        pages.append({"page": number, "seconds": round(elapsed, 3),
                      "generated_tokens": tokens, "truncated": truncated})
        logger.info("OCR page %d/%d completed in %.1fs%s", number, total, elapsed,
                    " (token limit reached)" if truncated else "")

    if b"%PDF-" in data[:1024]:
        import pypdfium2 as pdfium
        pdf = None
        try:
            pdf = pdfium.PdfDocument(data)
            count = len(pdf)
            if count < 1 or count > MAX_PAGES:
                raise HTTPException(400, f"PDFs must contain between 1 and {MAX_PAGES} pages.")
            for index in range(count):
                page = bitmap = image = None
                try:
                    page = pdf[index]
                    width, height = page.get_size()
                    if not all(math.isfinite(value) and value > 0 for value in (width, height)):
                        raise HTTPException(400, "The PDF contains an invalid page size.")
                    # Cap the 200-DPI render at a 1540px longest edge.
                    scale = min(200 / 72, LONGEST_EDGE / max(width, height))
                    bitmap = page.render(scale=scale)
                    rendered = bitmap.to_pil()
                    try:
                        image = normalize_image(rendered)
                    finally:
                        rendered.close()
                    page_ocr(image, index + 1, count)
                finally:
                    if image is not None:
                        image.close()
                    if bitmap is not None:
                        bitmap.close()
                    if page is not None:
                        page.close()
        except pdfium.PdfiumError as exc:
            raise HTTPException(400, "This PDF cannot be opened. It may be damaged or password-protected.") from exc
        finally:
            if pdf is not None:
                pdf.close()
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as original:
                    if original.format not in {"PNG", "JPEG", "WEBP"}:
                        raise HTTPException(415, "Choose a PNG, JPEG, WEBP image or PDF.")
                    original.load()
                    image = normalize_image(original)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
                Image.DecompressionBombWarning) as exc:
            raise HTTPException(400, "This image cannot be opened or is too large. Choose a valid PNG, JPEG, WEBP or PDF.") from exc
        try:
            page_ocr(image, 1, 1)
        finally:
            image.close()
    elapsed = time.perf_counter() - started
    return {"markdown": "\n\n---\n\n".join(output), "page_count": len(pages),
            "timing": {"total_seconds": round(elapsed, 3),
                       "inference_seconds": round(sum(page["seconds"] for page in pages), 3)},
            "truncated": any(page["truncated"] for page in pages), "pages": pages,
            "backend": engine.device, "dtype": engine.dtype_name}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = await run_in_threadpool(Engine)
    yield
    app.state.engine = None


app = FastAPI(title="Local LightOnOCR 3", lifespan=lifespan, docs_url=None, redoc_url=None,
              openapi_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])


@app.middleware("http")
async def security_headers(request: Request, call_next):
    if request.method == "POST":
        origin = request.headers.get("origin")
        expected = f"http://{request.headers.get('host', '')}"
        if (origin is not None and origin != expected) or request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "Cross-site requests are not allowed. Open this app's local URL."}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/", response_class=HTMLResponse)
async def index():
    nonce = secrets.token_urlsafe(24)
    html = (APP_DIR / "static" / "index.html").read_text(encoding="utf-8").replace("__CSP_NONCE__", nonce)
    response = HTMLResponse(html)
    response.headers["Content-Security-Policy"] = (
        "default-src 'none'; "
        f"script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
        "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'self'; "
        "frame-ancestors 'self' http://127.0.0.1:* http://localhost:*"
    )
    return response


@app.get("/healthz")
async def healthz(request: Request):
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return engine.health()


@app.post("/api/ocr")
async def ocr(request: Request):
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(503, "The model is still loading.")
    if not inference_lock.acquire(blocking=False):
        raise HTTPException(409, "An OCR job is already running. Wait for it to finish and try again.")
    form = None
    multipart_parser = None
    try:
        if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
            raise HTTPException(415, "Send a multipart form with a file, mode, and max_new_tokens.")
        length = request.headers.get("content-length")
        if length:
            try:
                too_large = int(length) > MAX_REQUEST_BYTES
            except ValueError:
                raise HTTPException(400, "Invalid Content-Length.")
            if too_large:
                raise HTTPException(413, "The upload limit is 25 MB.")
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                raise HTTPException(413, "The upload limit is 25 MB.")
            body.extend(chunk)

        async def body_stream():
            yield bytes(body)

        try:
            multipart_parser = MemoryMultipartParser(request.headers, body_stream(), max_files=1, max_fields=2)
            form = await multipart_parser.parse()
        except (MultiPartException, ValueError) as exc:
            raise HTTPException(400, "Invalid upload form. Choose one file and try again.") from exc
        if any(key not in {"file", "mode", "max_new_tokens"} for key in form.keys()):
            raise HTTPException(400, "Unexpected upload form fields.")
        if any(len(form.getlist(key)) > 1 for key in form.keys()):
            raise HTTPException(400, "Upload fields must not be repeated.")
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise HTTPException(400, "Choose a file first.")
        if upload.size is not None and upload.size > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "The upload limit is 25 MB.")
        data = await upload.read(MAX_UPLOAD_BYTES + 1)
        if not data:
            raise HTTPException(400, "The selected file is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "The upload limit is 25 MB.")
        mode = form.get("mode", "plain")
        if mode not in {"plain", "grounding"}:
            raise HTTPException(400, "Mode must be plain or grounding.")
        raw_tokens = form.get("max_new_tokens", "2048")
        try:
            max_new_tokens = int(raw_tokens)
        except (TypeError, ValueError):
            raise HTTPException(400, "The token limit must be an integer from 128 to 4096.")
        if not 128 <= max_new_tokens <= 4096:
            raise HTTPException(400, "The token limit must be an integer from 128 to 4096.")
        # No queue: the lock covers upload, preprocessing, and inference. The worker
        # remains joined even if a client leaves, avoiding overlapping model calls.
        return await run_in_threadpool(perform_ocr, engine, data, mode, max_new_tokens)
    except HTTPException:
        raise
    except Exception:
        logger.exception("OCR failed")
        raise HTTPException(500, "OCR failed. Check the launcher terminal. For CPU compatibility issues, try LIGHTONOCR_DTYPE=float32 and restart.")
    finally:
        try:
            if form is not None:
                await form.close()
            elif multipart_parser is not None:
                # Also close partially parsed uploads on unexpected parser errors.
                for temporary_file in getattr(multipart_parser, "_files_to_close_on_error", []):
                    temporary_file.close()
        finally:
            inference_lock.release()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Local LightOnOCR server")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    class ReadyServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            if self.started:
                print(f"LIGHTONOCR_READY http://127.0.0.1:{args.port}", flush=True)

    ReadyServer(uvicorn.Config(app, host="127.0.0.1", port=args.port, workers=1,
                              proxy_headers=False, access_log=False)).run()


if __name__ == "__main__":
    main()
