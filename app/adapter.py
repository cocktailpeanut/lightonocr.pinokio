"""Private, bounded OpenAI compatibility layer for the original LightOnOCR viewer.

This is deliberately not a general chat API: one local page image, optionally the
model's literal grounding prompt. Run on loopback, with one Uvicorn worker.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import math
import re
import threading
import time
import uuid
import warnings
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from engine import Engine, MAX_UPLOAD_BYTES, MODEL_ID, normalize_image

MAX_ENCODED_IMAGE_BYTES = 37 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_ENCODED_IMAGE_BYTES + 64 * 1024
MAX_PENDING_REQUESTS = 2  # One running page and at most one waiting page.
DEFAULT_MAX_TOKENS = 2048
_DATA_URL = re.compile(r"\Adata:image/(png|jpeg|webp);base64,")
logger = logging.getLogger("lightonocr.adapter")


@dataclass(frozen=True)
class PageRequest:
    image_url: str
    mode: str
    max_tokens: int
    temperature: float
    top_p: float


def invalid(message: str, status: int = 400) -> HTTPException:
    return HTTPException(status, message)


def _keys(value: Any, allowed: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not set(value) <= allowed:
        raise invalid(f"Invalid or unsupported {label} fields.")
    return value


def _number(value: Any, label: str, minimum: float, maximum: float,
            *, exclude_minimum: bool = False) -> float:
    if (type(value) not in {int, float} or not minimum <= value <= maximum
            or not math.isfinite(value)
            or (exclude_minimum and value == minimum)):
        raise invalid(f"Invalid {label}.")
    return float(value)


def validate_payload(payload: Any) -> PageRequest:
    payload = _keys(payload, {"model", "messages", "max_tokens", "temperature",
                              "top_p", "stream", "chat_template_kwargs"}, "request")
    if payload.get("model") != MODEL_ID:
        raise invalid(f"model must be {MODEL_ID}.")
    if payload.get("stream", False) is not False:
        raise invalid("Only non-streaming responses are supported.")
    if "chat_template_kwargs" in payload:
        kwargs = _keys(payload["chat_template_kwargs"], {"enable_thinking"}, "chat template")
        if kwargs.get("enable_thinking", False) is not False:
            raise invalid("Thinking must be disabled.")
    tokens = payload.get("max_tokens", DEFAULT_MAX_TOKENS)
    if type(tokens) is not int or not 128 <= tokens <= 4096:
        raise invalid("max_tokens must be an integer from 128 to 4096.")
    temperature = _number(payload.get("temperature", 0.2), "temperature", 0, 2)
    top_p = _number(payload.get("top_p", 1.0), "top_p", 0, 1, exclude_minimum=True)
    messages = payload.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise invalid("Send exactly one user message containing one page image.")
    message = _keys(messages[0], {"role", "content"}, "message")
    if message.get("role") != "user":
        raise invalid("Only a user message is supported.")
    content = message.get("content")
    if not isinstance(content, list) or not 1 <= len(content) <= 2:
        raise invalid("Send one image and, optionally, the text grounding.")
    image_url = None
    mode = "plain"
    for item in content:
        if not isinstance(item, dict):
            raise invalid("Invalid message content.")
        if item.get("type") == "image_url":
            _keys(item, {"type", "image_url"}, "image")
            if image_url is not None:
                raise invalid("Send exactly one image.")
            image = _keys(item.get("image_url"), {"url"}, "image URL")
            image_url = image.get("url")
            if not isinstance(image_url, str) or not _DATA_URL.match(image_url):
                raise invalid("Use a PNG, JPEG, or WEBP base64 data URL; remote URLs are not supported.")
            if len(image_url) > MAX_ENCODED_IMAGE_BYTES:
                raise invalid("The encoded image limit is 37 MB.", 413)
        elif item.get("type") == "text":
            _keys(item, {"type", "text"}, "text")
            if item.get("text") != "grounding" or mode != "plain":
                raise invalid("The only supported text prompt is grounding.")
            mode = "grounding"
        else:
            raise invalid("Unsupported message content type.")
    if image_url is None:
        raise invalid("Send exactly one image.")
    return PageRequest(image_url, mode, tokens, temperature, top_p)


def _decode_image(data_url: str) -> Image.Image:
    try:
        encoded = data_url.split(",", 1)[1]
        # Bound decoded memory before allocating it; padding can remove at most 2 bytes.
        if len(encoded) > 4 * ((MAX_UPLOAD_BYTES + 2) // 3):
            raise invalid("The decoded image limit is 25 MB.", 413)
        data = base64.b64decode(encoded, validate=True)
        if len(data) > MAX_UPLOAD_BYTES:
            raise invalid("The decoded image limit is 25 MB.", 413)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as original:
                if original.format not in {"PNG", "JPEG", "WEBP"}:
                    raise invalid("Use a PNG, JPEG, or WEBP image.")
                original.load()
                return normalize_image(original)
    except (ValueError, binascii.Error, UnidentifiedImageError, OSError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise invalid("The image is invalid or its pixel dimensions are too large.") from exc


async def _read_json(request: Request) -> Any:
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise invalid("Send application/json.")
    length = request.headers.get("content-length")
    if length is not None:
        try:
            declared = int(length)
        except ValueError as exc:
            raise invalid("Invalid Content-Length.") from exc
        if declared < 0:
            raise invalid("Invalid Content-Length.")
        if declared > MAX_REQUEST_BYTES:
            raise invalid("The encoded request is too large.", 413)
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_REQUEST_BYTES:
            raise invalid("The encoded request is too large.", 413)
        body.extend(chunk)
    try:
        return await run_in_threadpool(json.loads, body)
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise invalid("Invalid JSON request.") from exc


def _completion(engine: Engine, lock: threading.Lock, page: PageRequest) -> dict[str, Any]:
    # Keep this lock in the worker: client cancellation cannot unlock an active model.
    with lock:
        image = _decode_image(page.image_url)
        try:
            text, tokens, truncated = engine.transcribe(
                image, page.mode, page.max_tokens,
                temperature=page.temperature, top_p=page.top_p,
            )
            prompt_tokens = int(getattr(engine, "last_prompt_tokens", 0))
        finally:
            image.close()
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion", "created": int(time.time()), "model": MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "length" if truncated else "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": tokens,
                  "total_tokens": prompt_tokens + tokens},
    }


def create_app(engine: Engine) -> FastAPI:
    """Wrap an already-loaded engine; never download or load models here."""
    app = FastAPI(title="LightOnOCR Transformers adapter", docs_url=None,
                  redoc_url=None, openapi_url=None)
    app.state.engine = engine
    app.state.pending = 0
    app.state.tasks = set()
    lock = getattr(engine, "inference_lock", None) or threading.Lock()
    app.state.inference_lock = lock
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    @app.exception_handler(HTTPException)
    async def request_error(request: Request, exc: HTTPException):
        return JSONResponse({"error": {"message": str(exc.detail),
                                      "type": "invalid_request_error" if exc.status_code < 500 else "server_error",
                                      "param": None, "code": exc.status_code}},
                            status_code=exc.status_code, headers=exc.headers)

    @app.middleware("http")
    async def security(request: Request, call_next):
        origin = request.headers.get("origin")
        expected = f"http://{request.headers.get('host', '')}"
        if ((origin is not None and origin != expected)
                or request.headers.get("sec-fetch-site") == "cross-site"):
            response = JSONResponse({"error": {"message": "Cross-site requests are not allowed."}}, status_code=403)
        else:
            response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/healthz")
    async def healthz():
        health = dict(engine.health())
        health.update(busy=lock.locked(), pending_requests=app.state.pending,
                      max_pending_requests=MAX_PENDING_REQUESTS)
        return health

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model",
                                            "created": 0, "owned_by": "lightonai"}]}

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        if app.state.pending >= MAX_PENDING_REQUESTS:
            raise HTTPException(429, "OCR is busy; its one-page queue is full.", headers={"Retry-After": "2"})
        app.state.pending += 1
        task = None
        try:
            page = validate_payload(await _read_json(request))
            task = asyncio.create_task(run_in_threadpool(_completion, engine, lock, page))
            app.state.tasks.add(task)

            def finished(done):
                app.state.tasks.discard(done)
                app.state.pending -= 1
                # Retrieve an error even if the HTTP client disconnected meanwhile.
                if not done.cancelled():
                    done.exception()

            task.add_done_callback(finished)
            return await asyncio.shield(task)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Transformers OCR failed")
            raise HTTPException(500, "OCR failed. See the launcher terminal for details.") from exc
        finally:
            if task is None:
                app.state.pending -= 1

    return app
