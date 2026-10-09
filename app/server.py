"""Original LightOnOCR viewer with a private, in-process Transformers backend.

The vendored upstream application owns the viewer and run format. This wrapper
adds loopback-only serving, bounded uploads, and a fixed local model endpoint.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import re
import socket
import threading
import time
import warnings
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
from urllib.request import ProxyHandler, build_opener

from PIL import Image, UnidentifiedImageError

from engine import Engine, MODEL_ID, normalize_image
from model_catalog import MODELS, download_model, get_model
from filelock import FileLock, Timeout
from vendor.lightonocr.client import LightOnOCR
from vendor.lightonocr.models import check_mode
from vendor.lightonocr.server import Handler, ROUTES, STATIC, Server, safe_name, unique_name
from vendor.lightonocr.settings import write_settings

APP_DIR = Path(__file__).resolve().parent
OUT_DIR = APP_DIR / "out"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_PAGES = 20
Image.MAX_IMAGE_PIXELS = 40_000_000
logger = logging.getLogger("lightonocr")


def check_viewer_assets() -> None:
    """Fail before loading weights if installation omitted required UI assets."""
    required = ["index.html", "app.js", "app.css", "favicon.png", "lighton.svg", "lighton-dark.svg",
                "lib/marked.min.js", "lib/purify.min.js", "lib/katex/katex.min.css",
                "lib/katex/katex.min.js", "lib/katex/contrib/auto-render.min.js"]
    css = STATIC / "lib/katex/katex.min.css"
    if css.is_file():
        required.extend("lib/katex/" + name for name in re.findall(
            r"url\((?:['\"])?(fonts/[^)\"']+)(?:['\"])?\)", css.read_text(encoding="utf-8")))
    missing = [name for name in required if not (STATIC / name).is_file()]
    if missing:
        raise RuntimeError("Viewer assets are missing. Run Install or Update in the launcher, "
                           "then start again. Missing: " + ", ".join(missing[:5]))


def selected_pages(spec: str) -> list[int] | None:
    """The upstream page syntax, bounded before expanding any numeric range."""
    if not spec.strip() or spec.strip().lower() == "all":
        return None
    if len(spec) > 256:
        raise ValueError("Pages must look like 1,3-5 and select at most 20 pages.")
    result: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"([0-9]{1,2})(?:\s*-\s*([0-9]{1,2}))?", part)
        if not match:
            raise ValueError("Pages must look like 1,3-5, using page numbers from 1 to 20.")
        first, last = int(match[1]), int(match[2] or match[1])
        if not 1 <= first <= last <= MAX_PAGES:
            raise ValueError("Pages must use increasing ranges and page numbers from 1 to 20.")
        result.extend(range(first, last + 1))
        if len(result) > MAX_PAGES:
            raise ValueError("Select at most 20 pages.")
    if not result or len(set(result)) != len(result):
        raise ValueError("Select each page once, for example 1,3-5.")
    return result


def validate_upload(data: bytes, pages: list[int] | None) -> str:
    """Validate the actual bytes; return a safe extension for upstream rendering."""
    if b"%PDF-" in data[:1024]:
        import pypdfium2 as pdfium

        try:
            with pdfium.PdfDocument(data) as pdf:
                count = len(pdf)
                if not 1 <= count <= MAX_PAGES:
                    raise ValueError(f"PDFs must contain between 1 and {MAX_PAGES} pages.")
                if pages and any(n > count for n in pages):
                    raise ValueError(f"This PDF has {count} pages; the page selection is out of range.")
                for index in range(count):
                    page = pdf[index]
                    try:
                        if not all(math.isfinite(n) and n > 0 for n in page.get_size()):
                            raise ValueError("The PDF contains an invalid page size.")
                    finally:
                        page.close()
        except pdfium.PdfiumError as exc:
            raise ValueError("This PDF cannot be opened. It may be damaged or password-protected.") from exc
        return ".pdf"
    if pages is not None and pages != [1]:
        raise ValueError("An image contains one page. Leave Pages empty or select 1.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                image_format = image.format
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("This image cannot be opened or is too large. Choose a valid image or PDF.") from exc
    # Preserve ordinary source extensions without trusting the upload filename.
    return {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "TIFF": ".tiff",
            "GIF": ".gif", "BMP": ".bmp"}.get(image_format, ".image")


class LocalViewerServer(Server):
    def __init__(self, address: tuple[str, int], out: Path, ocr: LightOnOCR,
                 engine: Engine, adapter_url: str):
        out.mkdir(parents=True, exist_ok=True)
        super().__init__(address, out.resolve(), ocr, longest_edge=engine.longest_edge,
                         temperature=0.1, concurrency=1)
        self.RequestHandlerClass = LocalHandler
        self.engine = engine
        self.adapter_url = adapter_url
        self.upload_lock = threading.Lock()
        self.work_lock = threading.Lock()

    def work(self, job: dict, pages: list[int] | None, src: Path) -> None:
        # Multiple upstream jobs remain available, but one model is used serially.
        with self.work_lock:
            prepared = None
            try:
                # Upstream converts images directly to RGB. Prepare only inputs
                # needing white alpha compositing or EXIF orientation first, and
                # keep the original upload untouched for the run's history.
                if src.suffix.lower() != ".pdf":
                    with Image.open(src) as image:
                        if (image.mode in {"RGBA", "LA"} or "transparency" in image.info
                                or image.getexif().get(274, 1) != 1):
                            prepared = src.parent / ".normalized-input.png"
                            normalized = normalize_image(image, self.engine.longest_edge)
                            try:
                                normalized.save(prepared, format="PNG")
                            finally:
                                normalized.close()
                super().work(job, pages, prepared or src)
            except Exception as exc:
                job["status"], job["error"] = "error", f"{type(exc).__name__}: {exc}"
            finally:
                if prepared is not None:
                    prepared.unlink(missing_ok=True)
            if job["status"] == "error" and "Is vLLM running?" in (job.get("error") or ""):
                job["error"] = "Cannot reach the local Transformers backend. Restart the launcher."


class LocalHandler(Handler):
    server: LocalViewerServer

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(30)

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", (
            "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "font-src 'self'; img-src 'self' data:; connect-src 'self'; "
            "base-uri 'none'; form-action 'self'; object-src 'none'; "
            "frame-ancestors 'self' http://127.0.0.1:* http://localhost:*"
        ))
        super().end_headers()

    def dispatch(self, method: str) -> None:
        # Exact loopback authorities prevent DNS rebinding. Browser writes must
        # come from this viewer; CLI clients without Origin remain supported.
        port = self.server.server_port
        allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            allowed_hosts.update({"127.0.0.1", "localhost"})
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1 or hosts[0] not in allowed_hosts:
            return self.fail(HTTPStatus.BAD_REQUEST, "Use this app's 127.0.0.1 or localhost URL.")
        if method in {"POST", "PUT", "DELETE"}:
            origin = self.headers.get("Origin")
            if ((origin is not None and origin != f"http://{hosts[0]}") or
                    self.headers.get("Sec-Fetch-Site") == "cross-site"):
                return self.fail(HTTPStatus.FORBIDDEN, "Cross-site writes are not allowed. Open this app's local URL.")
            if self.headers.get("Transfer-Encoding") is not None:
                return self.fail(HTTPStatus.BAD_REQUEST, "Send a fixed-length request body.")
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) > 1 or (lengths and not re.fullmatch(r"[0-9]+", lengths[0])):
                return self.fail(HTTPStatus.BAD_REQUEST, "Invalid Content-Length.")
            # Cap the text length too, so arbitrary-size integers cannot escape validation.
            if lengths and (len(lengths[0]) > 10 or int(lengths[0]) > MAX_UPLOAD_BYTES):
                return self.fail(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "The upload limit is 25 MB.")
            self.body_length = int(lengths[0]) if lengths else 0
        try:
            url = urlsplit(self.path)
            path = unquote(url.path)
            self.query = {k: v[0] for k, v in parse_qs(url.query, max_num_fields=16).items()}
        except ValueError:
            return self.fail(HTTPStatus.BAD_REQUEST, "Invalid request URL.")
        if method == "GET" and path == "/healthz":
            return self.send_json({**self.server.engine.health(), "adapter_url": self.server.adapter_url,
                                   "viewer": "original-lightonocr"})
        if method == "GET" and path.startswith("/static/"):
            return self.static(path[len("/static/"):])
        if method == "POST" and path == "/api/runs":
            return self.create_run()
        if method == "PUT" and path == "/api/settings":
            return self.update_settings()
        # Resolve by method name, since upstream stores unbound Handler methods.
        for route_method, pattern, handler in ROUTES:
            match = re.fullmatch(pattern, path)
            if match and route_method == method:
                return getattr(self, handler.__name__)(*match.groups())
        self.fail(HTTPStatus.NOT_FOUND, "not found")

    def static(self, file: str) -> None:
        # KaTeX uses nested font paths. Resolve under the asset root, never outside.
        root = STATIC.resolve()
        try:
            path = (root / file).resolve()
        except (OSError, ValueError):
            return self.fail(HTTPStatus.NOT_FOUND, "not found")
        if not file or "\\" in file or not path.is_relative_to(root) or any(p.startswith(".") for p in Path(file).parts):
            return self.fail(HTTPStatus.NOT_FOUND, "not found")
        self.send_file(path)

    def _read_body(self) -> bytes | None:
        try:
            data = self.rfile.read(self.body_length)
        except (TimeoutError, OSError):
            self.fail(HTTPStatus.REQUEST_TIMEOUT, "Upload timed out. Try again.")
            return None
        if len(data) != self.body_length:
            self.fail(HTTPStatus.BAD_REQUEST, "The upload was incomplete. Try again.")
            return None
        return data

    def update_settings(self) -> None:
        if self.body_length > 16 * 1024:
            return self.fail(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Settings are limited to 16 KB.")
        data = self._read_body()
        if data is None:
            return
        try:
            body = json.loads(data or b"{}")
            if not isinstance(body, dict) or any(k not in {"base_url", "model"} for k in body):
                raise ValueError
            if any(v is not None and not isinstance(v, str) for v in body.values()):
                raise ValueError
            base_url, model = (str(body.get("base_url") or "").strip().rstrip("/"),
                               str(body.get("model") or "").strip())
        except (ValueError, TypeError, UnicodeError):
            return self.fail(HTTPStatus.BAD_REQUEST, "Body must be JSON with base_url and model.")
        if base_url and base_url != self.server.adapter_url:
            return self.fail(HTTPStatus.BAD_REQUEST, "This launcher uses its local Transformers backend. External or custom server URLs are disabled; keep the displayed local URL or leave it empty.")
        active_model = self.server.engine.model_id
        if model and model != active_model:
            return self.fail(HTTPStatus.BAD_REQUEST, f"This launcher is running {active_model}. Stop it and choose Start 0.8B, Start 1B, or Start 4B to change models.")
        write_settings(self.server.out, base_url=self.server.adapter_url, model=active_model)
        self.send_json(self.server.describe_endpoint())

    def create_run(self) -> None:
        mode = self.query.get("mode", "grounding")
        try:
            check_mode(self.server.engine.model_id, mode)
            pages = selected_pages(self.query.get("pages", ""))
        except ValueError as exc:
            return self.fail(HTTPStatus.BAD_REQUEST, str(exc))
        if self.body_length <= 0:
            return self.fail(HTTPStatus.BAD_REQUEST, "Choose a nonempty PDF or image.")
        data = self._read_body()
        if data is None:
            return
        try:
            suffix = validate_upload(data, pages)
        except ValueError as exc:
            return self.fail(HTTPStatus.BAD_REQUEST, str(exc))
        filename = Path(unquote(self.headers.get("X-Filename") or "document.pdf").replace("\\", "/")).name
        filename = filename.replace("\x00", "")[:255] or "document"
        # Atomic name selection keeps simultaneous uploads from sharing a folder.
        with self.server.upload_lock:
            name = unique_name(self.server.out, safe_name(self.query.get("name") or Path(filename).stem)[:100])
            run = self.server.out / name
            try:
                run.mkdir(parents=True)
                src = run / f"source{suffix}"
                src.write_bytes(data)
            except OSError:
                logger.exception("Could not save uploaded document")
                return self.fail(HTTPStatus.INTERNAL_SERVER_ERROR, "Could not save the document in app/out. Check free disk space and folder permissions.")
            job = self.server.start_job(name, filename, mode, pages, src)
        self.send_json(job, HTTPStatus.CREATED)

    def run_file(self, name: str, file: str) -> None:
        if safe_name(name) != name or file.startswith("."):
            return self.fail(HTTPStatus.NOT_FOUND, "not found")
        root = self.server.out.resolve()
        path = (root / name / file).resolve()
        if not path.is_relative_to(root):
            return self.fail(HTTPStatus.NOT_FOUND, "not found")
        self.send_file(path)

    def delete_run(self, name: str) -> None:
        if any(j["name"] == name and j["status"] == "running" for j in self.server.jobs):
            return self.fail(HTTPStatus.CONFLICT, "This run is still processing. Wait for it to finish before deleting it.")
        super().delete_run(name)


def make_client(adapter_url: str, model_id: str = MODEL_ID) -> LightOnOCR:
    """Use the upstream client with HTTP proxy inheritance explicitly disabled."""
    import httpx
    from openai import OpenAI, Timeout

    client = LightOnOCR(base_url=adapter_url, model=model_id, api_key="local-only")
    client.client.close()
    client.client = OpenAI(base_url=adapter_url, api_key="local-only",
                           timeout=Timeout(3600.0, connect=10.0),
                           http_client=httpx.Client(trust_env=False))
    return client


class LocalBackend:
    """A private OpenAI-compatible adapter sharing the one already loaded Engine."""
    def __init__(self, engine: Engine):
        import uvicorn
        from adapter import create_app

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(128)
        self.url = f"http://127.0.0.1:{self.socket.getsockname()[1]}"
        self.server = uvicorn.Server(uvicorn.Config(create_app(engine), host="127.0.0.1",
            port=self.socket.getsockname()[1], workers=1, proxy_headers=False,
            access_log=False, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run,
            kwargs={"sockets": [self.socket]}, name="local-transformers-api", daemon=True)

    def start(self) -> None:
        self.thread.start()
        opener = build_opener(ProxyHandler({}))
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if not self.thread.is_alive():
                raise RuntimeError("The local Transformers adapter stopped during startup.")
            if self.server.started:
                try:
                    with opener.open(self.url + "/healthz", timeout=1) as response:
                        if response.status == 200:
                            return
                except OSError:
                    pass
            time.sleep(0.05)
        raise RuntimeError("The local Transformers adapter did not become ready.")

    def close(self) -> None:
        self.server.should_exit = True
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        self.socket.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Original LightOnOCR viewer with local Transformers inference")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runtime_lock = FileLock(str(APP_DIR / ".runtime.lock"), timeout=0)
    try:
        runtime_lock.acquire()
    except Timeout:
        raise SystemExit("LightOnOCR is already running or preparing a model. Stop it in Pinokio before selecting another model.")
    try:
        check_viewer_assets()
        download_model(get_model(args.model))
        engine = Engine(args.model)
        backend = LocalBackend(engine)
        viewer = None
        ocr = None
        try:
            backend.start()
            adapter_url = backend.url + "/v1"
            ocr = make_client(adapter_url, engine.model_id)
            status = ocr.check()
            if not status["reachable"] or engine.model_id not in status["models"]:
                raise RuntimeError(f"The local model endpoint failed its readiness check: {status}")
            viewer = LocalViewerServer(("127.0.0.1", args.port), OUT_DIR, ocr, engine, adapter_url)
            # This marker is consumed by the launcher. Never print before all three
            # readiness conditions: loaded weights, responsive adapter, bound viewer.
            print(f"LIGHTONOCR_READY http://127.0.0.1:{viewer.server_port}", flush=True)
            logger.info("Original LightOnOCR viewer; documents and results saved in %s", viewer.out)
            viewer.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if viewer is not None:
                viewer.server_close()
            if ocr is not None:
                ocr.client.close()
            backend.close()
    finally:
        runtime_lock.release()


if __name__ == "__main__":
    main()
