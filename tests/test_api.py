"""API-only tests using a fake engine. These do NOT validate model inference.

Run: python -m unittest discover -s tests -p 'test_api.py' -v
Requires app dependencies plus httpx (or the TestClient transport for Starlette).
No torch, Transformers, downloaded weights, network, or accelerator is used.
"""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402


class FakeEngine:
    """Test-only OCR stub: deterministic output, never loads a real model."""
    device = "cpu"
    dtype_name = "float32"
    truncated = False

    def __init__(self):
        self.calls = []

    def health(self):
        return {"status": "ready", "backend": self.device, "dtype": self.dtype_name,
                "revision": server.MODEL_REVISION, "versions": {"torch": "test-only", "transformers": "test-only"},
                "busy": server.inference_lock.locked()}

    def transcribe(self, image, mode, max_new_tokens):
        if image.mode != "RGB" or max(image.size) > server.LONGEST_EDGE:
            raise AssertionError("Image must be RGB and bounded before inference")
        self.calls.append((image.size, mode, max_new_tokens))
        return "<script>unsafe()</script>\n# Fake OCR output", 10, self.truncated


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        buffer = io.BytesIO()
        with Image.new("RGB", (1800, 900), "white") as image:
            image.save(buffer, "PNG")
        cls.png = buffer.getvalue()

    def test_transparent_background_is_white(self):
        with Image.new("RGBA", (20, 20), (0, 0, 0, 0)) as image:
            image.putpixel((10, 10), (0, 0, 0, 255))
            result = server.normalize_image(image)
            try:
                self.assertEqual(result.getpixel((0, 0)), (255, 255, 255))
                self.assertEqual(result.getpixel((10, 10)), (0, 0, 0))
            finally:
                result.close()

    def setUp(self):
        self.previous_engine = getattr(server.app.state, "engine", None)
        self.engine = FakeEngine()
        server.app.state.engine = self.engine
        # Do not enter TestClient's context: that would start the real lifespan/model.
        self.client = TestClient(server.app, base_url="http://127.0.0.1:7788")

    def tearDown(self):
        self.client.close()
        server.app.state.engine = self.previous_engine
        self.assertFalse(server.inference_lock.locked(), "Request leaked the inference lock")

    def post(self, data=None, headers=None, content=None):
        return self.client.post("/api/ocr", files={"file": ("sample.png", self.png if content is None else content, "image/png")},
                                data=data or {}, headers=headers or {})

    @staticmethod
    def pdf(page_count):
        buffer = io.BytesIO()
        images = [Image.new("RGB", (60, 80), "white") for _ in range(page_count)]
        try:
            images[0].save(buffer, "PDF", save_all=True, append_images=images[1:])
        finally:
            for image in images:
                image.close()
        return buffer.getvalue()

    def test_ui_nonce_and_security_headers(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("__CSP_NONCE__", response.text)
        self.assertIn("script-src 'nonce-", response.headers["content-security-policy"])
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertNotIn(".innerHTML", response.text)

    def test_health(self):
        self.assertEqual(self.client.get("/healthz").json()["status"], "ready")

    def test_loading(self):
        server.app.state.engine = None
        self.assertEqual(self.client.get("/healthz").status_code, 503)
        self.assertEqual(self.post().status_code, 503)

    def test_image_ocr_and_safe_raw_output(self):
        response = self.post({"mode": "grounding", "max_new_tokens": "4096"},
                             {"origin": "http://127.0.0.1:7788"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["page_count"], 1)
        self.assertIn("<script>", response.json()["markdown"])
        self.assertFalse(response.json()["truncated"])
        self.assertEqual(self.engine.calls[0][1:], ("grounding", 4096))

    def test_cross_site_requests_are_rejected(self):
        for headers in ({"origin": "https://attacker.example"}, {"origin": "null"},
                        {"origin": "http://127.0.0.1:9999"}, {"sec-fetch-site": "cross-site"}):
            with self.subTest(headers=headers):
                self.assertEqual(self.post(headers=headers).status_code, 403)
        self.assertEqual(self.engine.calls, [])

    def test_host_guard(self):
        self.assertEqual(self.client.get("/healthz", headers={"host": "attacker.example"}).status_code, 400)

    def test_bad_mode_and_token_limits(self):
        for data in ({"mode": "bad"}, {"max_new_tokens": "127"}, {"max_new_tokens": "4097"},
                     {"max_new_tokens": "2048.5"}, {"max_new_tokens": "invalid"}):
            with self.subTest(data=data):
                self.assertEqual(self.post(data).status_code, 400)
        self.assertEqual(self.engine.calls, [])

    def test_invalid_and_empty_images(self):
        for content in (b"not an image", b""):
            with self.subTest(content=content):
                self.assertEqual(self.post(content=content).status_code, 400)

    def test_unsupported_image_type(self):
        buffer = io.BytesIO()
        with Image.new("RGB", (10, 10), "white") as image:
            image.save(buffer, "BMP")
        self.assertEqual(self.post(content=buffer.getvalue()).status_code, 415)

    def test_busy_request_is_rejected_without_queue(self):
        server.inference_lock.acquire()
        try:
            self.assertEqual(self.post().status_code, 409)
        finally:
            server.inference_lock.release()
        self.assertEqual(self.engine.calls, [])

    def test_multi_page_pdf(self):
        response = self.post(content=self.pdf(2))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["page_count"], 2)
        self.assertEqual(len(response.json()["pages"]), 2)
        self.assertIn("\n\n---\n\n", response.json()["markdown"])

    def test_pdf_page_cap(self):
        self.assertEqual(self.post(content=self.pdf(21)).status_code, 400)
        self.assertEqual(self.engine.calls, [])

    def test_upload_size_cap(self):
        self.assertEqual(self.post(content=b"x" * (server.MAX_UPLOAD_BYTES + 1)).status_code, 413)
        self.assertEqual(self.engine.calls, [])

    def test_stream_size_cap_without_content_length(self):
        chunk = b"x" * (1024 * 1024)
        response = self.client.post("/api/ocr", content=(chunk for _ in range(26)),
                                    headers={"content-type": "multipart/form-data; boundary=BOUNDARY"})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.engine.calls, [])

    def test_upload_stays_in_memory_and_is_closed(self):
        uploaded = []
        original_parse = server.MemoryMultipartParser.parse

        async def inspect_parse(parser):
            form = await original_parse(parser)
            uploaded.append(form["file"].file)
            self.assertFalse(form["file"].file._rolled, "Upload spilled to disk")
            return form

        # PNG decoders allow trailing data; make a valid upload larger than the
        # default Starlette 1 MB disk-spooling threshold.
        with patch.object(server.MemoryMultipartParser, "parse", inspect_parse):
            response = self.post(content=self.png + b"\0" * (2 * 1024 * 1024))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(uploaded[0].closed)

    def test_truncation_flag(self):
        self.engine.truncated = True
        response = self.post()
        self.assertTrue(response.json()["truncated"])
        self.assertTrue(response.json()["pages"][0]["truncated"])

    def test_model_errors_are_500_and_release_lock(self):
        with patch.object(self.engine, "transcribe", side_effect=ValueError("test-only model failure")):
            with self.assertLogs("lightonocr", level="ERROR"):
                response = self.post()
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("test-only model failure", response.text)
        self.assertFalse(server.inference_lock.locked())
        self.assertEqual(self.post().status_code, 200)

    def test_invalid_content_type(self):
        self.assertEqual(self.client.post("/api/ocr", json={"file": "x"}).status_code, 415)

    def test_unexpected_form_fields(self):
        self.assertEqual(self.post({"unexpected": "value"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
