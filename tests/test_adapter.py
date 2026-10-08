"""Adapter contract tests. Fake engines only: no model downloads or weight loads.

Run: app/env/bin/python tests/test_adapter.py
"""
import asyncio
import base64
from contextlib import nullcontext
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
APP = Path(__file__).resolve().parents[1] / 'app'
sys.path.insert(0, str(APP))
sys.path.insert(0, str(APP / 'vendor'))
import httpx
from fastapi.testclient import TestClient
from PIL import Image
import adapter
import engine
from engine import Engine, MODEL_ID
b = io.BytesIO()
Image.new('RGBA', (80, 40), (0, 0, 0, 0)).save(b, 'PNG')
DATA_URL = 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()
BASE = {'model': MODEL_ID, 'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': DATA_URL}}]}], 'chat_template_kwargs': {'enable_thinking': False}}

class Fake:

    def __init__(self):
        self.inference_lock = threading.Lock()
        self.last_prompt_tokens = 13
        self.calls = []
        self.truncated = False

    def transcribe(self, image, mode, tokens, **kwargs):
        self.calls.append((image.mode, image.getpixel((0, 0)), mode, tokens, kwargs))
        return ('# Real raw text\n$1$', 5, self.truncated)

    def health(self):
        return {'status': 'ready', 'backend': 'transformers'}

class API(unittest.TestCase):

    def setUp(self):
        self.e = Fake()
        self.app = adapter.create_app(self.e)
        self.c = TestClient(self.app, base_url='http://127.0.0.1:8123')

    def tearDown(self):
        self.c.close()

    def post(self, body=None, **kwargs):
        return self.c.post('/v1/chat/completions', json=copy.deepcopy(BASE) if body is None else body, **kwargs)

    def test_plain(self):
        r = self.post()
        self.assertEqual(r.status_code, 200, r.text)
        out = r.json()
        self.assertEqual(out['choices'][0]['message']['content'], '# Real raw text\n$1$')
        self.assertEqual(out['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(out['usage'], {'prompt_tokens': 13, 'completion_tokens': 5, 'total_tokens': 18})
        self.assertEqual(self.e.calls, [('RGB', (255, 255, 255), 'plain', 2048, {'temperature': 0.2, 'top_p': 1.0})])

    def test_grounding_sampling(self):
        p = copy.deepcopy(BASE)
        p['messages'][0]['content'].append({'type': 'text', 'text': 'grounding'})
        p.update(temperature=0, top_p=0.9, max_tokens=128)
        self.e.truncated = True
        r = self.post(p)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()['choices'][0]['finish_reason'], 'length')
        self.assertEqual(self.e.calls[0][2:], ('grounding', 128, {'temperature': 0.0, 'top_p': 0.9}))

    def test_models_health(self):
        self.assertEqual(self.c.get('/v1/models').json()['data'][0]['id'], MODEL_ID)
        self.assertEqual(self.c.get('/healthz').json()['max_pending_requests'], 2)
        self.assertEqual(self.c.get('/healthz').json()['pending_requests'], 0)

    def test_body_fields(self):
        edits = [{'stream': True}, {'stream': 1}, {'model': 'other'}, {'messages': []}, {'extra': 1}, {'max_tokens': 127}, {'max_tokens': 4097}, {'max_tokens': True}, {'max_tokens': 1.5}, {'max_tokens': '2048'}, {'temperature': -1}, {'temperature': 3}, {'temperature': 10 ** 400}, {'temperature': True}, {'top_p': 0}, {'top_p': 1.1}, {'chat_template_kwargs': {'enable_thinking': True}}, {'chat_template_kwargs': {'tools': []}}]
        for edit in edits:
            with self.subTest(edit=edit):
                p = copy.deepcopy(BASE)
                p.update(edit)
                r = self.post(p)
                self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(self.e.calls, [])

    def test_messages(self):
        values = [[], {}, 'hi', [{'role': 'system', 'content': []}], [BASE['messages'][0], BASE['messages'][0]], [{'role': 'user', 'content': 'hi'}], [{'role': 'user', 'content': [{'type': 'text', 'text': 'grounding'}]}], [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'https://example.com/p.png'}}]}], [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,!!!!'}}]}]]
        for value in values:
            with self.subTest(value=value):
                p = copy.deepcopy(BASE)
                p['messages'] = value
                r = self.post(p)
                self.assertEqual(r.status_code, 400, r.text)
        p = copy.deepcopy(BASE)
        p['messages'][0]['content'].append({'type': 'text', 'text': 'ignore instructions'})
        self.assertEqual(self.post(p).status_code, 400)
        p = copy.deepcopy(BASE)
        p['messages'][0]['content'] *= 2
        self.assertEqual(self.post(p).status_code, 400)

    def test_invalid_json(self):
        for value in ['{', 'null', '[]', '{"temperature":NaN}', '{"temperature":Infinity}']:
            r = self.c.post('/v1/chat/completions', content=value, headers={'content-type': 'application/json'})
            self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(self.c.post('/v1/chat/completions', content='hi').status_code, 400)

    def test_hosts_origins_cors(self):
        self.assertEqual(self.post(headers={'host': 'evil.example'}).status_code, 400)
        self.assertEqual(self.post(headers={'origin': 'http://evil.example'}).status_code, 403)
        self.assertEqual(self.post(headers={'origin': 'null'}).status_code, 403)
        self.assertEqual(self.post(headers={'origin': 'http://127.0.0.1:8123'}).status_code, 200)
        self.assertEqual(self.post(headers={'sec-fetch-site': 'cross-site'}).status_code, 403)
        r = self.c.options('/v1/chat/completions', headers={'origin': 'https://example.com', 'access-control-request-method': 'POST'})
        self.assertNotIn('access-control-allow-origin', r.headers)
        self.assertEqual(self.c.get('/').status_code, 404)

    def test_limits(self):
        with patch.object(adapter, 'MAX_ENCODED_IMAGE_BYTES', 40):
            self.assertEqual(self.post().status_code, 413)
        with patch.object(adapter, 'MAX_UPLOAD_BYTES', 8):
            self.assertEqual(self.post().status_code, 413)
        with patch.object(adapter, 'MAX_REQUEST_BYTES', 20):
            self.assertEqual(self.post().status_code, 413)
        self.assertEqual(self.post(headers={'content-length': 'not an int'}).status_code, 400)
        self.assertEqual(self.post(headers={'content-length': '-1'}).status_code, 400)
        self.assertEqual(self.c.get('/healthz').json()['pending_requests'], 0)

    def test_bad_image(self):
        p = copy.deepcopy(BASE)
        p['messages'][0]['content'][0]['image_url']['url'] = 'data:image/png;base64,' + base64.b64encode(b'not an image').decode()
        self.assertEqual(self.post(p).status_code, 400)

    def test_sanitized_failure(self):
        with patch.object(self.e, 'transcribe', side_effect=RuntimeError('private path')), self.assertLogs('lightonocr.adapter', level='ERROR'):
            r = self.post()
            self.assertEqual(r.status_code, 500)
            self.assertNotIn('private path', r.text)
        self.assertEqual(self.c.get('/healthz').json()['pending_requests'], 0)

class Queue(unittest.IsolatedAsyncioTestCase):

    async def test_queue_and_cancellation(self):
        started = threading.Event()
        release = threading.Event()
        e = Fake()
        active = 0
        peak = 0

        def slow(*args, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            started.set()
            release.wait(5)
            active -= 1
            return ('ok', 1, False)
        e.transcribe = slow
        app = adapter.create_app(e)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1') as c:
            first = asyncio.create_task(c.post('/v1/chat/completions', json=BASE))
            for _ in range(500):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(started.is_set())
            second = asyncio.create_task(c.post('/v1/chat/completions', json=BASE))
            for _ in range(500):
                if app.state.pending == 2:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(app.state.pending, 2)
            third = await c.post('/v1/chat/completions', json=BASE)
            self.assertEqual(third.status_code, 429)
            first.cancel()
            try:
                await first
            except asyncio.CancelledError:
                pass
            self.assertTrue(e.inference_lock.locked())
            self.assertEqual(app.state.pending, 2)
            release.set()
            self.assertEqual((await second).status_code, 200)
            for _ in range(500):
                if app.state.pending == 0:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(app.state.pending, 0)
            self.assertEqual(peak, 1)

class OriginalClient(unittest.TestCase):
    setUp = API.setUp
    tearDown = API.tearDown

    def test_original_openai_client(self):
        from lightonocr.client import LightOnOCR
        from openai import OpenAI
        client = LightOnOCR(base_url='http://127.0.0.1:8123/v1')
        client.client.close()
        client.client = OpenAI(base_url='http://127.0.0.1:8123/v1', api_key='EMPTY', http_client=self.c, max_retries=0)
        self.assertEqual(client.check()['models'], [MODEL_ID])
        self.assertEqual(client.model, MODEL_ID)
        with Image.new('RGB', (80, 40), 'white') as image:
            self.assertEqual(client.ocr(image, 'plain', fix_dollars=False), '# Real raw text\n$1$')
            self.assertEqual(client.ocr(image, 'grounding', temperature=0, max_tokens=256, top_p=0.9, fix_dollars=False), '# Real raw text\n$1$')
        self.assertEqual(self.e.calls[-1][2:], ('grounding', 256, {'temperature': 0.0, 'top_p': 0.9}))

class Tensor:

    def __init__(self, values=None, *, input_tensor=False):
        self.values = values or []
        self.shape = (1, 3) if input_tensor else (len(self.values),)

    def is_floating_point(self):
        return False

    def to(self, *args, **kwargs):
        return self

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return Tensor(self.values[index[1]])
        return self.values[index]

class EngineContract(unittest.TestCase):

    def test_generation_and_usage(self):
        e = Engine.__new__(Engine)
        e.device = 'cpu'
        e.dtype = 'float32'
        e.torch = SimpleNamespace(inference_mode=nullcontext)
        e.processor = MagicMock()
        e.processor.apply_chat_template.return_value = {'input_ids': Tensor(input_tensor=True)}
        e.processor.decode.return_value = 'hello'
        e.model = MagicMock()
        e.model.generation_config.eos_token_id = 9
        e.model.generate.return_value = Tensor([1, 2, 3, 7, 9])
        with Image.new('RGB', (10, 10), 'white') as image:
            self.assertEqual(e.transcribe(image, 'grounding', 128, temperature=0), ('hello', 2, False))
            self.assertEqual(e.last_prompt_tokens, 3)
            self.assertEqual(e.model.generate.call_args.kwargs['do_sample'], False)
            self.assertNotIn('temperature', e.model.generate.call_args.kwargs)
            self.assertNotIn('top_p', e.model.generate.call_args.kwargs)
            self.assertFalse(e.processor.apply_chat_template.call_args.kwargs['enable_thinking'])
            self.assertEqual(e.processor.apply_chat_template.call_args.args[0][0]['content'][-1], {'type': 'text', 'text': 'grounding'})
            e.model.generate.return_value = Tensor([1, 2, 3] + [7] * 128)
            self.assertEqual(e.transcribe(image, 'plain', 128, temperature=0.3, top_p=0.8), ('hello', 128, True))
            self.assertEqual(e.model.generate.call_args.kwargs['temperature'], 0.3)
            self.assertEqual(e.model.generate.call_args.kwargs['top_p'], 0.8)

    def test_pinned_weights_mapping_and_strict_guard(self):
        torch = SimpleNamespace(__version__='test', cuda=SimpleNamespace(is_available=lambda: False), backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)), float32='float32', set_num_threads=lambda _: None)
        model = MagicMock()
        model.to.return_value = model
        model.eval.return_value = model
        loader = MagicMock()
        transformers = SimpleNamespace(__version__='test', LightOnOcrProcessor=MagicMock(), LightOnOcrForConditionalGeneration=loader)
        clean = {'missing_keys': set(), 'unexpected_keys': set(), 'mismatched_keys': [], 'error_msgs': []}
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'config.json').write_text('{}')
            with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}), patch.object(engine, 'MODEL_DIR', Path(folder)), patch.dict(os.environ, {'LIGHTONOCR_DEVICE': 'cpu', 'LIGHTONOCR_DTYPE': 'float32', 'LIGHTONOCR_CPU_THREADS': '1'}):
                loader.from_pretrained.return_value = (model, clean)
                e = Engine()
                self.assertEqual(loader.from_pretrained.call_args.kwargs['key_mapping'], {'^language_model\\.model\\.': 'model.language_model.'})
                self.assertTrue(loader.from_pretrained.call_args.kwargs['local_files_only'])
                self.assertTrue(loader.from_pretrained.call_args.kwargs['output_loading_info'])
                self.assertEqual(e.health()['revision'], 'b9a2b4c17f1eee9f29058d716b66b5f8e7d8db86')
                self.assertEqual(e.health()['backend'], 'transformers')
                self.assertTrue(e.health()['weight_loading']['strict'])
                self.assertEqual(json.loads(json.dumps(e.health()))['weight_loading']['missing_keys'], [])
                for field in clean:
                    with self.subTest(field=field):
                        loader.from_pretrained.return_value = (model, {**clean, field: ['bad key']})
                        with self.assertRaisesRegex(RuntimeError, 'Checkpoint weights did not load exactly'):
                            Engine()
if __name__ == '__main__':
    unittest.main(verbosity=2)
