"""Model-selection contracts with fake loaders; no checkpoint downloads or OCR claims."""
import base64
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
import adapter, bootstrap, engine, model_catalog, server
from fastapi.testclient import TestClient
from filelock import FileLock
from PIL import Image

class ModelSelection(unittest.TestCase):
    def test_install_downloads_no_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); executable=root/'env'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
            executable.parent.mkdir(parents=True); executable.touch()
            with patch.object(bootstrap,'ROOT',root), patch.object(bootstrap,'run') as run, patch.object(bootstrap,'download_model') as download, patch.object(sys,'argv',['bootstrap.py','--device','cpu']):
                bootstrap.main()
            download.assert_not_called()
            self.assertFalse(any('--download-only' in str(call) for call in run.call_args_list))
            self.assertEqual(json.loads((root/'.installed').read_text())['models'],['0.8B','1B','4B'])

    def test_explicit_download_only(self):
        for variant in model_catalog.MODELS:
            with self.subTest(variant=variant), patch.object(bootstrap,'download_model') as download, patch.object(sys,'argv',['bootstrap.py','--download-only','--model',variant]):
                bootstrap.main(); download.assert_called_once_with(model_catalog.get_model(variant))
        with patch.object(bootstrap,'download_model') as download, patch.object(sys,'argv',['bootstrap.py','--download-only']):
            with self.assertRaises(SystemExit): bootstrap.main()
            download.assert_not_called()

    def test_legacy_cache_and_invalid_selection(self):
        self.assertEqual(model_catalog.get_model('1B').directory,APP/'models'/'lightonocr')
        self.assertEqual(len({spec.directory for spec in model_catalog.MODELS.values()}),3)
        for choice in [None,'','latest','1B;echo unsafe']:
            with self.assertRaises(ValueError): model_catalog.get_model(choice)

    def test_missing_selection_and_second_process_fail_before_download(self):
        with patch.object(sys,'argv',['server.py','--port','12345']), patch.object(server,'download_model') as download:
            with self.assertRaises(SystemExit): server.main()
            download.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with FileLock(str(root/'.runtime.lock')), patch.object(server,'APP_DIR',root), patch.object(sys,'argv',['server.py','--port','12345','--model','4B']), patch.object(server,'download_model') as download, patch.object(server,'Engine') as load:
                with self.assertRaisesRegex(SystemExit,'already running'): server.main()
                download.assert_not_called(); load.assert_not_called()

    def test_loader_families_and_strict_guard(self):
        torch=SimpleNamespace(__version__='test',cuda=SimpleNamespace(is_available=lambda:False),backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:False)),float32='float32',set_num_threads=lambda _:None)
        model=MagicMock(); model.to.return_value=model; model.eval.return_value=model
        pixtral=MagicMock(); qwen=MagicMock()
        clean={'missing_keys':[], 'unexpected_keys':[], 'mismatched_keys':[], 'error_msgs':[]}
        transformers=SimpleNamespace(__version__='test',LightOnOcrProcessor=MagicMock(),LightOnOcrForConditionalGeneration=pixtral,AutoProcessor=MagicMock(),Qwen3_5ForConditionalGeneration=qwen)
        with tempfile.TemporaryDirectory() as folder, patch.object(model_catalog,'APP_DIR',Path(folder)), patch.dict(sys.modules,{'torch':torch,'transformers':transformers}), patch.dict(os.environ,{'LIGHTONOCR_DEVICE':'cpu','LIGHTONOCR_DTYPE':'float32'}):
            for variant,spec in model_catalog.MODELS.items():
                with self.subTest(variant=variant):
                    spec.directory.mkdir(parents=True); (spec.directory/'config.json').write_text('{}')
                    loader=pixtral if variant=='1B' else qwen
                    loader.from_pretrained.return_value=(model,clean)
                    active=engine.Engine(variant)
                    self.assertEqual(active.health()['model'],spec.model_id)
                    self.assertEqual(active.health()['limits']['longest_edge'],spec.longest_edge)
                    self.assertEqual(loader.from_pretrained.call_args.args,(str(spec.directory),))
                    if variant=='1B': self.assertIn('key_mapping',loader.from_pretrained.call_args.kwargs)
                    else: self.assertNotIn('key_mapping',loader.from_pretrained.call_args.kwargs)
                    loader.from_pretrained.return_value=(model,{**clean,'missing_keys':['broken']})
                    with self.assertRaisesRegex(RuntimeError,'refusing to start'): engine.Engine(variant)

    def test_api_lists_and_accepts_only_active_model(self):
        buf=io.BytesIO(); Image.new('RGB',(2400,800),'white').save(buf,'PNG')
        image_url='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()
        for variant,spec in model_catalog.MODELS.items():
            with self.subTest(variant=variant):
                fake=SimpleNamespace(model_id=spec.model_id,longest_edge=spec.longest_edge,inference_lock=threading.Lock(),last_prompt_tokens=1)
                sizes=[]
                def transcribe(image,*args,**kwargs): sizes.append(image.size); return 'test',1,False
                fake.transcribe=transcribe; fake.health=lambda:{'model':spec.model_id,'status':'ready'}
                with TestClient(adapter.create_app(fake),base_url='http://127.0.0.1:8123') as client:
                    self.assertEqual([item['id'] for item in client.get('/v1/models').json()['data']],[spec.model_id])
                    body={'model':spec.model_id,'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':image_url}}]}]}
                    response=client.post('/v1/chat/completions',json=body)
                    self.assertEqual(response.status_code,200,response.text)
                    self.assertEqual(response.json()['model'],spec.model_id)
                    self.assertEqual(max(sizes[0]),spec.longest_edge)
                    for other in model_catalog.MODELS.values():
                        if other != spec:
                            body['model']=other.model_id
                            self.assertEqual(client.post('/v1/chat/completions',json=body).status_code,400)

if __name__=='__main__': unittest.main(verbosity=2)
