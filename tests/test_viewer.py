"""Original viewer contract/security tests using a fake inference engine.

Runs the real local adapter and the vendored upstream client/pipeline. No model
weights are loaded. Run: app/env/bin/python tests/test_viewer.py
"""
import io,json,socket,sys,tempfile,threading,time
from unittest.mock import patch
from pathlib import Path
APP = Path(__file__).resolve().parents[1] / 'app'
sys.path.insert(0, str(APP))
import httpx
from PIL import Image
import server

class FakeEngine:
    def __init__(self): self.calls=[]
    def health(self): return {'status':'ready', 'backend':'fake', 'model':server.MODEL_ID}
    def transcribe(self,image,mode,max_new_tokens,temperature=0.1,top_p=0.9):
        self.calls.append((image.size,mode,max_new_tokens,temperature,top_p))
        if mode == 'grounding':
            return '![title](10,20,900,150) Example title\n![text](10,170,900,500) **Bold** and $x^2$\n![table](10,500,900,900) <table><tr><td>**Total**</td></tr></table>',45,False
        return '# Example\n\n**Bold** and $x^2$',15,False

def check(condition, text):
    assert condition,text
    print('PASS',text)


def main():
    server.check_viewer_assets()
    check(True,'required local viewer assets exist')
    with tempfile.TemporaryDirectory(prefix='missing-viewer-assets-') as empty:
        with patch.object(server,'STATIC',Path(empty)):
            try:
                server.check_viewer_assets()
            except RuntimeError as error:
                check('Run Install or Update' in str(error),'missing assets fail with actionable startup message')
            else:
                raise AssertionError('Missing asset check did not fail')
    engine=FakeEngine(); backend=server.LocalBackend(engine); backend.start(); ocr=server.make_client(backend.url+'/v1')
    with tempfile.TemporaryDirectory(prefix='lightonocr-wrapper-test-') as tmp:
        viewer=server.LocalViewerServer(('127.0.0.1',0),Path(tmp),ocr,engine,backend.url+'/v1')
        thread=threading.Thread(target=viewer.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{viewer.server_port}'
        with httpx.Client(base_url=base,trust_env=False,timeout=10) as client:
            response=client.get('/')
            check(response.status_code==200 and 'id="viewer"' in response.text and 'https://cdn.jsdelivr.net' not in response.text,'original HTML with local libraries')
            for asset in ['/static/app.css','/static/app.js','/static/lib/marked.min.js','/static/lib/purify.min.js','/static/lib/katex/katex.min.css','/static/lib/katex/katex.min.js','/static/lib/katex/contrib/auto-render.min.js','/static/lib/katex/fonts/KaTeX_Main-Regular.woff2']:
                check(client.get(asset).status_code==200,'local asset '+asset)
            check('frame-ancestors' in response.headers['content-security-policy'] and response.headers['x-content-type-options']=='nosniff','security headers')
            check(client.get('/healthz').json()['adapter_url']==backend.url+'/v1','viewer health reports shared local backend')
            settings=client.get('/api/settings').json()
            check(settings['reachable'] and settings['grounding'] and settings['model']==server.MODEL_ID,'original model settings probe')
            check(client.put('/api/settings',json={'base_url':'https://example.com/v1'}).status_code==400,'external endpoint blocked')
            check(client.put('/api/settings',json={'model':'different/model'}).status_code==400,'uninstalled model blocked')
            check(client.put('/api/settings',json={}).status_code==200,'empty settings retain local endpoint')
            check(client.put('/api/settings',json={'base_url':backend.url+'/v1','model':server.MODEL_ID}).status_code==200,'displayed settings save')
            check(client.get('/api/runs',headers={'host':'evil.example'}).status_code==400,'DNS-rebinding Host blocked')
            check(client.post('/api/runs',content=b'a',headers={'origin':'https://evil.example'}).status_code==403,'cross-origin upload blocked')
            check(client.delete('/api/jobs/abcd',headers={'origin':'https://evil.example'}).status_code==403,'cross-origin deletion blocked')
            check(client.put('/api/settings',json={},headers={'sec-fetch-site':'cross-site'}).status_code==403,'cross-site settings blocked')
            check(client.post('/api/runs',content=b'a',headers={'origin':base}).status_code==400,'same-origin upload reaches byte validation')
            check(client.get('/static/%2e%2e/server.py').status_code==404,'static traversal blocked')
            check(client.get('/static/lib/%2e%2e/%2e%2e/server.py').status_code==404,'nested traversal blocked')
            check(client.post('/api/runs',content=b'bad').status_code==400,'invalid image rejected')
            image=Image.new('RGB',(100,200),'white');buf=io.BytesIO();image.save(buf,'PNG');png=buf.getvalue()
            check(client.post('/api/runs?pages=1-99999999999999',content=png).status_code==400,'huge page range bounded')
            check(client.post('/api/runs?pages=0',content=png).status_code==400,'page zero rejected')
            check(client.post('/api/runs?pages=1,1',content=png).status_code==400,'duplicate page selection rejected')
            check(client.post('/api/runs?pages=2',content=png).status_code==400,'image selection out of range rejected')
            check(client.post('/api/runs?mode=unsupported',content=png).status_code==400,'invalid OCR mode rejected')
            def run_upload(data, filename, name, mode='grounding', pages=''):
                response=client.post('/api/runs',params={'name':name,'mode':mode,'pages':pages},content=data,headers={'X-Filename':filename})
                check(response.status_code==201,'run upload '+filename)
                job=response.json(); deadline=time.monotonic()+10
                while time.monotonic()<deadline:
                    jobs=client.get('/api/jobs').json(); job=next(j for j in jobs if j['id']==job['id'])
                    if job['status'] != 'running':break
                    time.sleep(.02)
                check(job['status']=='done','completed '+filename+' '+str(job))
                return job,client.get('/api/runs/'+job['name']).json()
            job,run=run_upload(png,'test.png','test')
            check(run['pages'][0]['blocks'][0]['label']=='title','grounded boxes parsed through original pipeline')
            check(run['pages'][0]['width']==100 and run['pages'][0]['height']==200,'grounded image dimensions preserved')
            check(client.get(run['pages'][0]['image']).status_code==200,'original run image endpoint')
            check(client.get('/files/'+job['name']+'/source.png').content==png,'upload saved for history')
            job2,run2=run_upload(png,'another.png','test','plain')
            check(job2['name']=='test-2' and run2['pages'][0]['blocks'] is None,'unique run names and plain mode')
            pdf=io.BytesIO();image.save(pdf,format='PDF',save_all=True,append_images=[image,image])
            job3,run3=run_upload(pdf.getvalue(),'multipage.pdf','pdf','grounding','1,3')
            check([p['page'] for p in run3['pages']]==[1,3],'original selected PDF page numbers')
            check(client.post('/api/runs?pages=4',content=pdf.getvalue(),headers={'X-Filename':'multipage.pdf'}).status_code==400,'PDF out-of-range selection rejected')
            pdf21=io.BytesIO();image.save(pdf21,format='PDF',save_all=True,append_images=[image]*20)
            check(client.post('/api/runs',content=pdf21.getvalue()).status_code==400,'21-page PDF rejected')
            check(client.delete('/api/runs/'+job['name']).status_code==204,'original delete run')
            check(client.get('/api/runs/'+job['name']).status_code==404,'deleted run disappears')
            check(client.delete('/api/jobs/'+job['id']).status_code==204,'original dismiss job')
            check(not any(j['id']==job['id'] for j in client.get('/api/jobs').json()),'dismissed job disappears')
            alpha=Image.new('RGBA',(30,40),(0,0,0,0));alpha_buf=io.BytesIO();alpha.save(alpha_buf,'PNG')
            alpha_bytes=alpha_buf.getvalue()
            alpha_job,alpha_run=run_upload(alpha_bytes,'transparent.png','alpha')
            with Image.open(io.BytesIO(client.get(alpha_run['pages'][0]['image']).content)) as rendered:
                check(rendered.getpixel((0,0))==(255,255,255),'transparent input receives white background before upstream RGB conversion')
            check(client.get('/files/'+alpha_job['name']+'/source.png').content==alpha_bytes,'original transparent source retained')
            check(not (Path(tmp)/alpha_job['name']/'.normalized-input.png').exists(),'temporary normalized input cleaned up')
            rotated=Image.new('RGB',(80,40),'white');exif=rotated.getexif();exif[274]=6
            exif_buf=io.BytesIO();rotated.save(exif_buf,'JPEG',exif=exif)
            exif_job,exif_run=run_upload(exif_buf.getvalue(),'oriented.jpg','exif')
            check((exif_run['pages'][0]['width'],exif_run['pages'][0]['height'])==(40,80),'EXIF orientation applied before upstream rendering')
            check(len(engine.calls)==6,'one shared engine handled images, plain, and selected PDF pages')
            check(all(c[3]==.1 for c in engine.calls),'upstream pipeline uses requested temperature')
            # Send headers only: invalid lengths must fail before reading a body.
            def raw(headers):
                with socket.create_connection(('127.0.0.1',viewer.server_port),timeout=3) as sock:
                    sock.sendall((f'POST /api/runs HTTP/1.0\r\nHost: 127.0.0.1:{viewer.server_port}\r\n'+headers+'\r\n\r\n').encode())
                    return sock.recv(2048).decode(errors='replace')
            check(' 413 ' in raw('Content-Length: 26214401'),'25 MB upload bound before reading')
            check(' 400 ' in raw('Content-Length: not-a-number'),'invalid content length rejected')
            check(' 400 ' in raw('Content-Length: 1\r\nContent-Length: 1'),'duplicate content length rejected')
            check(' 400 ' in raw('Transfer-Encoding: chunked'),'unsupported chunked request rejected')
        viewer.shutdown();viewer.server_close();thread.join(timeout=2)
    ocr.client.close();backend.close()
    print('ALL VIEWER INTEGRATION CHECKS PASSED')

if __name__ == "__main__":
    main()
