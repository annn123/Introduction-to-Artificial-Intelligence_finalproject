from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from .core import ROOT, Retriever, analyze, read_json
from .llm import Ollama, ModelError, classify, DEFAULT_MODEL
from .extraction import extract_case
from .pipeline import model_pipeline

class Handler(BaseHTTPRequestHandler):
    retriever=Retriever()
    def send_json(self,status,value):
        data=json.dumps(value,ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(data)))
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers();self.wfile.write(data)
    def trusted_request(self):
        expected=f'127.0.0.1:{self.server.server_port}'
        if self.headers.get('Host') not in (expected,f'localhost:{self.server.server_port}'):
            return False
        origin=self.headers.get('Origin')
        return origin is None or origin in (f'http://{expected}',f'http://localhost:{self.server.server_port}')
    def do_GET(self):
        if not self.trusted_request():return self.send_json(403,{'error':'Local origin required'})
        path=urlsplit(self.path).path
        if path=='/api/demo':return self.send_json(200,read_json('demo.json'))
        if path=='/api/corpus':return self.send_json(200,self.retriever.documents)
        if path=='/api/health':return self.send_json(200,{'ok':True,'engine':'offline_rules','documents':len(self.retriever.documents),'paid_api':False})
        if path not in ('/','/index.html'):return self.send_json(404,{'error':'Not found'})
        body=(ROOT/'static/index.html').read_bytes()
        self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
        self.send_header('Content-Length',str(len(body)))
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers();self.wfile.write(body)
    def do_POST(self):
        if not self.trusted_request():return self.send_json(403,{'error':'Local origin required'})
        if self.path not in ('/api/analyze','/api/extract'):return self.send_json(404,{'error':'Not found'})
        if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.send_json(415,{'error':'application/json required'})
        client = None
        payload = None
        try:
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<=150000:raise ValueError('Request body must be 1–150000 bytes')
            payload=json.loads(self.rfile.read(n))
            if not isinstance(payload,dict):raise ValueError('JSON object required')
            mode=payload.get('mode','offline')
            if self.path=='/api/extract':
                client=Ollama(payload.get('model',DEFAULT_MODEL));status=client.status()
                result=extract_case(payload.get('description',''),client);result['model_status']=status
            elif mode=='offline':result=analyze(payload,self.retriever)
            elif mode in ('llm_only','rag','rag_verify'):
                client=Ollama(payload.get('model',DEFAULT_MODEL));status=client.status()
                result=model_pipeline(payload,mode,client,self.retriever);result['model_status']=status
            else:raise ValueError('Invalid mode')
            self.send_json(200,result)
        except (ValueError,TypeError) as exc:self.send_json(400,{'error':str(exc)})
        except ModelError as exc:
            response = {'error': str(exc)}
            if client is not None:
                try:
                    directory = ROOT/'results/model_errors'
                    directory.mkdir(parents=True, exist_ok=True)
                    filename = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]+'.json'
                    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                              'error': str(exc), 'input': payload, 'model': client.model,
                              'calls': client.calls, 'status': 'FAILED_NOT_A_VALID_SAMPLE'}
                    (directory/filename).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
                    response['debug_file'] = 'results/model_errors/'+filename
                except (OSError, TypeError, ValueError):
                    response['debug_save_failed'] = True
            self.send_json(503,response)
    def log_message(self,format,*args):
        # Do not print accident descriptions or evidence in request logs.
        pass

def serve(port=8765):
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'開啟 http://127.0.0.1:{server.server_port} ；Ctrl+C 結束。付費 API：未使用。',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
