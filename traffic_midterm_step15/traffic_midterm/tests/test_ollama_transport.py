import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from traffic.llm import Ollama, ModelError

class Stub(BaseHTTPRequestHandler):
    mode='ok'
    def log_message(self,*args):pass
    def reply(self,v):
        b=json.dumps(v).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    def do_GET(self):
        if self.path=='/api/tags':self.reply({'models':[{'name':'qwen2.5:3b-instruct','digest':'MOCK','remote_host':'remote.invalid'}] if self.mode=='remote' else [{'name':'qwen2.5:3b-instruct','digest':'MOCK'}]})
        else:self.reply({'version':'MOCK'})
    def do_POST(self):
        request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.last_body=request
        self.reply({'done':True,'done_reason':'length' if self.mode=='length' else 'stop','response':'{bad' if self.mode=='invalid' else '{"results":[]}'})

class TransportTests(unittest.TestCase):
    def setUp(self):
        Stub.mode='ok';self.server=ThreadingHTTPServer(('127.0.0.1',0),Stub)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.patch=patch.object(Ollama,'endpoint',f'http://127.0.0.1:{self.server.server_port}');self.patch.start()
    def tearDown(self):self.patch.stop();self.server.shutdown();self.server.server_close();self.thread.join()
    def test_transport_json_schema(self):
        client=Ollama();self.assertEqual(client.status()['digest'],'MOCK');self.assertEqual(client.generate('test',{'type':'object'}),{'results':[]});self.assertFalse(self.server.last_body['stream']);self.assertEqual(self.server.last_body['options']['temperature'],0)
    def test_remote_refused(self):
        Stub.mode='remote'
        with self.assertRaises(ModelError):Ollama().status()
    def test_truncation_retains_raw(self):
        Stub.mode='length';client=Ollama()
        with self.assertRaises(ModelError):client.generate('test',{})
        self.assertEqual(len(client.calls),1)
    def test_malformed_json_retains_raw(self):
        Stub.mode='invalid';client=Ollama()
        with self.assertRaises(ModelError):client.generate('test',{})
        self.assertEqual(client.calls[0]['response']['response'],'{bad')
