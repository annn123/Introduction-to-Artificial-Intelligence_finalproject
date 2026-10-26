import json
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from traffic.core import read_json
from traffic.llm import results_schema, validate_results, ModelError, classify
from traffic.server import Handler


def result():return {'label':'Insufficient Evidence','reason':'缺少證據','evidence_ids':[],'source_ids':[]}

class KeyedResultsTests(unittest.TestCase):
    def setUp(self):self.claims=read_json('demo.json')['claims']
    def test_schema_requires_all_exact_keys(self):
        schema=results_schema(self.claims)['properties']['results']
        self.assertEqual(schema['type'],'object');self.assertEqual(schema['required'],['C1','C2']);self.assertIs(schema['additionalProperties'],False)
    def test_result_association_uses_key_not_order(self):
        value={'results':{'C2':{**result(),'reason':'second'},'C1':{**result(),'reason':'first'}}}
        rows=validate_results(value,self.claims)
        self.assertEqual([x['id'] for x in rows],['C1','C2']);self.assertEqual(rows[0]['reason'],'first')
    def test_missing_and_extra_still_rejected(self):
        for value in [{'C1':result()},{'C1':result(),'C2':result(),'C3':result()}]:
            with self.subTest(value=value),self.assertRaises(ModelError):validate_results({'results':value},self.claims)
    def test_conflicting_embedded_id_rejected(self):
        with self.assertRaises(ModelError):validate_results({'results':{'C1':{**result(),'id':'C2'},'C2':result()}},self.claims)
    def test_null_row_rejected(self):
        with self.assertRaises(ModelError):validate_results({'results':{'C1':None,'C2':result()}},self.claims)
    def test_single_cv_claim_schema(self):
        self.assertEqual(results_schema([self.claims[1]])['properties']['results']['required'],['C2'])
    def test_no_claim_can_be_silently_filled(self):
        with self.assertRaises(ModelError):validate_results({'results':{}},self.claims)
    def test_classification_keyed_transport_all_modes(self):
        class Client:
            model='MOCK_NOT_A_MODEL'
            calls=[]
            def generate(self,prompt,schema):return {'results':{key:result() for key in schema['properties']['results']['required']}}
        for mode in ['llm_only','rag','rag_verify']:
            rows=classify(read_json('demo.json'),mode,Client())['claims']
            self.assertEqual([c['id'] for c in rows],['C1','C2'])

class ErrorRecordTests(unittest.TestCase):
    def test_failed_model_output_is_saved_without_marking_valid(self):
        class BrokenClient:
            model='MOCK_NOT_A_MODEL'
            calls=[{'response':{'response':'{"results":[]}'}}]
            def __init__(self,*args):pass
            def status(self):return {}
        with tempfile.TemporaryDirectory() as tmp,patch('traffic.server.ROOT',Path(tmp)),patch('traffic.server.Ollama',BrokenClient),patch('traffic.server.model_pipeline',side_effect=ModelError('missing C2')):
            server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                req=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/api/analyze',data=json.dumps({**read_json('demo.json'),'mode':'llm_only'}).encode(),headers={'Content-Type':'application/json'})
                with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(req)
                self.assertEqual(error.exception.code,503)
                response=json.load(error.exception);record=json.loads((Path(tmp)/response['debug_file']).read_text(encoding='utf8'))
                self.assertEqual(record['status'],'FAILED_NOT_A_VALID_SAMPLE');self.assertEqual(record['calls'],BrokenClient.calls)
            finally:server.shutdown();server.server_close();thread.join()
