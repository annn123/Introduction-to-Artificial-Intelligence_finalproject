import copy
import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from traffic.core import analyze, mediation, read_json, verify_claim
from traffic.extraction import validate_extraction, extract_case, complete_reasoning_chain
from traffic.llm import ModelError, classify, PROMPT_VERSION
from traffic.pipeline import model_pipeline
from traffic.server import Handler

DESCRIPTION='我騎機車直行，對方開汽車左轉。對方說我無照，所以我應負全部責任。我有行車紀錄器。'

def extracted():
    return {'case_fields':[
        {'field':'A.vehicle','value':'機車','source_quote':'我騎機車直行'},
        {'field':'A.action','value':'直行','source_quote':'我騎機車直行'},
        {'field':'B.vehicle','value':'汽車','source_quote':'對方開汽車左轉'},
        {'field':'B.action','value':'左轉','source_quote':'對方開汽車左轉'},
        {'field':'evidence_mention','value':'行車紀錄器','source_quote':'我有行車紀錄器'}],
        'claims':[
            {'text':'A無照','subject':'A','kind':'fact','predicate':'unlicensed','polarity':True,'source_quote':'對方說我無照','speaker':'B','premise_indices':[]},
            {'text':'A應負全部責任','subject':'A','kind':'liability','predicate':'liability:100','polarity':True,'source_quote':'所以我應負全部責任','speaker':'B','premise_indices':[1]}]}

class ExtractionTests(unittest.TestCase):
    def test_three_levels_and_correct_dependencies(self):
        r=validate_extraction(extracted(),DESCRIPTION)
        self.assertEqual([c['kind'] for c in r['claims']],['fact','causal','liability'])
        self.assertEqual([c['id'] for c in r['claims']],['C1','C2','C3'])
        self.assertEqual(r['claims'][1]['derived_from'],['C1']);self.assertEqual(r['claims'][2]['derived_from'],['C1']);self.assertEqual(r['claims'][2]['required_checks'],['C2'])
        self.assertEqual(r['claims'][1]['origin'],'generated_check_question');self.assertEqual(r['claims'][1]['source_quote'],'')
        self.assertEqual(r['claims'][0]['subject'],'A');self.assertEqual(r['claims'][0]['speaker'],'B')
    def test_video_mention_is_not_evidence(self):
        r=validate_extraction(extracted(),DESCRIPTION)
        self.assertEqual(r['case']['evidence_mentions'],['行車紀錄器']);self.assertEqual(r['evidence'],[])
        self.assertEqual(verify_claim(r['claims'][0],[])['label'],'Insufficient Evidence')
    def test_unknown_signal_stays_unknown(self):
        case=validate_extraction(extracted(),DESCRIPTION)['case'];self.assertEqual(case['signals'],{'A':None,'B':None});self.assertIsNone(case['right_of_way'])
    def test_vehicles_and_actions_from_quotes(self):
        case=validate_extraction(extracted(),DESCRIPTION)['case'];self.assertEqual(case['participants']['A']['vehicle'],'機車');self.assertEqual(case['participants']['B']['actions'],['左轉'])
    def test_fabricated_claim_quote_rejected(self):
        x=extracted();x['claims'][0]['source_quote']='警方證實A無照'
        with self.assertRaises(ModelError):validate_extraction(x,DESCRIPTION)
    def test_fabricated_field_quote_rejected(self):
        x=extracted();x['case_fields'][0]['source_quote']='我開卡車'
        with self.assertRaises(ModelError):validate_extraction(x,DESCRIPTION)
    def test_field_value_not_in_quote_rejected(self):
        x=extracted();x['case_fields'][0]['value']='卡車'
        with self.assertRaises(ModelError):validate_extraction(x,DESCRIPTION)
    def test_ambiguous_field_actor_abstains(self):
        x=extracted();x['case_fields'][0]['source_quote']='機車直行'
        case=validate_extraction(x,DESCRIPTION)['case'];self.assertIsNone(case['participants']['A']['vehicle']);self.assertTrue(case['review_warnings'])
    def test_conflicting_vehicle_remains_unknown(self):
        desc=DESCRIPTION+'我開汽車，我騎機車。';x=extracted();x['case_fields'] += [{'field':'A.vehicle','value':'汽車','source_quote':'我開汽車'},{'field':'A.vehicle','value':'機車','source_quote':'我騎機車'}]
        self.assertIsNone(validate_extraction(x,desc)['case']['participants']['A']['vehicle'])
    def test_forward_reference_rejected(self):
        x=extracted();x['claims'][0]['premise_indices']=[2]
        with self.assertRaises(ModelError):validate_extraction(x,DESCRIPTION)
    def test_self_reference_rejected(self):
        x=extracted();x['claims'][0]['premise_indices']=[1]
        with self.assertRaises(ModelError):validate_extraction(x,DESCRIPTION)
    def test_invalid_predicate_or_type_rejected(self):
        for kind,pred in [('fact','cause:unlicensed'),('causal','unlicensed'),('liability','speeding')]:
            x=extracted();x['claims'][0].update(kind=kind,predicate=pred)
            with self.subTest(kind=kind),self.assertRaises(ModelError):validate_extraction(x,DESCRIPTION)
    def test_no_invented_relation_without_explicit_premise(self):
        x=extracted();x['claims'][1]['premise_indices']=[]
        self.assertEqual(len(validate_extraction(x,DESCRIPTION)['claims']),2)
    def test_unknown_actor_does_not_get_causal_question(self):
        x=extracted();x['claims'][0]['subject']='unknown'
        self.assertEqual(len(validate_extraction(x,DESCRIPTION)['claims']),2)
    def test_negative_fact_does_not_become_positive_causal_check(self):
        x=extracted();x['claims'][0]['polarity']=False
        self.assertEqual(len(validate_extraction(x,DESCRIPTION)['claims']),2)
    def test_existing_causal_claim_not_duplicated(self):
        x=extracted();x['claims'].insert(1,{'text':'A無照造成事故','subject':'A','kind':'causal','predicate':'cause:unlicensed','polarity':True,'source_quote':'對方說我無照造成事故','speaker':'B','premise_indices':[1]})
        r=validate_extraction(x,DESCRIPTION+'對方說我無照造成事故。');self.assertEqual(len(r['claims']),3)
        self.assertFalse(any(c.get('origin')=='generated_check_question' for c in r['claims']))
    def test_fabricated_causal_assertion_rejected_even_when_quote_exists(self):
        x=extracted();x['claims'][0].update(kind='causal',predicate='cause:unlicensed')
        with self.assertRaises(ModelError):validate_extraction(x,DESCRIPTION)
    def test_compound_fact_rejected(self):
        x=extracted();x['claims'][0]['text']='A無照且超速'
        with self.assertRaises(ModelError):validate_extraction(x,DESCRIPTION)
    def test_input_not_mutated(self):
        x=extracted();before=copy.deepcopy(x);validate_extraction(x,DESCRIPTION);self.assertEqual(x,before)
    def test_extraction_does_not_promote_statement_to_evidence(self):
        r=validate_extraction(extracted(),DESCRIPTION)
        result=analyze({'description':DESCRIPTION,'claims':r['claims'],'evidence':[]})
        self.assertEqual([c['label'] for c in result['claims']],['Insufficient Evidence','Insufficient Evidence','Overgeneralized'])

class FakeClient:
    model='MOCK_NOT_A_REAL_MODEL'
    def __init__(self):self.calls=[]
    def generate(self,prompt,schema,**kwargs):
        self.calls.append({'prompt':prompt,'mock':True})
        if 'case_fields' in schema['properties']:
            values={key:{'value':'','source_quote':''} for key in schema['properties']['case_fields']['required']}
            for f in extracted()['case_fields']:values[f['field']]={k:f[k] for k in ('value','source_quote')}
            return {'case_fields':values}
        if 'candidate_review' in schema['properties']:
            return {'candidate_review':{key:{field:spec['enum'][0] for field,spec in row['properties'].items()} for key,row in schema['properties']['candidate_review']['properties'].items()}}
        if 'claim_groups' in schema['properties']:
            groups={key:[] for key in ('fact','causal','liability','irrelevant')}
            for c in extracted()['claims']:groups[c['kind']].append({k:v for k,v in c.items() if k!='kind'})
            return {'claim_groups':groups}
        return {'results':{key:{'label':'Insufficient Evidence','reason':'缺少本案證據','evidence_ids':[],'source_ids':[]} for key in schema['properties']['results']['required']}}

class PipelineTests(unittest.TestCase):
    def test_automatic_extraction_then_classification(self):
        client=FakeClient();r=model_pipeline({'description':DESCRIPTION},'llm_only',client)
        self.assertEqual(len(client.calls),2);self.assertEqual(len(r['claims']),3);self.assertEqual(r['case']['extraction_method'],'hybrid_rules_with_local_role_review')
        self.assertEqual(r['extraction']['generated_check_questions'],['C2']);self.assertEqual(r['prompt_version'],PROMPT_VERSION)
    def test_fixed_claim_path_does_not_extract(self):
        client=FakeClient();r=model_pipeline(read_json('demo.json'),'llm_only',client)
        self.assertEqual(len(client.calls),1);self.assertEqual(r['extraction']['method'],'supplied_claims')
    def test_cv_keeps_raw_outputs_and_gate_changes(self):
        client=FakeClient();r=model_pipeline({'description':DESCRIPTION},'rag_verify',client)
        self.assertEqual(len(client.calls),4)
        self.assertEqual(r['claims'][2]['raw_model_result']['label'],'Insufficient Evidence');self.assertEqual(r['claims'][2]['label'],'Overgeneralized')
    def test_raw_extraction_is_saved_in_calls(self):
        client=FakeClient();r=extract_case(DESCRIPTION,client);self.assertEqual(len(r['calls']),1)
        self.assertNotIn('claims',r['case'])
    def test_mediator_does_not_turn_fact_into_responsibility(self):
        c=read_json('demo.json')['claims'][0];c['label']='Overgeneralized'
        text=mediation([c])['text'];self.assertIn('事實指控',text);self.assertNotIn('C1 的全部責任推論',text)
    def test_mediator_does_not_declare_model_supported_as_verified_fact(self):
        c=read_json('demo.json')['claims'][0];c['label']='Supported'
        self.assertIn('輸出標示支持',mediation([c])['text']);self.assertNotIn('與提供的證據轉錄相符',mediation([c])['text'])

class ExtractionEndpointTests(unittest.TestCase):
    def test_extract_endpoint_and_auto_analyze(self):
        class Client(FakeClient):
            def __init__(self,*args):super().__init__()
            def status(self):return {'model':self.model}
        with patch('traffic.server.Ollama',Client):
            server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                for path,body in [('/api/extract',{'description':DESCRIPTION}),('/api/analyze',{'description':DESCRIPTION,'mode':'rag_verify'})]:
                    request=urllib.request.Request(f'http://127.0.0.1:{server.server_port}'+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(request) as response:r=json.load(response)
                    self.assertEqual(len(r['claims']),3)
            finally:server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()
