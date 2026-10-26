import copy
import json
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from traffic.core import Retriever, analyze, extract_claims, infer_claim, read_json, validate_payload, verify_claim, structure_case
from traffic.evaluate import classification_metrics, retrieval_metrics, review_metrics, run
from traffic.llm import Ollama, ModelError, classify, validate_results
from traffic.server import Handler


def fact(text='A超速'):
    return {'id':'C1','text':text,'subject':'A','predicate':'speeding','kind':'fact','polarity':True}

def ev(polarity=True,reviewed=True,subject='A'):
    return {'id':'E1','source_type':'transcript','description':'合成證據','reviewed':reviewed,'assertions':[{'subject':subject,'predicate':'speeding','polarity':polarity}]}

class CoreTests(unittest.TestCase):
    def test_missing_evidence(self):self.assertEqual(verify_claim(fact(),[])['label'],'Insufficient Evidence')
    def test_reviewed_evidence(self):self.assertEqual(verify_claim(fact(),[ev()])['label'],'Supported')
    def test_unreviewed_video(self):self.assertEqual(verify_claim(fact(),[ev(reviewed=False)])['label'],'Insufficient Evidence')
    def test_other_party_not_support(self):self.assertEqual(verify_claim(fact(),[ev(subject='B')])['label'],'Insufficient Evidence')
    def test_negative_words_inside_violation_predicate(self):
        for text in ['A並未保持安全距離','A並未讓直行車','A沒有駕照','A未領駕照']:
            with self.subTest(text=text):self.assertTrue(infer_claim(text)['polarity'])
        for text in ['A並非超速','A未闖紅燈','A沒有酒駕','A未變換車道']:
            with self.subTest(text=text):self.assertFalse(infer_claim(text)['polarity'])
    def test_negative_evidence(self):self.assertTrue(verify_claim(fact(),[ev(False)])['conflict'])
    def test_conflicting_evidence(self):
        e=ev(False);e['id']='E2'
        self.assertEqual(verify_claim(fact(),[ev(),e])['label'],'Insufficient Evidence')
    def test_negative_claim(self):
        c=infer_claim('A沒有超速');self.assertFalse(c['polarity']);self.assertEqual(verify_claim(c,[ev(False)])['label'],'Supported')
    def test_composite_not_supported_by_one_fact(self):
        self.assertEqual(verify_claim(infer_claim('A超速且闖紅燈'),[ev()])['label'],'Insufficient Evidence')
    def test_uncertain_claim_not_verified(self):
        self.assertEqual(verify_claim(infer_claim('A可能超速'),[ev()])['label'],'Insufficient Evidence')
    def test_reporting_actor_ambiguous(self):
        self.assertEqual(infer_claim('B主張A超速')['subject'],'unknown')
    def test_causal_not_implied(self):
        c=fact();c.update(kind='causal',predicate='cause:speeding')
        self.assertEqual(verify_claim(c,[ev()])['label'],'Insufficient Evidence')
    def test_full_liability_overreach(self):
        cs=extract_claims('B無照，所以B應負全部責任。')
        self.assertEqual(len(cs),2);self.assertEqual(verify_claim(cs[1],[])['label'],'Overgeneralized')
    def test_negative_liability_not_overgeneralized(self):
        c=infer_claim('B並非一定全責');self.assertEqual(verify_claim(c,[])['label'],'Insufficient Evidence')
    def test_judgment_cannot_be_case_evidence(self):
        c=infer_claim('B一定全責')
        e=ev();e['assertions']=[{'subject':'B','predicate':'liability:100','polarity':True}]
        self.assertEqual(verify_claim(c,[e])['label'],'Overgeneralized')
        e['source_type']='court_finding';self.assertEqual(verify_claim(c,[e])['label'],'Supported')
    def test_actor_schema(self):
        result=structure_case('A騎機車直行，B開汽車左轉。我有行車紀錄器。')
        self.assertEqual(result['participants']['A']['vehicle'],'機車');self.assertEqual(result['participants']['B']['actions'],['左轉']);self.assertIsNone(result['traffic_signal'])
    def test_unknown_actor_not_inherited(self):
        self.assertEqual(extract_claims('B無照，所以應負全部責任')[1]['subject'],'unknown')
    def test_unknown_kept_for_review(self):self.assertEqual(extract_claims('事故發生於昨天')[0]['predicate'],'unknown')
    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):validate_payload({'description':'事故','claims':[fact(),fact()]})
    def test_boolean_strings_rejected(self):
        c=fact();c['polarity']='false'
        with self.assertRaises(ValueError):validate_payload({'description':'事故','claims':[c]})
    def test_evidence_assertion_required_bool(self):
        e=ev();e['assertions'][0]['polarity']='false'
        with self.assertRaises(ValueError):validate_payload({'description':'事故','evidence':[e]})
    def test_payload_not_object(self):
        with self.assertRaises(ValueError):validate_payload([])
    def test_no_retrieval_hit(self):self.assertEqual(Retriever().search('zzzzxyz'),[])
    def test_bm25_ties_stable(self):
        docs=[dict(id=x,kind='law',title='測試',text='安全距離') for x in ['b','a']]
        self.assertEqual([x['id'] for x in Retriever(docs).search('安全距離')],['a','b'])
    def test_type_filter(self):self.assertTrue(all(x['kind']=='law' for x in Retriever().search('車門','law')))
    def test_invalid_k(self):
        with self.assertRaises(ValueError):Retriever().search('車門',k=0)
    def test_corpus_integrity(self):
        docs=read_json('corpus.json');self.assertEqual(len(docs),22);self.assertEqual(sum(d['kind']=='case' for d in docs),10)
        self.assertTrue(all(d['url'].startswith('https://') for d in docs))
    def test_dataset_labels_length_and_ids(self):
        for c in read_json('cases.json'):
            validate_payload(c);self.assertEqual(len(c['claims']),len(c['silver_labels']));self.assertFalse(c['human_reviewed'])
    def test_input_not_mutated(self):
        payload=read_json('demo.json');before=copy.deepcopy(payload);analyze(payload);self.assertEqual(payload,before)

class MetricTests(unittest.TestCase):
    def test_macro_f1_missing_label(self):
        m=classification_metrics(['Supported'],['Supported']);self.assertEqual(m['macro_f1_present_gold_labels'],1);self.assertEqual(m['macro_f1_all_five_labels'],.2)
    def test_metric_empty(self):self.assertIsNone(classification_metrics([],[])['accuracy'])
    def test_precision_denominator_k(self):self.assertAlmostEqual(retrieval_metrics(['a'],['a'],3)['precision_at_k'],1/3)
    def test_no_relevance_unscored(self):self.assertIsNone(retrieval_metrics([],['a'],3)['recall_at_k'])
    def test_offline_is_not_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=run(tmp);self.assertEqual(r['metadata']['llm_experiment_status'],'NOT_RUN');self.assertNotIn('llm_only',r['classification'])
    def test_empty_human_gold_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'gold.json';p.write_text(json.dumps(read_json('annotation_template.json')))
            with self.assertRaises(ValueError):run(tmp,gold_path=p)
    def test_unfilled_manual_review_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run(tmp)
            with self.assertRaises(ValueError):review_metrics(Path(tmp)/'manual_output_review.csv')

class FakeClient:
    model='mock_NOT_A_REAL_LLM'
    def __init__(self,invalid=False):self.calls=[];self.invalid=invalid
    def generate(self,prompt,schema):
        self.calls.append({'prompt':prompt,'mock':True})
        tail=prompt.split('本次僅輸出這個主張：')[-1]
        c=json.loads(tail) if '本次僅輸出這個主張：' in prompt else json.loads(prompt.split('輸入資料：\n')[-1])['claims'][0]
        return {'results':[{'id':c['id'],'label':'Supported','reason':'MOCK','evidence_ids':[],'source_ids':['FAKE'] if self.invalid else []}]}

class LLMTests(unittest.TestCase):
    def test_cloud_model_rejected(self):
        with self.assertRaises(ValueError):Ollama('qwen3-cloud')
    def test_missing_ids_rejected(self):
        with self.assertRaises(ModelError):validate_results({'results':[]},[fact()])
    def test_duplicate_ids_rejected(self):
        r=dict(id='C1',label='Supported',reason='test',evidence_ids=[],source_ids=[])
        with self.assertRaises(ModelError):validate_results({'results':[r,r]},[fact()])
    def test_cv_blocks_unsupported_fact(self):
        result=classify({'description':'A超速','claims':[fact()]},'rag_verify',FakeClient())
        self.assertEqual(result['claims'][0]['label'],'Insufficient Evidence');self.assertTrue(result['claims'][0]['gate_changed_label'])
    def test_baseline_raw_kept(self):
        result=classify({'description':'A超速','claims':[fact()]},'llm_only',FakeClient(True))
        self.assertEqual(result['claims'][0]['label'],'Supported');self.assertFalse(result['claims'][0]['reference_membership_valid'])
    def test_invalid_reference_separate_from_supported_verdict(self):
        result=classify({'description':'A超速','claims':[fact()],'evidence':[ev()]},'rag_verify',FakeClient(True))
        self.assertEqual(result['claims'][0]['label'],'Supported');self.assertEqual(result['claims'][0]['source_ids'],[]);self.assertTrue(result['claims'][0]['citation_warnings'])
    def test_rag_same_sources(self):
        payload={'description':'A超速','claims':[fact()]}
        a=classify(payload,'rag',FakeClient());b=classify(payload,'rag_verify',FakeClient())
        self.assertEqual(a['claims'][0]['retrieved'],b['claims'][0]['retrieved'])

class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler);cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.server.server_port}'
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def request(self,path,data=None,headers=None):
        req=urllib.request.Request(self.url+path,data=None if data is None else json.dumps(data).encode(),headers=headers or {'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=10) as r:return r.status,r.read()
        except urllib.error.HTTPError as e:return e.code,e.read()
    def test_health(self):self.assertEqual(json.loads(self.request('/api/health')[1])['documents'],22)
    def test_analyze(self):
        code,body=self.request('/api/analyze',read_json('demo.json'));self.assertEqual(code,200);self.assertEqual(json.loads(body)['engine'],'offline_rules_NOT_LLM')
    def test_invalid_body(self):self.assertEqual(self.request('/api/analyze',[])[0],400)
    def test_cross_origin(self):self.assertEqual(self.request('/api/analyze',read_json('demo.json'),{'Content-Type':'application/json','Origin':'https://example.com'})[0],403)
    def test_wrong_content_type(self):self.assertEqual(self.request('/api/analyze',{}, {'Content-Type':'text/plain'})[0],415)
    def test_path_traversal(self):self.assertEqual(self.request('/../../data/cases.json')[0],404)
    def test_page(self):self.assertIn('先核對證據'.encode(),self.request('/')[1])

if __name__=='__main__':unittest.main()
