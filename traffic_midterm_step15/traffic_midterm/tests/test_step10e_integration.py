import copy
import unittest
from traffic.llm import classify

class Client:
    model='mock'
    calls=[]
    def generate(self,prompt,schema):
        return {'results':{key:{'label':'Insufficient Evidence','reason':'original','evidence_ids':['FAKE'],'source_ids':[]} for key in schema['properties']['results']['properties']}}

def payload():
    return {'description':'B超速，所以B應負全部責任','claims':[{'id':'C1','text':'B超速','subject':'B','kind':'fact','predicate':'speeding','polarity':True}], 'evidence':[{'id':'E1','description':'合成資料','reviewed':True,'source_type':'transcript','assertions':[{'subject':'B','predicate':'speeding','polarity':True}]}]}

class Integration(unittest.TestCase):
    def test_support_and_citation_separation(self):
        r=classify(payload(),'rag_verify',Client())['claims'][0]
        self.assertEqual(r['label'],'Supported');self.assertEqual(r['evidence_ids'],['E1']);self.assertTrue(r['citation_warnings']);self.assertEqual(r['raw_model_result']['label'],'Insufficient Evidence')
    def test_evidence_boundaries(self):
        for variant in ('unreviewed','opposite','conflict','subject','causal'):
            with self.subTest(variant=variant):
                p=payload()
                if variant=='unreviewed':p['evidence'][0]['reviewed']=False
                if variant=='opposite':p['evidence'][0]['assertions'][0]['polarity']=False
                if variant=='conflict':
                    e=copy.deepcopy(p['evidence'][0]);e['id']='E2';e['assertions'][0]['polarity']=False;p['evidence'].append(e)
                if variant=='subject':p['evidence'][0]['assertions'][0]['subject']='A'
                if variant=='causal':p['claims'][0].update(kind='causal',predicate='cause:speeding')
                self.assertEqual(classify(p,'rag_verify',Client())['claims'][0]['label'],'Insufficient Evidence')
    def test_liability_and_baselines(self):
        p=payload();p['claims'][0].update(kind='liability',predicate='liability:100',text='所以B應負全部責任')
        self.assertEqual(classify(p,'rag_verify',Client())['claims'][0]['label'],'Overgeneralized')
        for mode in ('rag','llm_only'):
            r=classify(payload(),mode,Client())['claims'][0];self.assertEqual(r['label'],'Insufficient Evidence');self.assertNotIn('gate',r)
