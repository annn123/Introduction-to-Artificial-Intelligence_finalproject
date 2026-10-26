import copy
import json
import tempfile
import unittest
from pathlib import Path
from traffic.core import ROOT, Retriever, read_json
from traffic.evaluate import load_reference, run

class ReferenceAnnotations(unittest.TestCase):
    def setUp(self):
        self.path=ROOT/'data/reference_annotations.json'
        self.value=json.loads(self.path.read_text(encoding='utf8'))
        self.cases=[c for c in read_json('cases.json') if c['split']=='test']
        self.retriever=Retriever()

    def test_complete_reference(self):
        data=load_reference(self.path,self.cases,self.retriever)
        self.assertEqual(len(data['cases']),10)
        self.assertEqual(sum(len(c['labels']) for c in data['cases']),15)
        self.assertFalse(data['provenance']['two_reviewer_consensus_recorded'])
        self.assertEqual(data['provenance']['classification_initial_agreement']['same'],11)

    def test_reject_bad_identifiers_or_versions(self):
        for field in ('duplicate_case','claim_id','source_id','cases_sha256','corpus_sha256'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as td:
                value=copy.deepcopy(self.value)
                if field=='duplicate_case':value['cases'].append(value['cases'][0])
                elif field=='claim_id':value['cases'][0]['labels']['FAKE']='Supported'
                elif field=='source_id':value['cases'][0]['relevant_document_ids']=['FAKE']
                else:value[field]='wrong'
                path=Path(td)/'reference.json';path.write_text(json.dumps(value),encoding='utf8')
                with self.assertRaises(ValueError):load_reference(path,self.cases,self.retriever)

    def test_offline_metadata_keeps_ai_assistance_visible(self):
        with tempfile.TemporaryDirectory() as td:
            result=run(td,backend='offline',split='test',gold_path=self.path)
            self.assertEqual(result['metadata']['annotation'],'TWO_INITIAL_REVIEWERS_AI_ASSISTED_USER_ADOPTED')
            self.assertIn('not an isolated test',result['metadata']['verification_decision_policy'])
            self.assertEqual(result['classification']['offline_rules']['n_claims'],15)
            self.assertEqual(result['metadata']['llm_experiment_status'],'NOT_RUN')

    def test_retrieval_uses_classification_query(self):
        with tempfile.TemporaryDirectory() as td:
            result=run(td,backend='offline',split='test',gold_path=self.path)
        relevant={c['id']:c['relevant_document_ids'] for c in self.value['cases']}
        for kind in ('law','case'):
            available={d['id'] for d in self.retriever.documents if d['kind']==kind}
            expected=[]
            for case in self.cases:
                rel=set(relevant[case['id']]) & available
                if not rel:continue
                query=case['description']+' '+' '.join(c['text'] for c in case['claims'])
                ids=[d['id'] for d in self.retriever.dual(query)[kind]][:3]
                expected.append(len(rel.intersection(ids))/3)
            self.assertEqual(result['retrieval'][kind]['n_queries'],len(expected))
            self.assertAlmostEqual(result['retrieval'][kind]['precision_at_k'],sum(expected)/len(expected))

if __name__=='__main__':unittest.main()
