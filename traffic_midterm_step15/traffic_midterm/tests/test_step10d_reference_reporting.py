import copy
import unittest
from traffic.extraction import validate_extraction
from traffic.llm import reference_issues, reference_error_message, classify
from test_step9_extraction import extracted,DESCRIPTION
from traffic.core import Retriever

class ReferenceReporting(unittest.TestCase):
    def test_causal_question_targets_behavior_not_reporting(self):
        r=validate_extraction(extracted(),DESCRIPTION)
        self.assertEqual(r['claims'][1]['text'],'A無照是否與事故發生有因果關係？')
        self.assertEqual(r['claims'][1]['based_on_quote'],'對方說我無照')
    def test_other_subject_behavior(self):
        value=extracted();value['case_fields']=[];value['claims'][0].update(text='B超速',subject='B',predicate='speeding',source_quote='我說對方超速');value['claims'][1].update(subject='B',source_quote='對方應負全責')
        r=validate_extraction(value,'我說對方超速，對方應負全責。')
        self.assertEqual(r['claims'][1]['text'],'B超速是否與事故發生有因果關係？')
    def test_actual_wrong_bucket_is_not_called_missing(self):
        row={'source_ids':['L217'],'evidence_ids':['L217']}
        issues=reference_issues(row,{'L217'},set(),set())
        self.assertEqual(issues,[{'field':'evidence_ids','id':'L217','category':'retrieval_source_used_as_case_evidence'}])
        self.assertNotIn('未提供',reference_error_message(issues))
    def test_unknown_id(self):
        issues=reference_issues({'source_ids':['MISSING'],'evidence_ids':[]},{'L217'},set(),set())
        self.assertEqual(issues[0]['category'],'id_not_provided')
    def test_unreviewed_evidence(self):
        issues=reference_issues({'source_ids':[],'evidence_ids':['E1']},set(),set(),{'E1'})
        self.assertEqual(issues[0]['category'],'unreviewed_case_evidence')
    def test_reversed_bucket(self):
        issues=reference_issues({'source_ids':['E1'],'evidence_ids':[]},set(),{'E1'},{'E1'})
        self.assertEqual(issues[0]['category'],'case_evidence_used_as_retrieval_source')
    def test_valid_refs(self):
        self.assertEqual(reference_issues({'source_ids':['L1'],'evidence_ids':['E1']},{'L1'},{'E1'},{'E1'}),[])

if __name__=='__main__':unittest.main()
