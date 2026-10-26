import copy
import json
import unittest
from traffic.claim_candidates import candidates,review_schema,assemble
from traffic.extraction import extract_case
from traffic.llm import ModelError
from test_step9_extraction import FakeClient,DESCRIPTION

def review(rows):
    return {'candidate_review':{key:{field:spec['enum'][0] for field,spec in slot['properties'].items()} for key,slot in review_schema(rows)['properties']['candidate_review']['properties'].items()}}

class CandidatesTests(unittest.TestCase):
    def test_observed_failure_has_no_free_text_or_premise_output(self):
        rows=candidates(DESCRIPTION);schema=review_schema(rows)
        self.assertEqual(len(rows),2)
        for slot in schema['properties']['candidate_review']['properties'].values():
            self.assertEqual(set(slot['properties']),{'subject','speaker','polarity'})
        claims=assemble(rows,review(rows))
        self.assertEqual([c['subject'] for c in claims],['A','A'])
        self.assertEqual([c['speaker'] for c in claims],['B','B'])
        self.assertEqual(claims[0]['predicate'],'unlicensed')
        self.assertEqual(claims[0]['premise_indices'],[])
        self.assertEqual(claims[1]['premise_indices'],[1])
    def test_reversed_roles_other_topic(self):
        rows=candidates('我說對方超速，所以對方應全責。');claims=assemble(rows,review(rows))
        self.assertEqual([(c['subject'],c['speaker']) for c in claims],[('B','A'),('B','A')])
        self.assertEqual(claims[0]['predicate'],'speeding')
    def test_negative_claim(self):
        rows=candidates('對方說我沒有超速。')
        self.assertFalse(rows[0]['polarity'])
    def test_negative_violation_word_is_positive_allegation(self):
        self.assertTrue(candidates('我說對方未禮讓。')[0]['polarity'])
    def test_negative_liability(self):
        self.assertFalse(candidates('我不應負全部責任。')[0]['polarity'])
    def test_no_link_across_sentences(self):
        rows=candidates('對方說我無照。所以我應負全部責任。')
        self.assertEqual(rows[1]['premise_indices'],[])
        self.assertEqual(rows[1]['speaker'],'A')
    def test_complex_clause_flagged_not_fabricated(self):
        self.assertEqual(candidates('對方說我無照所以應全責。'),[{'unparsed':'對方說我無照所以應全責'}])
    def test_unknown_subject_stays_unknown(self):
        self.assertEqual(candidates('有人超速。')[0]['subject'],'unknown')
    def test_bad_model_role_rejected(self):
        rows=candidates(DESCRIPTION);r=review(rows);r['candidate_review']['K1']['subject']='B'
        with self.assertRaises(ModelError):assemble(rows,r)
    def test_extra_model_text_rejected(self):
        rows=candidates(DESCRIPTION);r=review(rows);r['candidate_review']['K1']['text']='我騎機車直行'
        with self.assertRaises(ModelError):assemble(rows,r)
    def test_missing_candidate_rejected(self):
        rows=candidates(DESCRIPTION);r=review(rows);r['candidate_review'].pop('K2')
        with self.assertRaises(ModelError):assemble(rows,r)
    def test_multiple_premises_do_not_guess_last_fact(self):
        rows=candidates('我無照，對方超速，所以我應全責。')
        self.assertEqual(rows[-1]['premise_indices'],[])
    def test_different_subject_does_not_get_wrong_premise(self):
        rows=candidates('我無照，所以對方應全責。')
        self.assertEqual(rows[-1]['premise_indices'],[])
    def test_conditional_is_not_asserted_as_a_fact(self):
        self.assertIn('unparsed',candidates('如果我無照。')[0])
    def test_third_party_reporting_is_flagged(self):
        self.assertIn('unparsed',candidates('警方說我超速。')[0])
    def test_input_not_mutated(self):
        rows=candidates(DESCRIPTION);r=review(rows);old=copy.deepcopy((rows,r));assemble(rows,r);self.assertEqual((rows,r),old)
    def test_end_to_end_transport_stub_method_and_evidence(self):
        r=extract_case(DESCRIPTION,FakeClient())
        self.assertEqual(r['extraction_method'],'hybrid_rules_with_local_role_review')
        self.assertEqual([c['kind'] for c in r['claims']],['fact','causal','liability'])
        self.assertEqual(r['evidence'],[])
        self.assertFalse(r['role_model_called'])
        self.assertEqual(len(r['calls']),1)
        self.assertEqual(r['claims'][2]['required_checks'],['C2'])
        self.assertEqual(r['claims'][2]['speaker'],'B')
    def test_unrecognised_description_requests_manual_input(self):
        with self.assertRaises(ModelError):extract_case('這是一段不明確的描述。',FakeClient())

if __name__=='__main__':unittest.main()
