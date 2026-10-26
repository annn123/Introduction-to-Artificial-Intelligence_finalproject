import copy
import unittest
from traffic.extraction import case_schema, claim_schema, flatten_staged, validate_extraction
from traffic.llm import ModelError

TEXT='我騎機車直行，對方開汽車左轉。對方說我無照，所以我應負全部責任。我有行車紀錄器。'

def fields():
    slots={key:{'value':'','source_quote':''} for key in case_schema(TEXT)['properties']['case_fields']['required']}
    for key,value,quote in [('A.vehicle','機車','我騎機車直行'),('B.vehicle','汽車','對方開汽車左轉'),('A.action','直行','我騎機車直行'),('B.action','左轉','對方開汽車左轉')]:slots[key]={'value':value,'source_quote':quote}
    return {'case_fields':slots}

def groups():return {'claim_groups':{
    'fact':[{'text':'A無照','subject':'A','predicate':'unlicensed','polarity':True,'source_quote':'對方說我無照','speaker':'B','premise_indices':[]}],
    'causal':[],
    'liability':[{'text':'A應負全部責任','subject':'A','predicate':'liability:100','polarity':True,'source_quote':'所以我應負全部責任','speaker':'B','premise_indices':[1]}],
    'irrelevant':[]}}

class StagedTests(unittest.TestCase):
    def test_no_road_word_means_only_unknown_road_slot(self):
        slot=case_schema(TEXT)['properties']['case_fields']['properties']['road_type']['properties']
        self.assertEqual(slot['value']['enum'],['']);self.assertEqual(slot['source_quote']['enum'],[''])
    def test_vehicle_cannot_be_unlicensed(self):
        slot=case_schema(TEXT)['properties']['case_fields']['properties']['A.vehicle']['properties']['value']['enum']
        self.assertIn('機車',slot);self.assertNotIn('無照',slot)
    def test_quote_choices_are_source_substrings(self):
        for slot in case_schema(TEXT)['properties']['case_fields']['properties'].values():
            for quote in slot['properties']['source_quote']['enum']:
                self.assertTrue(quote=='' or quote in TEXT)
    def test_causal_schema_cannot_output_liability_predicate(self):
        row=claim_schema(TEXT)['properties']['claim_groups']['properties']['causal']['items']
        self.assertNotIn('liability:100',row['properties']['predicate']['enum']);self.assertNotIn('kind',row['properties'])
    def test_unmentioned_predicates_not_available_as_known_facts(self):
        pred=claim_schema(TEXT)['properties']['claim_groups']['properties']['fact']['items']['properties']['predicate']['enum']
        self.assertEqual(pred,['unlicensed','unknown'])
    def test_stages_produce_three_levels_without_fabricating_a_causal_assertion(self):
        r=validate_extraction(flatten_staged(fields(),groups()),TEXT)
        self.assertEqual([c['kind'] for c in r['claims']],['fact','causal','liability'])
        self.assertEqual(r['claims'][1]['origin'],'generated_check_question')
    def test_liability_false_is_not_a_way_to_express_disagreement(self):
        g=groups();g['claim_groups']['liability'][0]['polarity']=False
        with self.assertRaises(ModelError):validate_extraction(flatten_staged(fields(),g),TEXT)
    def test_missing_case_slots_rejected(self):
        f=fields();del f['case_fields']['road_type']
        with self.assertRaises(ModelError):flatten_staged(f,groups())
    def test_missing_claim_group_rejected(self):
        g=groups();del g['claim_groups']['causal']
        with self.assertRaises(ModelError):flatten_staged(fields(),g)
    def test_unknown_case_slot_is_not_an_asserted_fact(self):
        r=validate_extraction(flatten_staged(fields(),groups()),TEXT)
        self.assertIsNone(r['case']['road_type']);self.assertEqual(r['evidence'],[])
    def test_actual_observed_fault_is_rejected_instead_of_silently_fixed(self):
        old={'case_fields':[{'field':'road_type','value':'直行、左轉','source_quote':'我騎機車直行，對方開汽車左轉。'}],
             'claims':[{'text':'A負全部責任','subject':'A','kind':'causal','predicate':'liability:100','polarity':False,'source_quote':'對方說我無照，所以我應負全部責任。','speaker':'B','premise_indices':[]}]}
        with self.assertRaises(ModelError):validate_extraction(old,TEXT)
    def test_claim_quote_must_contain_the_behavior(self):
        g=groups();g['claim_groups']['fact'][0]['source_quote']='我騎機車直行'
        with self.assertRaises(ModelError):validate_extraction(flatten_staged(fields(),g),TEXT)
    def test_inputs_remain_unchanged(self):
        f=fields();g=groups();original=copy.deepcopy((f,g));flatten_staged(f,g);self.assertEqual((f,g),original)

if __name__=='__main__':unittest.main()
