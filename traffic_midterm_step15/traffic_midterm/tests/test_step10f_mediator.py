import unittest
from traffic.core import mediation

class MediatorConflict(unittest.TestCase):
    def claim(self,**extra):
        return dict(id='C1',text='B超速',kind='fact',label='Insufficient Evidence',**extra)
    def test_nested_opposite(self):
        r=mediation([self.claim(gate={'conflict':True,'contradicting_evidence_ids':['E1']})])
        self.assertIn('相反',r['text']);self.assertIn('E1',r['text']);self.assertNotIn('尚需補充',r['text'])
    def test_flat_conflict(self):
        self.assertIn('證據衝突',mediation([self.claim(conflict=True)])['text'])
    def test_opposite_without_flag(self):
        self.assertIn('E2',mediation([self.claim(contradicting_evidence_ids=['E2'])])['text'])
    def test_no_conflict_still_missing(self):
        self.assertIn('尚需補充',mediation([self.claim()])['text'])
    def test_supported_and_liability_unchanged(self):
        for label,word in [('Supported','輸出標示支持'),('Overgeneralized','責任推論')]:
            c=self.claim();c.update(label=label,kind='liability')
            self.assertIn(word,mediation([c])['text'])
