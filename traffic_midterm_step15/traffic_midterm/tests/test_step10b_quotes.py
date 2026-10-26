import unittest
from traffic.extraction import claim_schema

class QuoteConstraints(unittest.TestCase):
    def groups(self,text):return claim_schema(text)['properties']['claim_groups']['properties']
    def quotes(self,g,kind):return g[kind]['items']['properties']['source_quote'].get('enum',[])
    def test_wrong_liability_quote_is_no_longer_available(self):
        text='我騎機車直行，對方開汽車左轉。對方說我無照，所以我應負全部責任。我有行車紀錄器。'
        g=self.groups(text)
        self.assertNotIn('我騎機車直行，對方開汽車左轉',self.quotes(g,'liability'))
        self.assertIn('所以我應負全部責任',self.quotes(g,'liability'))
        self.assertEqual(self.quotes(g,'fact'),['對方說我無照'])
        self.assertEqual(g['causal']['maxItems'],0)
    def test_reverse_actor_and_different_violation(self):
        g=self.groups('我說對方超速，所以對方應全責。')
        self.assertEqual(self.quotes(g,'fact'),['我說對方超速'])
        self.assertIn('所以對方應全責',self.quotes(g,'liability'))
    def test_absent_responsibility_means_empty_group(self):
        self.assertEqual(self.groups('對方闖紅燈。')['liability']['maxItems'],0)
    def test_cause_quote_is_source_and_not_just_a_fact(self):
        self.assertEqual(self.quotes(self.groups('對方超速。超速導致碰撞。'),'causal'),['超速導致碰撞'])
    def test_generic_fact_still_available(self):
        self.assertIn('我在停車場',self.quotes(self.groups('我在停車場。'),'fact'))

if __name__=='__main__':unittest.main()
