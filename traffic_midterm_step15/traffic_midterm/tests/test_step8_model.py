import unittest
from unittest.mock import patch
from traffic.llm import DEFAULT_MODEL, Ollama, ModelError
from traffic.core import ROOT

class Step8Tests(unittest.TestCase):
    def test_default_is_qwen25(self):self.assertEqual(Ollama().model, 'qwen2.5:3b-instruct')
    def test_no_think_parameter_for_qwen25(self):
        c=Ollama()
        with patch.object(c,'_request',return_value={'done':True,'done_reason':'stop','response':'{"status":"ok"}'}) as req:
            self.assertEqual(c.generate('JSON',{},max_tokens=64),{'status':'ok'})
            self.assertNotIn('think',req.call_args.args[1]);self.assertEqual(req.call_args.args[1]['options']['num_predict'],64)
    def test_legacy_qwen3_parameter_preserved(self):
        c=Ollama('qwen3:4b')
        with patch.object(c,'_request',return_value={'done':True,'response':'{}'}) as req:
            c.generate('JSON',{});self.assertIs(req.call_args.args[1]['think'],False)
    def test_thinking_not_silently_removed(self):
        for response in [{'done':True,'response':'{}','thinking':'reasoning'}, {'done':True,'response':'<think>x</think>{}'}]:
            c=Ollama()
            with self.subTest(response=response),patch.object(c,'_request',return_value=response):
                with self.assertRaises(ModelError):c.generate('JSON',{})
                self.assertEqual(len(c.calls),1)
    def test_token_limit_rejected(self):
        with self.assertRaises(ValueError):Ollama().generate('JSON',{},max_tokens=0)
    def test_smoke_requires_exact_output(self):
        c=Ollama()
        with patch.object(c,'status',return_value={'model':DEFAULT_MODEL}),patch.object(c,'_request',return_value={'done':True,'done_reason':'stop','response':'{"status":"ok"}'}) as req:
            r=c.smoke_test();self.assertEqual(r['structured_output_test'],'passed');self.assertEqual(req.call_args.args[1]['options']['num_predict'],64)
    def test_smoke_rejects_wrong_json(self):
        c=Ollama()
        with patch.object(c,'status',return_value={}),patch.object(c,'_request',return_value={'done':True,'response':'{"status":"wrong"}'}):
            with self.assertRaises(ModelError):c.smoke_test()
    def test_ui_default(self):
        html=(ROOT/'static/index.html').read_text(encoding='utf8')
        self.assertIn('<select id="model"><option>qwen2.5:3b-instruct</option>',html)

if __name__=='__main__':unittest.main()
