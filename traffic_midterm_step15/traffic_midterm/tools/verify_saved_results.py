"""Audit saved artifacts without contacting Ollama or the internet."""
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from traffic.core import Retriever, read_json
from traffic.evaluate import classification_metrics, retrieval_metrics

def audit():
    manifest_path=ROOT/'MANIFEST.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf8'))
        for name,expected in manifest['sha256'].items():
            path=ROOT/name
            if not path.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Manifest path outside package')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Changed file: '+name)
    runs=json.loads((ROOT/'results/step13_comparison/runs.json').read_text(encoding='utf8'))
    summary=json.loads((ROOT/'results/step13_comparison/summary.json').read_text(encoding='utf8'))
    ref_path=ROOT/'data/reference_annotations.json'
    ref=json.loads(ref_path.read_text(encoding='utf8'))
    refs={c['id']:c for c in ref['cases']}
    cases={c['id']:c for c in read_json('cases.json') if c['split']=='test'}
    retriever=Retriever()
    assert hashlib.sha256(ref_path.read_bytes()).hexdigest()==summary['metadata']['reference_sha256']
    assert hashlib.sha256((ROOT/'data/cases.json').read_bytes()).hexdigest()==summary['metadata']['cases_sha256']
    assert retriever.fingerprint==summary['metadata']['corpus_sha256']
    assert len(runs)==30
    final_labels={}
    for mode in ('llm_only','rag','rag_verify'):
        rows=[r for r in runs if r['mode']==mode]
        assert len(rows)==10 and {r['case_id'] for r in rows}==set(cases)
        gold=[];pred=[]
        for row in rows:
            assert row['status']=='ok'
            assert {c['id'] for c in row['result']['claims']}==set(refs[row['case_id']]['labels'])
            for claim in row['result']['claims']:
                gold.append(refs[row['case_id']]['labels'][claim['id']]);pred.append(claim['label'])
                if mode=='rag_verify':final_labels[(row['case_id'],claim['id'])]=claim['label']
        for key,value in classification_metrics(gold,pred).items():assert value==summary['classification'][mode][key]
    for case_id in cases:
        a=next(r['result'] for r in runs if r['mode']=='rag' and r['case_id']==case_id)
        b=next(r['result'] for r in runs if r['mode']=='rag_verify' and r['case_id']==case_id)
        assert [c['retrieved'] for c in a['claims']]==[c['retrieved'] for c in b['claims']]
    for kind in ('law','case'):
        available={d['id'] for d in retriever.documents if d['kind']==kind};metrics=[]
        for case_id,case in cases.items():
            relevant=set(refs[case_id]['relevant_document_ids'])&available
            query=case['description']+' '+' '.join(c['text'] for c in case['claims'])
            metric=retrieval_metrics(relevant,[d['id'] for d in retriever.dual(query)[kind]],3)
            if metric['precision_at_k'] is not None:metrics.append(metric)
        assert len(metrics)==summary['retrieval'][kind]['n_queries']
        for key in ('precision_at_k','recall_at_k'):
            assert math.isclose(sum(m[key] for m in metrics)/len(metrics),summary['retrieval'][kind][key])
    baseline=json.loads((ROOT/'results/step15_rules_baseline/runs.json').read_text(encoding='utf8'))
    assert len(baseline)==10
    for row in baseline:
        assert row['status']=='ok' and row['mode']=='offline_rules'
        for claim in row['result']['claims']:
            assert claim['label']==final_labels[(row['case_id'],claim['id'])]
    print('SAVED_RESULTS_VERIFIED: 30 model case-runs, 15 claims per condition, matching rule-only baseline.')
    print('No Ollama call or internet request was made.')

if __name__=='__main__':audit()
